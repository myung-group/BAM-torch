import math
import random
from typing import Iterator, Optional

import torch
from torch.utils.data import Sampler
import torch.distributed as dist


class DistributedBalancedAtomCountBatchSampler(Sampler[list[int]]):
    def __init__(
        self,
        dataset: list[int],
        batch_size: int,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False,
        reference: str = 'nodes'
    ):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        if rank >= num_replicas or rank < 0:
            raise ValueError(
                f"Invalid rank {rank}, rank should be in the interval [0, {num_replicas - 1}]"
            )
        self.dataset = dataset
        if reference == 'nodes':
            self.atom_counts = [data.num_nodes for data in dataset]
        elif reference == 'edges':
            self.atom_counts = [data.num_edges for data in dataset]
        self.batch_size = batch_size
        self.num_replicas = num_replicas
        self.rank = rank
        self.shuffle = shuffle
        self.seed = seed
        self.drop_last = drop_last
        self.epoch = 0
        # If the dataset length is evenly divisible by # of replicas, then there
        # is no need to drop any data, since the dataset will be split equally.
        if self.drop_last and len(self.dataset) % self.num_replicas != 0:  # type: ignore[arg-type]
            # Split to nearest available length that is evenly divisible.
            # This is to ensure each rank receives the same amount of data when
            # using this Sampler.
            self.num_samples = math.ceil(
                (len(self.dataset) - self.num_replicas) / self.num_replicas  # type: ignore[arg-type]
            )
        else:
            self.num_samples = math.ceil(len(self.dataset) / self.num_replicas)  # type: ignore[arg-type]
        self.total_size = self.num_samples * self.num_replicas

    def _generate_batches(self) -> list[list[int]]:
        indices = list(range(len(self.dataset)))
        if self.shuffle:
            g = random.Random(self.seed + self.epoch)
            g.shuffle(indices)

        indices.sort(key=lambda i: -self.atom_counts[i])
        #num_batches = len(indices) // self.batch_size 
        #if len(indices) % self.batch_size != 0:
        #    num_batches += 1
        num_batches = self.total_size
        batches = [[] for _ in range(num_batches)]
        total_atoms = [0] * num_batches

        for i in indices:
            candidate_batches = [j for j in range(len(batches)) if len(batches[j]) < self.batch_size]
            if not candidate_batches:
                break  
                
            min_batch = min(candidate_batches, key=lambda j: total_atoms[j])
            batches[min_batch].append(i)
            total_atoms[min_batch] += self.atom_counts[i]

        batches = [b for b in batches if b]
        if self.shuffle:
            random.shuffle(batches)
        return batches

    def __iter__(self) -> Iterator[list[int]]:
        all_batches = self._generate_batches()

        rank_batches = all_batches[self.rank :: self.num_replicas]
        if self.drop_last:
            rank_batches = [
                b for b in rank_batches if len(b) == self.batch_size
            ]
        indices = [x for sublist in rank_batches for x in sublist]
        #assert len(indices) == self.num_samples
        
        return iter(indices)

    def __len__(self) -> int:
        return math.ceil(len(self.atom_counts) / self.batch_size / self.num_replicas)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch
