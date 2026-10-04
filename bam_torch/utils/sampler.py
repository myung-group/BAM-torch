import math
import random
from typing import Iterator, List, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import Sampler
import torch.distributed as dist
from torch_geometric.data import Batch as DataBatch


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

def _data_size(data, reference: str = 'nodes') -> int:
    if reference == 'edges':
        return int(data.num_edges)
    return int(data.num_nodes)


def create_size_buckets_tg(
    dataset: Sequence,
    n_buckets: int = 8,
    reference: str = 'nodes',
    seed: int = 0,
) -> List[List[int]]:
    """Randomly assign torch_geometric Data indices into n_buckets."""
    rng = np.random.RandomState(seed)
    indices = rng.permutation(len(dataset)).tolist()
    n = len(indices)
    return [
        indices[i * n // n_buckets : (i + 1) * n // n_buckets]
        for i in range(n_buckets)
    ]


class BucketedDataLoader:
    """DataLoader that batches torch_geometric Data objects of similar sizes.

    Analogue of :class:`bam_torch.data.BucketedDataLoader`, but for
    ``torch_geometric.data.Data`` objects (used by ``MPTrainer``). Yields
    batched ``torch_geometric.data.Batch`` instances built via
    ``Batch.from_data_list``.

    For distributed training, set ``rank`` and ``world_size`` so that each
    process receives a disjoint shard of graphs. Call :meth:`set_epoch` at
    the start of every epoch to re-seed the shuffle.
    """

    def __init__(
        self,
        dataset: Sequence,
        batch_size: int = 20,
        n_buckets: int = 8,
        shuffle: bool = True,
        drop_last: bool = False,
        max_edges_per_batch: int = 16384, #Optional[int] = None,
        seed: int = 0,
        reference: str = 'edges',
        rank: int = 0,
        world_size: int = 1,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.n_buckets = n_buckets
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.max_edges_per_batch = max_edges_per_batch
        self.reference = reference
        self.seed = seed
        self.epoch = 0
        self.rank = rank
        self.world_size = world_size
        self.rng = np.random.RandomState(seed)

        self.buckets = create_size_buckets_tg(dataset, n_buckets, reference, seed)
        self._compute_epoch_buckets()

    def _compute_epoch_buckets(self) -> None:
        """Build buckets for the current epoch using current rng state."""
        if self.shuffle:
            all_indices = self.rng.permutation(len(self.dataset)).tolist()
            n = len(all_indices)
            self._epoch_buckets = [
                all_indices[i * n // self.n_buckets : (i + 1) * n // self.n_buckets]
                for i in range(self.n_buckets)
            ]
        else:
            self._epoch_buckets = [list(b) for b in self.buckets]
        self._epoch_bucket_sizes = [
            [_data_size(self.dataset[i], self.reference) for i in b]
            for b in self._epoch_buckets
        ]

    def set_epoch(self, epoch: int) -> None:
        """Set epoch to reseed shuffling each epoch across all ranks."""
        self.epoch = epoch
        self.rng = np.random.RandomState(self.seed + epoch)
        self._compute_epoch_buckets()

    def __iter__(self) -> Iterator[DataBatch]:
        # Materialize all batches first so we can sync the batch count
        # across ranks. With max_edges_per_batch, different ranks pack
        # different numbers of batches from their stride-sliced shards;
        # DDP collectives (gradient all-reduce in backward, plus any
        # explicit all_reduce/barrier in the train loop) would deadlock
        # when ranks iterate unequal numbers of times.
        all_batches: List[List] = []
        for bucket in self._epoch_buckets:
            if self.world_size > 1:
                bucket = bucket[self.rank::self.world_size]

            pos = 0
            while pos < len(bucket):
                remaining = len(bucket) - pos
                if self.drop_last and remaining < self.batch_size:
                    break
                n_take = min(self.batch_size, remaining)
                graphs = []
                n_nodes = 0
                for j in range(n_take):
                    g = self.dataset[bucket[pos + j]]
                    n_nodes += _data_size(g, self.reference)
                    if (self.max_edges_per_batch is not None
                            and n_nodes >= self.max_edges_per_batch
                            and graphs):
                        break
                    graphs.append(g)
                if not graphs:
                    break
                pos += len(graphs)
                all_batches.append(graphs)

        # DDP with the default broadcast_buffers=True fires a collective on
        # every forward (sync_module_buffers), so valid also needs ranks to
        # take the same number of forward steps. Trim to the global min batch
        # count for both train and valid; the alternative is an NCCL hang.
        if (
            self.world_size > 1
            and dist.is_available()
            and dist.is_initialized()
        ):
            backend = dist.get_backend()
            device = (
                torch.device("cuda", torch.cuda.current_device())
                if backend == "nccl"
                else torch.device("cpu")
            )
            n = torch.tensor(len(all_batches), dtype=torch.long, device=device)
            dist.all_reduce(n, op=dist.ReduceOp.MIN)
            all_batches = all_batches[: int(n.item())]

        for graphs in all_batches:
            yield DataBatch.from_data_list(graphs)

    def __len__(self) -> int:
        total = 0
        for b, sizes in zip(self._epoch_buckets, self._epoch_bucket_sizes):
            if not b:
                continue
            if self.world_size > 1:
                shard = b[self.rank::self.world_size]
                shard_sizes = sizes[self.rank::self.world_size]
            else:
                shard = b
                shard_sizes = sizes
            n = len(shard)
            if n == 0:
                continue
            if self.drop_last:
                graph_cap_batches = n // self.batch_size
            else:
                graph_cap_batches = (n + self.batch_size - 1) // self.batch_size
            if self.max_edges_per_batch is not None and shard_sizes:
                # Graphs whose size already exceeds the cap each occupy a
                # solo batch; only the rest can pack together.
                solo = sum(1 for s in shard_sizes if s >= self.max_edges_per_batch)
                packable_total = sum(s for s in shard_sizes if s < self.max_edges_per_batch)
                packed_batches = (
                    (packable_total + self.max_edges_per_batch - 1)
                    // self.max_edges_per_batch
                ) if packable_total > 0 else 0
                edge_cap_batches = solo + packed_batches
                total += max(graph_cap_batches, edge_cap_batches)
            else:
                total += graph_cap_batches
        return total
