"""
Multihead finetuning trainer for CG models (2026-08-26).

Wires the existing multihead machinery (RACEUnified heads + foundation
readout expansion from MultiheadTrainer) into the CG shard-streaming path:

- model: race_unified with heads=[dataset names], CG species embedding
  (num_cg_types), interaction_block passthrough; forces via autograd
  (matches CG uMLP foundation — no force decoders).
- foundation: NN.foundation_model pkl; trunk loaded by name, final readout
  replicated per head (MultiheadTrainer._load_from_state_dict reused).
- data: json["multihead"]["datasets"] = [{name, shard_dir,
  max_train_shards?, max_valid_shards?}, ...]. Shards are streamed
  round-robin-interleaved; every graph is tagged with its head index
  (RACEUnified routes readout per graph via data["head"]).
- v1 scope: equal loss weights (balance via max_*_shards), no delta/prior
  (plain force labels, same convention as the foundation pretraining).
"""
import os
import json
import pickle
import torch

from .cg_trainer import CGTrainer
from .multihead_trainer import MultiheadTrainer
from e3nn import o3


class MultiheadCGTrainer(CGTrainer):
    def __init__(self, json_data, rank=0, world_size=1):
        mh = json_data.get("multihead", {})
        if not mh.get("enabled", False):
            raise ValueError("multihead.enabled must be true for cg_multihead trainer")
        self.mh_datasets = mh.get("datasets", [])
        if len(self.mh_datasets) < 2:
            raise ValueError("multihead.datasets needs >=2 entries (target + replay)")
        self.heads = [d.get("name", f"head_{i}") for i, d in enumerate(self.mh_datasets)]
        self.num_heads = len(self.heads)
        if not json_data.get("NN", {}).get("foundation_model"):
            raise ValueError("NN.foundation_model is required for cg_multihead")
        json_data["model"] = "race_unified"
        super().__init__(json_data, rank, world_size)
        if self.delta_learning:
            raise ValueError("cg_multihead v1 does not support delta_learning")

    # ---------------- model ----------------
    def set_model(self):
        cfg = self.json_data
        num_cg_types = cfg.get("num_cg_types", 1)
        regress_forces = cfg.get("regress_forces", "auto")
        if regress_forces is True:
            regress_forces = "autograd"
        elif regress_forces is False:
            regress_forces = "false"

        from bam_torch.model.models import RACEUnified
        model = RACEUnified(
            cutoff=cfg.get("cutoff", 6.5),
            avg_num_neighbors=cfg.get("avg_num_neighbors", 30),
            num_species=num_cg_types,
            max_ell=cfg.get("max_ell", 3),
            num_basis_func=cfg.get("num_radial_basis", 8),
            hidden_irreps=o3.Irreps(cfg.get("hidden_channels", "32x0e+32x1o")),
            nlayers=cfg.get("nlayers", 2),
            features_dim=cfg.get("features_dim", 32),
            output_irreps=o3.Irreps(cfg.get("output_channels", "1x0e")),
            active_fn=cfg.get("active_fn", "identity"),
            regress_forces=regress_forces,
            cueq_config=None,
            interaction_block=cfg.get("interaction_block") or "slow",
            compute_stress=cfg.get("compute_stress", True),
            heads=self.heads,
        )
        if self.rank == 0:
            print(f"\nMultihead-CG Model: race_unified | heads {self.heads} | "
                  f"types {num_cg_types} | block {cfg.get('interaction_block') or 'slow'} | "
                  f"hidden {cfg.get('hidden_channels')}")
        self._mh_load_foundation(model)
        return model

    def _mh_load_foundation(self, model):
        path = self.json_data["NN"]["foundation_model"]
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        state = ckpt["params"]
        if self.rank == 0:
            print(f"Loading CG foundation: {path}")
        # Reuse the generic readout-expansion loader from MultiheadTrainer
        MultiheadTrainer._load_from_state_dict(self, model, state, self.num_heads)

    # ---------------- data ----------------
    def _configure_dataloader_from_shards(self, cg_cutoff, num_cg_types):
        per_head_tr, per_head_va = [], []
        for hi, d in enumerate(self.mh_datasets):
            with open(os.path.join(d["shard_dir"], "manifest.json")) as f:
                man = json.load(f)
            if man.get("delta_prior_applied", False):
                raise RuntimeError(f"dataset '{d.get('name')}' shards are "
                                   "delta-subtracted — unsupported in cg_multihead v1")
            tr = man["train_shards"]
            va = man["valid_shards"]
            if d.get("max_train_shards"):
                tr = tr[: int(d["max_train_shards"])]
            if d.get("max_valid_shards"):
                va = va[: int(d["max_valid_shards"])]
            per_head_tr.append([(os.path.join(d["shard_dir"], x), hi) for x in tr])
            per_head_va.append([(os.path.join(d["shard_dir"], x), hi) for x in va])
            if self.rank == 0:
                print(f"  [head {hi}:{self.heads[hi]}] train {len(tr)} shards, "
                      f"valid {len(va)} shards  ({d['shard_dir']})")
        # round-robin interleave so replay is spread across the epoch
        def interleave(lists):
            out, idx = [], [0] * len(lists)
            while any(idx[i] < len(lists[i]) for i in range(len(lists))):
                for i in range(len(lists)):
                    if idx[i] < len(lists[i]):
                        out.append(lists[i][idx[i]])
                        idx[i] += 1
            return out
        self.shard_mode = True
        self._train_shards = interleave(per_head_tr)
        self._valid_shards = interleave(per_head_va)
        self._shard_bt = None
        uniq_type = {i: i for i in range(num_cg_types)}
        enr_avg = {i: 0.0 for i in range(num_cg_types)}
        if self.rank == 0:
            print(f"\nData mode: multihead sharded streaming "
                  f"({len(self._train_shards)} train / {len(self._valid_shards)} valid shards)")
        return None, None, uniq_type, enr_avg

    def _load_shard_loader(self, shard_entry, shuffle):
        from torch_geometric.loader import DataLoader
        shard_path, head = shard_entry
        with open(shard_path, "rb") as f:
            graphset = pickle.load(f)
        pad_nodes = max(g.num_nodes for g in graphset)
        pad_edges = max(g.num_edges for g in graphset)
        for g in graphset:
            g.pad_nodes_to = pad_nodes
            g.pad_edges_to = pad_edges
            g.head = torch.tensor([head], dtype=torch.long)
        loader = DataLoader(graphset, batch_size=self.json_data["nbatch"],
                            shuffle=shuffle, drop_last=True)
        return graphset, loader
