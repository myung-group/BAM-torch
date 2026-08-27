"""Shard a CG npz into graph pkl shards for mp-style streaming training.

Usage:
  python -m bam_torch.coarse_grain.shard_cg --npz X.npz --out shards_dir \
      --cutoff 12.0 --shard-size 5000 --nvalid 500 --ntest 500 --beads-per-mol 2 --bonds '[[0,1]]'
"""
import argparse, os, pickle, json
import numpy as np
import torch
from bam_torch.utils.cg_dataset import CGDataset
from bam_torch.utils.utils import get_graphset_cg, get_cg_enr_avg_per_type


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--npz', required=True)
    ap.add_argument('--out', required=True, help='output shard directory')
    ap.add_argument('--cutoff', type=float, required=True)
    ap.add_argument('--shard-size', type=int, default=5000, help='frames per shard')
    ap.add_argument('--nvalid', type=int, default=500)
    ap.add_argument('--ntest', type=int, default=0, help='held-out test frames (train:valid:test)')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--beads-per-mol', type=int, default=None)
    ap.add_argument('--bonds', type=str, default=None, help="JSON bonds e.g. '[[0,1]]'")
    ap.add_argument('--type-remap', type=str, default=None,
                    help="comma-separated new type per old type, e.g. '0,1,1,0' merges symmetric beads")
    args = ap.parse_args()

    ds = CGDataset(args.npz)
    if args.type_remap:
        remap = [int(x) for x in args.type_remap.split(',')]
        ds.types = np.array([remap[int(t)] for t in ds.types], dtype=ds.types.dtype)
        print(f"applied type-remap {remap}: types now {sorted(set(ds.types.tolist()))}")
    n = len(ds)
    torch.manual_seed(args.seed)
    idx = torch.randperm(n).tolist()
    test_idx = idx[:args.ntest]
    valid_idx = idx[args.ntest:args.ntest + args.nvalid]
    train_idx = idx[args.ntest + args.nvalid:]
    num_cg_types = int(np.max(ds.types)) + 1

    bt = None
    if args.beads_per_mol and args.bonds:
        bt = {'n_beads_per_mol': args.beads_per_mol, 'bonds': json.loads(args.bonds)}

    os.makedirs(args.out, exist_ok=True)
    sample = ds.get_subset(train_idx[:min(2000, len(train_idx))]) + ds.get_subset(valid_idx)
    enr_avg, uniq_type, enr_var = get_cg_enr_avg_per_type(sample, num_cg_types)

    def build(indices, prefix):
        files = []
        for si, s in enumerate(range(0, len(indices), args.shard_size)):
            chunk = indices[s:s + args.shard_size]
            data = ds.get_subset(chunk)
            g = get_graphset_cg(data, args.cutoff, uniq_type, enr_avg, enr_var, True, None,
                                show_progress=True, desc=f'{prefix} shard {si}', bond_topology=bt)
            fn = f'{prefix}_shard_{si:04d}.pkl'
            with open(os.path.join(args.out, fn), 'wb') as f:
                pickle.dump(g, f)
            files.append(fn)
            print(f'  saved {fn}: {len(g)} graphs')
        return files

    train_files = build(train_idx, 'train')
    valid_files = build(valid_idx, 'valid')
    test_files = build(test_idx, 'test') if len(test_idx) else []

    manifest = {
        'npz': os.path.abspath(args.npz), 'cutoff': args.cutoff,
        'num_cg_types': num_cg_types,
        'uniq_type': {str(k): v for k, v in uniq_type.items()},
        'enr_avg_per_type': {str(k): float(v) for k, v in enr_avg.items()},
        'enr_var': float(enr_var),
        'n_train': len(train_idx), 'n_valid': len(valid_idx), 'n_test': len(test_idx),
        'shard_size': args.shard_size,
        'train_shards': train_files, 'valid_shards': valid_files, 'test_shards': test_files,
        'bond_topology': bt,
    }
    with open(os.path.join(args.out, 'manifest.json'), 'w') as f:
        json.dump(manifest, f, indent=2)
    print(f'manifest: {len(train_files)} train + {len(valid_files)} valid + {len(test_files)} test shards')


if __name__ == '__main__':
    main()
