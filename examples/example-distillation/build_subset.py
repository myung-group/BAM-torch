"""Stream-parse MPtrj JSON and build train/valid/test ASE traj splits.

Splits are deterministic by mp-id hash bucket so a structure (and all its
relaxation frames) is guaranteed never to leak across splits::

    bucket = int(hashlib.sha1(mp_id).hexdigest()[:8], 16) % 100
    0..79  -> train  (80%)
    80..94 -> valid  (15%)
    95..99 -> test   (5%)

For each mp-id we keep one mid-relaxation frame (forces are typically
non-trivial there). The walk stops when all three buckets reach their target
counts; ``--max-mp-ids`` caps how far we ever walk into the file.

Requires ``ijson`` (``pip install ijson``).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

import ijson
import numpy as np
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import write


def bucket_for(mp_id: str) -> int:
    return int(hashlib.sha1(mp_id.encode("utf-8")).hexdigest()[:8], 16) % 100


def split_for(mp_id: str) -> str:
    b = bucket_for(mp_id)
    if b < 80:
        return "train"
    if b < 95:
        return "valid"
    return "test"


def frame_to_atoms(frame: dict, mp_id: str, frame_id: str) -> Atoms:
    """Convert one MPtrj JSON frame to an ASE Atoms object.

    The default ``calc.results['energy']`` is set to the *uncorrected*
    raw VASP total energy, which is the convention BAM_MPtrj_v1 was trained
    on. The MP2020-corrected and per-atom energies are kept on
    ``atoms.info`` for downstream comparison.
    """
    struct = frame["structure"]
    species = [site["species"][0]["element"] for site in struct["sites"]]
    positions = [site["xyz"] for site in struct["sites"]]
    cell = struct["lattice"]["matrix"]
    atoms = Atoms(symbols=species, positions=positions, cell=cell, pbc=True)

    forces = np.asarray(frame["force"], dtype=float)
    atoms.info["mp_id"] = mp_id
    atoms.info["frame_id"] = frame_id
    atoms.info["uncorrected_total_energy"] = float(frame["uncorrected_total_energy"])
    atoms.info["corrected_total_energy"] = float(frame["corrected_total_energy"])
    atoms.info["energy_per_atom"] = float(frame["energy_per_atom"])

    atoms.calc = SinglePointCalculator(
        atoms,
        energy=float(frame["uncorrected_total_energy"]),
        forces=forces,
    )
    return atoms


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--json", required=True, help="path to MPtrj_2022.9_full.json")
    p.add_argument("--out-dir", required=True, help="output directory for train/valid/test trajs")
    p.add_argument("--n-train", type=int, default=50_000)
    p.add_argument("--n-valid", type=int, default=5_000)
    p.add_argument("--n-test", type=int, default=5_000)
    p.add_argument("--max-mp-ids", type=int, default=200_000)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not os.path.exists(args.json):
        sys.exit(f"MPtrj JSON not found: {args.json}")
    os.makedirs(args.out_dir, exist_ok=True)

    targets = {"train": args.n_train, "valid": args.n_valid, "test": args.n_test}
    samples: dict[str, list] = {"train": [], "valid": [], "test": []}
    mp_ids: dict[str, list[str]] = {"train": [], "valid": [], "test": []}

    print(f"[build_subset] streaming {args.json}")
    print(f"[build_subset] targets train={args.n_train} valid={args.n_valid} test={args.n_test}")

    with open(args.json, "rb") as fh:
        walked = 0
        for mp_id, frames in ijson.kvitems(fh, ""):
            walked += 1
            if walked > args.max_mp_ids:
                print(f"[build_subset] hit --max-mp-ids={args.max_mp_ids}, stopping")
                break

            split = split_for(mp_id)
            if len(samples[split]) >= targets[split]:
                if all(len(samples[s]) >= targets[s] for s in targets):
                    break
                continue

            frame_keys = list(frames.keys())
            mid = frame_keys[len(frame_keys) // 2]
            try:
                atoms = frame_to_atoms(frames[mid], mp_id, mid)
            except Exception as e:
                print(f"  skip {mp_id} {mid}: {e}")
                continue

            samples[split].append(atoms)
            mp_ids[split].append(mp_id)

            total = sum(len(samples[s]) for s in samples)
            if total % 1000 == 0:
                print(f"  walked={walked} train={len(samples['train'])} "
                      f"valid={len(samples['valid'])} test={len(samples['test'])}")

            if all(len(samples[s]) >= targets[s] for s in targets):
                break

    for split in ("train", "valid", "test"):
        out_traj = os.path.join(args.out_dir, f"{split}.traj")
        write(out_traj, samples[split])
        print(f"[build_subset] wrote {len(samples[split])} -> {out_traj}")

    with open(os.path.join(args.out_dir, "splits.json"), "w") as fh:
        json.dump(mp_ids, fh, indent=2)
    print(f"[build_subset] wrote {os.path.join(args.out_dir, 'splits.json')}")

    print("\n[build_subset] summary:")
    for split in ("train", "valid", "test"):
        atoms_list = samples[split]
        if not atoms_list:
            continue
        sizes = [len(a) for a in atoms_list]
        elems = set()
        for a in atoms_list:
            elems.update(a.get_chemical_symbols())
        print(f"  {split:5s}: n={len(atoms_list):6d}  "
              f"atoms/struct min={min(sizes)} median={sorted(sizes)[len(sizes)//2]} max={max(sizes)}  "
              f"distinct elements={len(elems)}")


if __name__ == "__main__":
    main()
