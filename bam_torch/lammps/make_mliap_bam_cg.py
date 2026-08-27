"""Export a trained CG model as a LAMMPS ML-IAP (unified) model file.

Usage:
    python -m bam_torch.lammps.make_mliap_bam_cg --pkl model_cg.pkl --out model_mliap.pt

The exported file is self-contained: weights and every derived setting are
embedded, so it does not depend on the source pkl, the dataset NPZ or any
absolute path at load time.
"""
import argparse

import torch

from bam_torch.lammps.lammps_mliap_bam_cg import rebuild_bam_mliap


def main():
    ap = argparse.ArgumentParser(description="Export CG model for LAMMPS ML-IAP unified")
    ap.add_argument("--pkl", required=True, help="trained CG model pkl")
    ap.add_argument("--out", required=True, help="output .pt path")
    ap.add_argument("--bond-cutoff", type=float, default=3.2,
                    help="fallback scalar bond cutoff (A) when no per-type matrix")
    ap.add_argument("--backend", default="e3nn", choices=["e3nn", "oeq"],
                    help="tensor-product backend used at inference")
    ap.add_argument("--interaction-block", default="slow", choices=["slow", "fast"],
                    help="must match the block the model was trained with")
    ap.add_argument("--no-load-weights", action="store_true",
                    help="architecture only, random weights (benchmarking only)")
    ap.add_argument("--bond-cutoff-matrix", default="auto", choices=["auto", "none"],
                    help="derive per-type-pair bond cutoffs from the training NPZ")
    a = ap.parse_args()

    obj = rebuild_bam_mliap(a.pkl, a.backend, a.bond_cutoff,
                            a.interaction_block, not a.no_load_weights,
                            "auto" if a.bond_cutoff_matrix == "auto" else None)
    torch.save(obj, a.out)
    print("SAVED:", a.out)
    print("  elements:", obj.element_types, "rcutfac:", obj.rcutfac)
    print("  use_bond_flag:", obj.model.use_bond_flag,
          "bond_cutoff:", float(obj.model.bond_cutoff))


if __name__ == "__main__":
    main()
