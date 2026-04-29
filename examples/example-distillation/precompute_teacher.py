"""Precompute teacher model predictions on a trajectory.

For each frame in the input traj we save:

  * energy:  raw model output (residual w.r.t. teacher's enr_avg_per_element).
             *No* scale_shift, *no* `valid_scale_shift` correction. This is
             exactly what the student model is trained to predict, so the
             distillation loss can compare these tensors directly.
  * forces:  per-atom force tensor, shape ``(n_atoms_i, 3)``.

Output is a dict with three keys, written via ``torch.save``::

    {
        "energy":  Tensor[N]                 # one residual per frame
        "forces":  List[Tensor[n_atoms, 3]]  # variable shape
        "n_atoms": Tensor[N]                 # for sanity checks
    }

The order is the same as ``read(traj, index=':')`` so the student dataset
loader can attach predictions by integer frame index.
"""
from __future__ import annotations

import argparse
import os
import sys
from copy import deepcopy

import torch
from ase.io import read

from bam_torch.predicting.evaluator import Evaluator


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--traj", required=True, help="ASE traj file to forward through teacher")
    p.add_argument("--teacher-ckpt", required=True, help="path to pretrained teacher .pkl")
    p.add_argument("--out", required=True, help="output .pt path")
    return p.parse_args()


def build_eval_cfg(ckpt_path: str, traj_path: str, out_dir: str) -> dict:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = deepcopy(ckpt["input.json"])

    cfg["gpu-parallel"] = False
    cfg["device"] = "gpu" if torch.cuda.is_available() else "cpu"
    cfg["nbatch"] = 1
    cfg["NN"]["restart"] = False
    cfg["NN"]["fname_pkl"] = ckpt_path
    cfg["predict"] = {
        "evaluate_tag": True,
        "fname_traj": traj_path,
        "ndata": traj_path,
        "model": ckpt_path,
        "fname_plog": os.path.join(out_dir, "predict.out"),
        "loss_config": {"energy_loss": "rmse", "force_loss": "rmse"},
    }
    cfg["train"] = {"fname_log": os.path.join(out_dir, "loss_eval.out")}
    return cfg


def main() -> None:
    args = parse_args()
    if not os.path.exists(args.traj):
        sys.exit(f"traj not found: {args.traj}")
    if not os.path.exists(args.teacher_ckpt):
        sys.exit(f"teacher ckpt not found: {args.teacher_ckpt}")

    args.traj = os.path.abspath(args.traj)
    args.teacher_ckpt = os.path.abspath(args.teacher_ckpt)
    args.out = os.path.abspath(args.out)
    out_dir = os.path.dirname(args.out)
    os.makedirs(out_dir, exist_ok=True)

    cfg = build_eval_cfg(args.teacher_ckpt, args.traj, out_dir)
    cwd = os.getcwd()
    os.chdir(out_dir)
    try:
        evaluator = Evaluator(cfg)
    finally:
        os.chdir(cwd)

    n_expected = len(read(args.traj, index=":"))
    print(f"[precompute] forwarding teacher on {n_expected} frames from {args.traj}")

    energies: list[torch.Tensor] = []
    forces: list[torch.Tensor] = []
    n_atoms: list[int] = []

    evaluator.model.eval()
    for i, data in enumerate(evaluator.data_loader):
        data = data.to(evaluator.device)
        preds = evaluator.model(data, backprop=False)
        energies.append(preds["energy"].detach().cpu().flatten())
        forces.append(preds["forces"].detach().cpu())
        n_atoms.append(int(forces[-1].shape[0]))
        if (i + 1) % 200 == 0:
            print(f"  {i + 1}/{n_expected}")

    energies_t = torch.cat(energies, dim=0)
    n_atoms_t = torch.tensor(n_atoms, dtype=torch.long)

    if energies_t.numel() != n_expected:
        sys.exit(
            f"frame count mismatch: traj has {n_expected} but forwarded {energies_t.numel()}. "
            "Likely DataLoader drop_last=True is set; rerun with nbatch=1 (default)."
        )

    torch.save({"energy": energies_t, "forces": forces, "n_atoms": n_atoms_t}, args.out)
    print(f"[precompute] wrote {args.out}: "
          f"energy shape={tuple(energies_t.shape)}, "
          f"forces list len={len(forces)}, "
          f"total atoms={int(n_atoms_t.sum())}")


if __name__ == "__main__":
    main()
