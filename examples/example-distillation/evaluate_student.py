"""Evaluate student vs teacher on a held-out MPtrj test trajectory.

Reports for both models:

  * force MAE / RMSE  (meV/A)
  * energy MAE / RMSE / bias per atom against ``uncorrected_total_energy``

Also prints a wall-clock forward-pass timing comparison on the same test
structures so you can see the speedup directly.

The student checkpoint must contain the embedded ``input.json`` of the trained
student architecture (BAM-torch saves this in every ckpt).
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from copy import deepcopy

import numpy as np
import torch
from ase.io import read

from bam_torch.predicting.evaluator import Evaluator


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--student-ckpt", required=True)
    p.add_argument("--teacher-ckpt", required=True)
    p.add_argument("--test-traj", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--time-n", type=int, default=50,
                   help="number of structures to use for the timing benchmark")
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


def run_evaluator(label: str, ckpt_path: str, traj_path: str, work_dir: str) -> dict:
    """Run BAM-torch's Evaluator on the given ckpt; return per-frame predictions."""
    sub = os.path.join(work_dir, label)
    os.makedirs(sub, exist_ok=True)
    cfg = build_eval_cfg(ckpt_path, traj_path, sub)
    cwd = os.getcwd()
    os.chdir(sub)
    try:
        Evaluator(cfg).evaluate()
    finally:
        os.chdir(cwd)

    tv = torch.load(os.path.join(sub, "test_values.pkl"), weights_only=False)
    e_pred = torch.cat([t.flatten() for t in tv["energy"]]).numpy()
    fx = torch.cat(tv["force_x"]).numpy()
    fy = torch.cat(tv["force_y"]).numpy()
    fz = torch.cat(tv["force_z"]).numpy()
    fx_t = torch.cat(tv["exact_force_x"]).numpy()
    fy_t = torch.cat(tv["exact_force_y"]).numpy()
    fz_t = torch.cat(tv["exact_force_z"]).numpy()
    fdiff = np.stack([fx - fx_t, fy - fy_t, fz - fz_t], axis=-1)
    return {"e_pred": e_pred, "fdiff": fdiff}


def per_atom_metrics(label: str, e_pred: np.ndarray, e_ref: np.ndarray, n_atoms: np.ndarray, fdiff: np.ndarray) -> None:
    diff = e_pred - e_ref
    per_atom = diff / n_atoms
    print(f"  {label}:")
    print(f"    energy MAE/atom  = {np.mean(np.abs(per_atom)) * 1000:6.2f} meV")
    print(f"    energy RMSE/atom = {np.sqrt(np.mean(per_atom ** 2)) * 1000:6.2f} meV")
    print(f"    energy bias/atom = {np.mean(per_atom) * 1000:+7.2f} meV")
    print(f"    force MAE        = {np.mean(np.abs(fdiff)) * 1000:6.2f} meV/A")
    print(f"    force RMSE       = {np.sqrt(np.mean(fdiff ** 2)) * 1000:6.2f} meV/A")


def time_forward(ckpt_path: str, traj_path: str, n: int, scratch_dir: str) -> tuple[float, int]:
    """Time forward passes on the first ``n`` structures, return (median_ms, params)."""
    cfg = build_eval_cfg(ckpt_path, traj_path, scratch_dir)
    os.makedirs(scratch_dir, exist_ok=True)
    cwd = os.getcwd()
    os.chdir(scratch_dir)
    try:
        ev = Evaluator(cfg)
    finally:
        os.chdir(cwd)
    ev.model.eval()

    times: list[float] = []
    seen = 0
    for data in ev.data_loader:
        if seen >= n:
            break
        data = data.to(ev.device)
        if ev.device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        _ = ev.model(data, backprop=False)
        if ev.device.type == "cuda":
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
        seen += 1
    return float(np.median(times) * 1000.0), ev.n_params


def main() -> None:
    args = parse_args()
    args.student_ckpt = os.path.abspath(args.student_ckpt)
    args.teacher_ckpt = os.path.abspath(args.teacher_ckpt)
    args.test_traj = os.path.abspath(args.test_traj)
    args.out_dir = os.path.abspath(args.out_dir)

    if not os.path.exists(args.student_ckpt):
        sys.exit(f"student ckpt not found: {args.student_ckpt}")
    if not os.path.exists(args.teacher_ckpt):
        sys.exit(f"teacher ckpt not found: {args.teacher_ckpt}")
    if not os.path.exists(args.test_traj):
        sys.exit(f"test traj not found: {args.test_traj}")

    os.makedirs(args.out_dir, exist_ok=True)

    print(f"[eval] loading test traj {args.test_traj}")
    traj = read(args.test_traj, index=":")
    n = len(traj)
    n_atoms = np.array([len(a) for a in traj], dtype=float)
    e_uncorr = np.array([a.info["uncorrected_total_energy"] for a in traj])

    print(f"\n[eval] running TEACHER on {n} structures")
    teach_res = run_evaluator("teacher", args.teacher_ckpt, args.test_traj, args.out_dir)

    print(f"\n[eval] running STUDENT on {n} structures")
    stud_res = run_evaluator("student", args.student_ckpt, args.test_traj, args.out_dir)

    print(f"\n[eval] timing forward passes (median over {args.time_n})")
    timing_dir = os.path.join(args.out_dir, "_timing")
    t_student_ms, n_params_student = time_forward(args.student_ckpt, args.test_traj, args.time_n, timing_dir)
    t_teacher_ms, n_params_teacher = time_forward(args.teacher_ckpt, args.test_traj, args.time_n, timing_dir)

    bar = "=" * 72
    print(f"\n{bar}")
    print(f"Held-out test set: {n} structures, {int(n_atoms.min())}/{int(np.median(n_atoms))}/{int(n_atoms.max())} atoms (min/median/max)")
    print(bar)

    per_atom_metrics("TEACHER", teach_res["e_pred"], e_uncorr, n_atoms, teach_res["fdiff"])
    print()
    per_atom_metrics("STUDENT", stud_res["e_pred"], e_uncorr, n_atoms, stud_res["fdiff"])
    print()
    print("Forward-pass timing:")
    print(f"  TEACHER ({n_params_teacher/1e6:5.2f}M params): {t_teacher_ms:6.2f} ms / structure")
    print(f"  STUDENT ({n_params_student/1e6:5.2f}M params): {t_student_ms:6.2f} ms / structure")
    print(f"  speedup (teacher/student): {t_teacher_ms / t_student_ms:.2f}x")
    print()


if __name__ == "__main__":
    main()
