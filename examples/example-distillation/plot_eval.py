"""Plot energy + force error analysis from evaluate_student.py output.

Reads ``test_values.pkl`` from both ``<eval-dir>/teacher/`` and
``<eval-dir>/student/`` and produces a 2x2 panel figure:

    +----------------------+----------------------+
    | Energy parity        | Force-component      |
    | (per atom)           | parity (eV/A)        |
    +----------------------+----------------------+
    | Energy error         | Force-component      |
    | histogram (meV/atom) | error histogram      |
    +----------------------+----------------------+

Teacher in blue, student in orange, diagonal reference in grey.
"""
from __future__ import annotations

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from ase.io import read


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--eval-dir", required=True, help="dir containing teacher/ and student/ subdirs")
    p.add_argument("--test-traj", required=True)
    p.add_argument("--out", default=None, help="output png path (default: <eval-dir>/error_plots.png)")
    p.add_argument("--title", default=None, help="figure title")
    return p.parse_args()


def load_pred(path: str):
    tv = torch.load(path, weights_only=False)
    e_pred = torch.cat([t.flatten() for t in tv["energy"]]).numpy()
    e_true = torch.cat([t.flatten() for t in tv["exact_energy"]]).numpy()
    fx = torch.cat(tv["force_x"]).numpy()
    fy = torch.cat(tv["force_y"]).numpy()
    fz = torch.cat(tv["force_z"]).numpy()
    fx_t = torch.cat(tv["exact_force_x"]).numpy()
    fy_t = torch.cat(tv["exact_force_y"]).numpy()
    fz_t = torch.cat(tv["exact_force_z"]).numpy()
    f_pred = np.stack([fx, fy, fz], axis=-1)
    f_true = np.stack([fx_t, fy_t, fz_t], axis=-1)
    return e_pred, e_true, f_pred, f_true


def main() -> None:
    args = parse_args()
    teacher_pkl = os.path.join(args.eval_dir, "teacher", "test_values.pkl")
    student_pkl = os.path.join(args.eval_dir, "student", "test_values.pkl")
    for p in (teacher_pkl, student_pkl):
        if not os.path.exists(p):
            sys.exit(f"missing: {p}")

    out = args.out or os.path.join(args.eval_dir, "error_plots.png")

    traj = read(args.test_traj, index=":")
    n_atoms_per_struct = np.array([len(a) for a in traj], dtype=float)

    e_pT, e_T, f_pT, f_T = load_pred(teacher_pkl)
    e_pS, e_S, f_pS, f_S = load_pred(student_pkl)

    n = min(len(e_pT), len(e_pS), len(n_atoms_per_struct))
    e_pT, e_T = e_pT[:n], e_T[:n]
    e_pS, e_S = e_pS[:n], e_S[:n]
    natoms = n_atoms_per_struct[:n]

    e_pT_pa = e_pT / natoms
    e_T_pa = e_T / natoms
    e_pS_pa = e_pS / natoms
    e_S_pa = e_S / natoms

    err_T_meV = (e_pT_pa - e_T_pa) * 1000.0
    err_S_meV = (e_pS_pa - e_S_pa) * 1000.0

    f_pT_flat = f_pT.reshape(-1)
    f_T_flat = f_T.reshape(-1)
    f_pS_flat = f_pS.reshape(-1)
    f_S_flat = f_S.reshape(-1)

    err_fT_meV = (f_pT_flat - f_T_flat) * 1000.0
    err_fS_meV = (f_pS_flat - f_S_flat) * 1000.0

    rng = np.random.default_rng(0)
    n_force = len(f_T_flat)
    if n_force > 50_000:
        idx = rng.choice(n_force, size=50_000, replace=False)
        fT_plot = (f_T_flat[idx], f_pT_flat[idx])
        fS_plot = (f_S_flat[idx], f_pS_flat[idx])
    else:
        fT_plot = (f_T_flat, f_pT_flat)
        fS_plot = (f_S_flat, f_pS_flat)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    title = args.title or f"teacher vs distilled student — {os.path.basename(args.eval_dir.rstrip('/'))}"
    fig.suptitle(title, fontsize=14)

    ax = axes[0, 0]
    ax.scatter(e_T_pa, e_pT_pa, s=4, alpha=0.4, color="C0", label=f"teacher (MAE={np.mean(np.abs(err_T_meV)):.1f} meV/atom)")
    ax.scatter(e_S_pa, e_pS_pa, s=4, alpha=0.4, color="C1", label=f"student (MAE={np.mean(np.abs(err_S_meV)):.1f} meV/atom)")
    lo = min(e_T_pa.min(), e_pT_pa.min(), e_S_pa.min(), e_pS_pa.min())
    hi = max(e_T_pa.max(), e_pT_pa.max(), e_S_pa.max(), e_pS_pa.max())
    ax.plot([lo, hi], [lo, hi], "k--", alpha=0.4, lw=1)
    ax.set_xlabel("DFT energy [eV/atom]")
    ax.set_ylabel("Predicted energy [eV/atom]")
    ax.set_title("Energy parity (per atom)")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.3)

    ax = axes[0, 1]
    ax.scatter(*fT_plot, s=2, alpha=0.3, color="C0", label=f"teacher (MAE={np.mean(np.abs(err_fT_meV)):.1f} meV/A)")
    ax.scatter(*fS_plot, s=2, alpha=0.3, color="C1", label=f"student (MAE={np.mean(np.abs(err_fS_meV)):.1f} meV/A)")
    lo = min(fT_plot[0].min(), fT_plot[1].min(), fS_plot[0].min(), fS_plot[1].min())
    hi = max(fT_plot[0].max(), fT_plot[1].max(), fS_plot[0].max(), fS_plot[1].max())
    ax.plot([lo, hi], [lo, hi], "k--", alpha=0.4, lw=1)
    ax.set_xlabel("DFT force component [eV/A]")
    ax.set_ylabel("Predicted force component [eV/A]")
    ax.set_title("Force-component parity")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.3)

    ax = axes[1, 0]
    bins = np.linspace(-300, 300, 61)
    ax.hist(np.clip(err_T_meV, bins[0], bins[-1]), bins=bins, alpha=0.55, color="C0", label="teacher")
    ax.hist(np.clip(err_S_meV, bins[0], bins[-1]), bins=bins, alpha=0.55, color="C1", label="student")
    ax.axvline(0, color="k", alpha=0.3, lw=1)
    ax.set_xlabel("Energy error [meV/atom]  (clipped to +/-300)")
    ax.set_ylabel("structures")
    ax.set_title("Energy error distribution")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    ax = axes[1, 1]
    bins = np.linspace(-500, 500, 81)
    ax.hist(np.clip(err_fT_meV, bins[0], bins[-1]), bins=bins, alpha=0.55, color="C0", label="teacher")
    ax.hist(np.clip(err_fS_meV, bins[0], bins[-1]), bins=bins, alpha=0.55, color="C1", label="student")
    ax.axvline(0, color="k", alpha=0.3, lw=1)
    ax.set_xlabel("Force component error [meV/A]  (clipped to +/-500)")
    ax.set_ylabel("force components")
    ax.set_title("Force error distribution")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
