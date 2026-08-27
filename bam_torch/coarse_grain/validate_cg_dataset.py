"""
CG Dataset Validation Script

Validates CG NPZ dataset quality before training.
Checks: PBC wrapping, force consistency, energy stability, structural sanity.

Usage:
    python -m bam_torch.coarse_grain.validate_cg_dataset octane_cg.npz
    python -m bam_torch.coarse_grain.validate_cg_dataset octane_cg.npz --topology 2  # 2 beads/mol

    # In code:
    from bam_torch.coarse_grain.validate_cg_dataset import validate_cg_npz
    report = validate_cg_npz("octane_cg.npz", beads_per_mol=2)
    if not report['passed']:
        print("VALIDATION FAILED:", report['errors'])
"""

import os
import sys
import argparse
import numpy as np
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))


def validate_cg_npz(
    npz_path: str,
    beads_per_mol: Optional[int] = None,
    bond_threshold: float = 10.0,
    force_sum_threshold: float = 0.1,
    max_force_threshold: float = 50.0,
    n_sample_frames: int = 100,
    verbose: bool = True,
) -> Dict:
    """
    Validate CG NPZ dataset quality.

    Args:
        npz_path: Path to CG NPZ file
        beads_per_mol: Number of beads per molecule (for bond check).
                       If None, bond check is skipped.
        bond_threshold: Maximum allowed bond distance (Å). Bonds exceeding
                        this are flagged as broken (PBC wrapping issue).
        force_sum_threshold: Maximum allowed |Σ F| per frame (eV/Å).
                             Should be ~0 for isolated systems.
        max_force_threshold: Maximum allowed |F| per bead (eV/Å).
                             Forces exceeding this are flagged as unphysical.
        n_sample_frames: Number of frames to sample for checks.
        verbose: Print detailed results.

    Returns:
        Dict with keys:
            passed: bool — overall pass/fail
            errors: list of error strings
            warnings: list of warning strings
            metrics: dict of computed metrics
    """
    errors = []
    warnings = []
    metrics = {}

    if verbose:
        print("=" * 60)
        print(f"CG Dataset Validation: {npz_path}")
        print("=" * 60)

    # ─── Load data ───
    if not os.path.exists(npz_path):
        return {'passed': False, 'errors': [f'File not found: {npz_path}'],
                'warnings': [], 'metrics': {}}

    data = np.load(npz_path, allow_pickle=True)
    positions = data['positions']
    forces = data['forces']
    energies = data['energies']
    types = data['types']
    cells = data['cells']

    n_frames, n_beads, _ = positions.shape
    n_types = len(np.unique(types))

    metrics['n_frames'] = n_frames
    metrics['n_beads'] = n_beads
    metrics['n_types'] = n_types

    if verbose:
        print(f"\n  Frames: {n_frames}, Beads: {n_beads}, Types: {n_types}")

    # Sample frames for validation
    if n_sample_frames >= n_frames:
        sample_indices = list(range(n_frames))
    else:
        sample_indices = np.linspace(0, n_frames - 1, n_sample_frames, dtype=int).tolist()

    # ─── Check 1: PBC Bond Distances ───
    if verbose:
        print(f"\n--- Check 1: PBC Bond Distances ---")

    if beads_per_mol is not None and beads_per_mol > 1:
        n_mol = n_beads // beads_per_mol
        broken_counts = []
        all_bond_dists = []

        for frame in sample_indices:
            cell_diag = np.diag(cells[frame])
            use_pbc = np.abs(cell_diag).sum() > 1e-6
            broken = 0

            for m in range(n_mol):
                for b in range(beads_per_mol - 1):
                    i = m * beads_per_mol + b
                    j = m * beads_per_mol + b + 1
                    r_vec = positions[frame, j] - positions[frame, i]
                    if use_pbc:
                        r_vec -= cell_diag * np.round(r_vec / cell_diag)
                    r = np.linalg.norm(r_vec)
                    all_bond_dists.append(r)
                    if r > bond_threshold:
                        broken += 1

            broken_counts.append(broken)

        all_bond_dists = np.array(all_bond_dists)
        total_bonds = n_mol * (beads_per_mol - 1)
        avg_broken = np.mean(broken_counts)
        pct_broken = avg_broken / total_bonds * 100

        metrics['bond_mean'] = float(all_bond_dists.mean())
        metrics['bond_std'] = float(all_bond_dists.std())
        metrics['bond_min'] = float(all_bond_dists.min())
        metrics['bond_max'] = float(all_bond_dists.max())
        metrics['pct_broken_bonds'] = float(pct_broken)

        if verbose:
            print(f"  Bond distance: mean={all_bond_dists.mean():.4f}, "
                  f"std={all_bond_dists.std():.4f}, "
                  f"min={all_bond_dists.min():.4f}, max={all_bond_dists.max():.4f} Å")
            print(f"  Broken bonds (>{bond_threshold}Å): "
                  f"{avg_broken:.0f}/{total_bonds} ({pct_broken:.1f}%)")

        if pct_broken > 1.0:
            errors.append(
                f"PBC WRAPPING ERROR: {pct_broken:.1f}% of bonds broken "
                f"(>{bond_threshold}Å). CG bead COM mapping likely incorrect. "
                f"Use PBC-aware COM (compute_com with box argument)."
            )
        elif pct_broken > 0:
            warnings.append(
                f"Minor PBC issue: {pct_broken:.2f}% bonds broken"
            )
        else:
            if verbose:
                print(f"  ✓ PASSED: 0% broken bonds")
    else:
        if verbose:
            print(f"  SKIPPED (beads_per_mol not specified)")

    # ─── Check 2: Total Force Conservation ───
    if verbose:
        print(f"\n--- Check 2: Total Force Conservation (Σ F ≈ 0) ---")

    force_sums = []
    for frame in sample_indices:
        f_total = forces[frame].sum(axis=0)  # (3,)
        f_mag = np.linalg.norm(f_total)
        force_sums.append(f_mag)

    force_sums = np.array(force_sums)
    metrics['force_sum_mean'] = float(force_sums.mean())
    metrics['force_sum_max'] = float(force_sums.max())

    if verbose:
        print(f"  |Σ F|: mean={force_sums.mean():.6f}, "
              f"max={force_sums.max():.6f} eV/Å")

    if force_sums.mean() > force_sum_threshold:
        errors.append(
            f"FORCE CONSERVATION ERROR: mean |Σ F| = {force_sums.mean():.4f} eV/Å "
            f"(threshold: {force_sum_threshold}). "
            f"Total force should be ~0. Check force mapping or external forces."
        )
    else:
        if verbose:
            print(f"  ✓ PASSED: |Σ F| < {force_sum_threshold} eV/Å")

    # ─── Check 3: Force Distribution ───
    if verbose:
        print(f"\n--- Check 3: Force Distribution ---")

    f_all = forces[sample_indices]
    f_magnitudes = np.linalg.norm(f_all, axis=-1)  # (n_sample, n_beads)

    metrics['force_mean'] = float(f_all.mean())
    metrics['force_std'] = float(f_all.std())
    metrics['force_mag_mean'] = float(f_magnitudes.mean())
    metrics['force_mag_max'] = float(f_magnitudes.max())

    if verbose:
        print(f"  F components: mean={f_all.mean():.6f}, std={f_all.std():.6f} eV/Å")
        print(f"  |F|: mean={f_magnitudes.mean():.4f}, max={f_magnitudes.max():.4f} eV/Å")

    # Check for unphysically large forces
    n_large = np.sum(f_magnitudes > max_force_threshold)
    pct_large = n_large / f_magnitudes.size * 100
    metrics['pct_large_forces'] = float(pct_large)

    if pct_large > 0.1:
        errors.append(
            f"LARGE FORCE ERROR: {pct_large:.2f}% of forces exceed "
            f"{max_force_threshold} eV/Å. Possible constraint artifacts or bad mapping."
        )
    elif pct_large > 0:
        warnings.append(
            f"Minor: {n_large} forces exceed {max_force_threshold} eV/Å ({pct_large:.4f}%)"
        )
    else:
        if verbose:
            print(f"  ✓ PASSED: no forces > {max_force_threshold} eV/Å")

    # ─── Check 4: Energy Stability ───
    if verbose:
        print(f"\n--- Check 4: Energy Stability ---")

    e_mean = energies.mean()
    e_std = energies.std()
    e_range = energies.max() - energies.min()

    metrics['energy_mean'] = float(e_mean)
    metrics['energy_std'] = float(e_std)
    metrics['energy_range'] = float(e_range)

    # Check for energy drift (linear fit)
    x = np.arange(n_frames)
    slope = np.polyfit(x, energies, 1)[0]
    drift_per_frame = abs(slope)
    total_drift = drift_per_frame * n_frames
    metrics['energy_drift_total'] = float(total_drift)

    if verbose:
        print(f"  Energy: mean={e_mean:.4f}, std={e_std:.4f} eV")
        print(f"  Range: {e_range:.4f} eV")
        print(f"  Drift: {total_drift:.4f} eV over {n_frames} frames")

    # Drift > 10*std is suspicious
    if total_drift > 10 * e_std and e_std > 0:
        warnings.append(
            f"Energy drift: {total_drift:.4f} eV over trajectory "
            f"(10x std={10*e_std:.4f}). System may not be equilibrated."
        )
    else:
        if verbose:
            print(f"  ✓ PASSED: energy drift within normal range")

    # ─── Check 5: Structural Sanity ───
    if verbose:
        print(f"\n--- Check 5: Structural Sanity ---")

    # Minimum inter-bead distance (should not be unrealistically small)
    frame0 = positions[0]
    cell0 = cells[0]
    cell_diag = np.diag(cell0)
    use_pbc = np.abs(cell_diag).sum() > 1e-6

    # Sample pairwise distances (not all pairs — too expensive)
    min_dist = float('inf')
    np.random.seed(42)
    for _ in range(min(1000, n_beads * (n_beads - 1) // 2)):
        i, j = np.random.choice(n_beads, 2, replace=False)
        r_vec = frame0[j] - frame0[i]
        if use_pbc:
            r_vec -= cell_diag * np.round(r_vec / cell_diag)
        r = np.linalg.norm(r_vec)
        if r > 0.01:
            min_dist = min(min_dist, r)

    metrics['min_pairwise_dist'] = float(min_dist)

    if verbose:
        print(f"  Min pairwise distance (sample): {min_dist:.4f} Å")

    if min_dist < 1.0:
        warnings.append(
            f"Very small pairwise distance: {min_dist:.4f} Å. "
            f"Possible overlapping beads."
        )
    else:
        if verbose:
            print(f"  ✓ PASSED: no overlapping beads detected")

    # ─── Check 6: Baseline Learnability ───
    if verbose:
        print(f"\n--- Check 6: Baseline Learnability ---")

    force_var = f_all.var()
    baseline_mse = (f_all**2).mean()  # MSE if model predicts F=0

    metrics['force_variance'] = float(force_var)
    metrics['baseline_mse'] = float(baseline_mse)

    if verbose:
        print(f"  Force variance: {force_var:.6f} (eV/Å)²")
        print(f"  Baseline MSE (F=0): {baseline_mse:.6f} (eV/Å)²")
        print(f"  → Model must achieve MSE < {baseline_mse:.6f} to beat trivial baseline")

    # ─── Summary ───
    passed = len(errors) == 0

    if verbose:
        print(f"\n{'=' * 60}")
        if passed:
            print(f"  ✓ VALIDATION PASSED")
        else:
            print(f"  ✗ VALIDATION FAILED ({len(errors)} error(s))")

        if errors:
            print(f"\n  ERRORS:")
            for e in errors:
                print(f"    ✗ {e}")

        if warnings:
            print(f"\n  WARNINGS:")
            for w in warnings:
                print(f"    ⚠ {w}")

        print(f"{'=' * 60}")

    return {
        'passed': passed,
        'errors': errors,
        'warnings': warnings,
        'metrics': metrics,
    }


def main():
    parser = argparse.ArgumentParser(
        description='Validate CG NPZ dataset before training'
    )
    parser.add_argument('npz_path', help='Path to CG NPZ file')
    parser.add_argument('--topology', '-t', type=int, default=None,
                        help='Number of beads per molecule (for bond check)')
    parser.add_argument('--bond-threshold', type=float, default=10.0,
                        help='Max allowed bond distance in Å (default: 10)')
    parser.add_argument('--force-threshold', type=float, default=0.1,
                        help='Max allowed |Σ F| in eV/Å (default: 0.1)')
    parser.add_argument('--n-samples', type=int, default=100,
                        help='Number of frames to sample (default: 100)')

    args = parser.parse_args()

    report = validate_cg_npz(
        args.npz_path,
        beads_per_mol=args.topology,
        bond_threshold=args.bond_threshold,
        force_sum_threshold=args.force_threshold,
        n_sample_frames=args.n_samples,
    )

    sys.exit(0 if report['passed'] else 1)


if __name__ == '__main__':
    main()
