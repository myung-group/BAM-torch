"""
Universal Dataset Validation for BAM-torch

Validates dataset quality before training, regardless of format.
Supports: NPZ (CG), XYZ/trajectory (AA), and any ASE-readable format.

Usage:
    # CG NPZ
    python -m bam_torch.utils.validate_dataset octane_cg.npz --topology 2

    # AA XYZ
    python -m bam_torch.utils.validate_dataset train_300K.xyz

    # In code (auto-detect format):
    from bam_torch.utils.validate_dataset import validate_dataset
    report = validate_dataset("data.npz", beads_per_mol=2)
    report = validate_dataset("train.xyz")
"""

import os
import sys
import argparse
import numpy as np
from typing import Dict, Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))


def _load_npz(path: str, n_sample: int = 100):
    """Load CG NPZ dataset."""
    data = np.load(path, allow_pickle=True)
    positions = data['positions']   # (n_frames, n_atoms, 3)
    forces = data['forces']         # (n_frames, n_atoms, 3)
    energies = data['energies']     # (n_frames,)
    cells = data['cells']           # (n_frames, 3, 3)

    n_frames = positions.shape[0]
    indices = np.linspace(0, n_frames - 1, min(n_sample, n_frames), dtype=int)

    return {
        'positions': positions,
        'forces': forces,
        'energies': energies,
        'cells': cells,
        'n_frames': n_frames,
        'n_atoms': positions.shape[1],
        'sample_indices': indices,
        'format': 'npz',
    }


def _load_ase(path: str, n_sample: int = 100):
    """Load AA dataset via ASE (xyz, traj, extxyz, etc.)."""
    from ase.io import read

    frames = read(path, index=slice(None))
    n_frames = len(frames)
    indices = np.linspace(0, n_frames - 1, min(n_sample, n_frames), dtype=int)

    n_atoms = len(frames[0])
    positions = np.zeros((n_frames, n_atoms, 3))
    forces = np.zeros((n_frames, n_atoms, 3))
    energies = np.zeros(n_frames)
    cells = np.zeros((n_frames, 3, 3))

    has_forces = True
    has_energy = True

    for i, frame in enumerate(frames):
        positions[i] = frame.positions
        cells[i] = frame.cell.array

        try:
            forces[i] = frame.get_forces()
        except Exception:
            has_forces = False

        try:
            energies[i] = frame.get_potential_energy()
        except Exception:
            has_energy = False

    return {
        'positions': positions,
        'forces': forces if has_forces else None,
        'energies': energies if has_energy else None,
        'cells': cells,
        'n_frames': n_frames,
        'n_atoms': n_atoms,
        'sample_indices': indices,
        'format': 'ase',
        'has_forces': has_forces,
        'has_energy': has_energy,
    }


def _detect_format(path: str) -> str:
    """Detect dataset format from file extension."""
    ext = os.path.splitext(path)[1].lower()
    if ext == '.npz':
        return 'npz'
    else:
        return 'ase'  # ASE can read xyz, traj, extxyz, etc.


def validate_dataset(
    data_path: str,
    beads_per_mol: Optional[int] = None,
    bond_threshold: float = 10.0,
    force_sum_threshold: float = 0.1,
    max_force_threshold: float = 50.0,
    n_sample_frames: int = 100,
    verbose: bool = True,
) -> Dict:
    """
    Validate dataset quality before training.

    Checks:
        1. PBC bond distances (CG only, needs beads_per_mol)
        2. Total force conservation: |Σ F| ≈ 0
        3. Force distribution: no unphysically large forces
        4. Energy stability: no drift
        5. Structural sanity: no overlapping atoms/beads
        6. Baseline learnability: force variance for reference

    Args:
        data_path: Path to dataset file (NPZ, XYZ, TRAJ, etc.)
        beads_per_mol: Beads per molecule for bond check (CG only)
        bond_threshold: Max bond distance in Å (default: 10)
        force_sum_threshold: Max |Σ F| in eV/Å (default: 0.1)
        max_force_threshold: Max |F| per atom in eV/Å (default: 50)
        n_sample_frames: Frames to sample (default: 100)
        verbose: Print results

    Returns:
        Dict with: passed, errors, warnings, metrics
    """
    errors = []
    warnings = []
    metrics = {}

    if verbose:
        print("=" * 60)
        print(f"Dataset Validation: {data_path}")
        print("=" * 60)

    # ─── Load data ───
    if not os.path.exists(data_path):
        return {'passed': False, 'errors': [f'File not found: {data_path}'],
                'warnings': [], 'metrics': {}}

    fmt = _detect_format(data_path)
    if fmt == 'npz':
        dataset = _load_npz(data_path, n_sample_frames)
    else:
        dataset = _load_ase(data_path, n_sample_frames)

    positions = dataset['positions']
    forces = dataset.get('forces')
    energies = dataset.get('energies')
    cells = dataset['cells']
    n_frames = dataset['n_frames']
    n_atoms = dataset['n_atoms']
    sample_idx = dataset['sample_indices']

    has_forces = forces is not None and (fmt == 'npz' or dataset.get('has_forces', True))
    has_energy = energies is not None and (fmt == 'npz' or dataset.get('has_energy', True))

    metrics['n_frames'] = n_frames
    metrics['n_atoms'] = n_atoms
    metrics['format'] = fmt

    if verbose:
        print(f"\n  Format: {fmt.upper()}")
        print(f"  Frames: {n_frames}, Atoms/beads: {n_atoms}")
        print(f"  Forces: {'yes' if has_forces else 'NO'}")
        print(f"  Energy: {'yes' if has_energy else 'NO'}")

    if not has_forces:
        errors.append("NO FORCE DATA: dataset has no forces. Cannot train force field.")
        return {'passed': False, 'errors': errors, 'warnings': warnings, 'metrics': metrics}

    # ─── Check 1: PBC Bond Distances ───
    if verbose:
        print(f"\n--- Check 1: PBC Bond Distances ---")

    if beads_per_mol is not None and beads_per_mol > 1:
        n_mol = n_atoms // beads_per_mol
        broken_counts = []
        all_bond_dists = []

        for frame in sample_idx:
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
        pct_broken = np.mean(broken_counts) / total_bonds * 100

        metrics['bond_mean'] = float(all_bond_dists.mean())
        metrics['bond_std'] = float(all_bond_dists.std())
        metrics['bond_max'] = float(all_bond_dists.max())
        metrics['pct_broken_bonds'] = float(pct_broken)

        if verbose:
            print(f"  Bond: mean={all_bond_dists.mean():.4f}, std={all_bond_dists.std():.4f}, "
                  f"max={all_bond_dists.max():.4f} Å")
            print(f"  Broken (>{bond_threshold}Å): {pct_broken:.1f}%")

        if pct_broken > 1.0:
            errors.append(f"PBC WRAPPING: {pct_broken:.1f}% bonds broken. Fix COM mapping.")
        elif pct_broken > 0:
            warnings.append(f"Minor PBC: {pct_broken:.2f}% bonds broken")
        else:
            if verbose:
                print(f"  ✓ PASSED")
    else:
        if verbose:
            print(f"  SKIPPED (beads_per_mol not specified)")

    # ─── Check 2: Total Force Conservation ───
    if verbose:
        print(f"\n--- Check 2: Total Force Conservation ---")

    force_sums = np.array([np.linalg.norm(forces[f].sum(axis=0)) for f in sample_idx])
    metrics['force_sum_mean'] = float(force_sums.mean())
    metrics['force_sum_max'] = float(force_sums.max())

    if verbose:
        print(f"  |Σ F|: mean={force_sums.mean():.6f}, max={force_sums.max():.6f} eV/Å")

    if force_sums.mean() > force_sum_threshold:
        errors.append(f"FORCE CONSERVATION: mean |Σ F| = {force_sums.mean():.4f} eV/Å (> {force_sum_threshold})")
    else:
        if verbose:
            print(f"  ✓ PASSED")

    # ─── Check 3: Force Distribution ───
    if verbose:
        print(f"\n--- Check 3: Force Distribution ---")

    f_sample = forces[sample_idx]
    f_mag = np.linalg.norm(f_sample, axis=-1)

    metrics['force_mean'] = float(f_sample.mean())
    metrics['force_std'] = float(f_sample.std())
    metrics['force_mag_mean'] = float(f_mag.mean())
    metrics['force_mag_max'] = float(f_mag.max())

    n_large = np.sum(f_mag > max_force_threshold)
    pct_large = n_large / f_mag.size * 100
    metrics['pct_large_forces'] = float(pct_large)

    if verbose:
        print(f"  |F|: mean={f_mag.mean():.4f}, max={f_mag.max():.4f} eV/Å")
        print(f"  Large forces (>{max_force_threshold}): {pct_large:.2f}%")

    if pct_large > 0.1:
        errors.append(f"LARGE FORCES: {pct_large:.2f}% exceed {max_force_threshold} eV/Å")
    else:
        if verbose:
            print(f"  ✓ PASSED")

    # ─── Check 4: Energy Stability ───
    if verbose:
        print(f"\n--- Check 4: Energy Stability ---")

    if has_energy:
        e_std = energies.std()
        slope = np.polyfit(np.arange(n_frames), energies, 1)[0]
        total_drift = abs(slope) * n_frames
        metrics['energy_mean'] = float(energies.mean())
        metrics['energy_std'] = float(e_std)
        metrics['energy_drift'] = float(total_drift)

        if verbose:
            print(f"  Energy: mean={energies.mean():.4f}, std={e_std:.4f} eV")
            print(f"  Drift: {total_drift:.4f} eV over {n_frames} frames")

        if e_std > 0 and total_drift > 10 * e_std:
            warnings.append(f"Energy drift: {total_drift:.4f} eV (> 10σ={10*e_std:.4f})")
        else:
            if verbose:
                print(f"  ✓ PASSED")
    else:
        if verbose:
            print(f"  SKIPPED (no energy data)")

    # ─── Check 5: Structural Sanity ───
    if verbose:
        print(f"\n--- Check 5: Structural Sanity ---")

    frame0 = positions[0]
    cell_diag = np.diag(cells[0])
    use_pbc = np.abs(cell_diag).sum() > 1e-6

    np.random.seed(42)
    min_dist = float('inf')
    for _ in range(min(1000, n_atoms * (n_atoms - 1) // 2)):
        i, j = np.random.choice(n_atoms, 2, replace=False)
        r_vec = frame0[j] - frame0[i]
        if use_pbc:
            r_vec -= cell_diag * np.round(r_vec / cell_diag)
        r = np.linalg.norm(r_vec)
        if r > 0.01:
            min_dist = min(min_dist, r)

    metrics['min_pairwise_dist'] = float(min_dist)

    if verbose:
        print(f"  Min pairwise distance: {min_dist:.4f} Å")

    if min_dist < 0.5:
        warnings.append(f"Very small distance: {min_dist:.4f} Å")
    else:
        if verbose:
            print(f"  ✓ PASSED")

    # ─── Check 6: Baseline ───
    if verbose:
        print(f"\n--- Check 6: Baseline Learnability ---")

    baseline_mse = (f_sample**2).mean()
    metrics['baseline_mse'] = float(baseline_mse)

    if verbose:
        print(f"  Force variance: {f_sample.var():.6f} (eV/Å)²")
        print(f"  Baseline MSE (F=0): {baseline_mse:.6f}")

    # ─── Summary ───
    passed = len(errors) == 0

    if verbose:
        print(f"\n{'=' * 60}")
        if passed:
            print(f"  ✓ VALIDATION PASSED")
        else:
            print(f"  ✗ VALIDATION FAILED ({len(errors)} error(s))")
        if errors:
            for e in errors:
                print(f"    ✗ {e}")
        if warnings:
            for w in warnings:
                print(f"    ⚠ {w}")
        print(f"{'=' * 60}")

    return {'passed': passed, 'errors': errors, 'warnings': warnings, 'metrics': metrics}


def main():
    parser = argparse.ArgumentParser(description='Validate dataset before training')
    parser.add_argument('data_path', help='Dataset file (NPZ, XYZ, TRAJ, etc.)')
    parser.add_argument('--topology', '-t', type=int, default=None,
                        help='Beads/atoms per molecule for bond check')
    parser.add_argument('--bond-threshold', type=float, default=10.0)
    parser.add_argument('--force-threshold', type=float, default=0.1)
    parser.add_argument('--n-samples', type=int, default=100)
    args = parser.parse_args()

    report = validate_dataset(
        args.data_path,
        beads_per_mol=args.topology,
        bond_threshold=args.bond_threshold,
        force_sum_threshold=args.force_threshold,
        n_sample_frames=args.n_samples,
    )
    sys.exit(0 if report['passed'] else 1)


if __name__ == '__main__':
    main()
