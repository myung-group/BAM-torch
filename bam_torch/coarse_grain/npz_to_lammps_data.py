"""
Convert CG npz file to LAMMPS data format.

Usage:
    python -m bam_torch.lammps.npz_to_lammps_data water_cg.npz water_cg.data --frame 0

Or in Python:
    from bam_torch.lammps.npz_to_lammps_data import npz_to_lammps_data
    npz_to_lammps_data('water_cg.npz', 'water_cg.data', frame=0)
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
import numpy as np
import argparse


def npz_to_lammps_data(npz_path, output_path, frame=0, masses=None, comment=None):
    """Convert CG npz file to LAMMPS data format.

    Args:
        npz_path: Path to the CG npz file
        output_path: Path to save the LAMMPS data file
        frame: Frame index to extract (default: 0)
        masses: Per-type masses as dict {type_id: mass} or single float for all types.
            If None, attempts to read from NPZ metadata, then falls back to 72.0.
        comment: Comment for the data file header

    Returns:
        output_path: Path where the data file was saved
    """
    # Load npz
    data = np.load(npz_path, allow_pickle=True)

    positions = data['positions']
    types = data['types']
    cells = data['cells']

    n_frames = positions.shape[0]
    n_atoms = positions.shape[1]

    if frame >= n_frames:
        raise ValueError(f"Frame {frame} out of range. Max frame: {n_frames - 1}")

    # Extract single frame
    pos = positions[frame]  # (n_atoms, 3)
    cell = cells[frame]     # (3, 3)

    # Get unique types
    unique_types = np.unique(types)
    n_types = len(unique_types)

    # Resolve per-type masses
    if masses is None:
        # Try to read from metadata
        if 'metadata' in data:
            try:
                meta = data['metadata'].item()
                if isinstance(meta, dict) and 'bead_masses' in meta:
                    bead_masses_list = meta['bead_masses']
                    # bead_masses_list is per-bead; extract unique per-type
                    mass_by_type = {}
                    all_types = data['types']
                    for i, t in enumerate(all_types):
                        if int(t) not in mass_by_type and i < len(bead_masses_list):
                            mass_by_type[int(t)] = bead_masses_list[i]
                    if mass_by_type:
                        masses = mass_by_type
            except Exception:
                pass
        if masses is None:
            masses = 72.0  # generic default

    # Convert single mass to per-type dict
    if isinstance(masses, (int, float)):
        masses = {int(t): float(masses) for t in unique_types}

    # Cell dimensions (assuming orthorhombic)
    # For triclinic, cell matrix diagonal gives box lengths
    xlo, ylo, zlo = 0.0, 0.0, 0.0
    xhi = cell[0, 0]
    yhi = cell[1, 1]
    zhi = cell[2, 2]

    # Check for triclinic (off-diagonal elements)
    xy = cell[1, 0] if abs(cell[1, 0]) > 1e-6 else 0.0
    xz = cell[2, 0] if abs(cell[2, 0]) > 1e-6 else 0.0
    yz = cell[2, 1] if abs(cell[2, 1]) > 1e-6 else 0.0

    is_triclinic = (xy != 0.0 or xz != 0.0 or yz != 0.0)

    # Create comment
    if comment is None:
        comment = f"CG data from {npz_path} (frame {frame})"

    # Write LAMMPS data file
    with open(output_path, 'w') as f:
        f.write(f"# {comment}\n\n")

        f.write(f"{n_atoms} atoms\n")
        f.write(f"{n_types} atom types\n\n")

        if is_triclinic:
            f.write(f"{xlo:.6f} {xhi:.6f} xlo xhi\n")
            f.write(f"{ylo:.6f} {yhi:.6f} ylo yhi\n")
            f.write(f"{zlo:.6f} {zhi:.6f} zlo zhi\n")
            f.write(f"{xy:.6f} {xz:.6f} {yz:.6f} xy xz yz\n\n")
        else:
            f.write(f"{xlo:.6f} {xhi:.6f} xlo xhi\n")
            f.write(f"{ylo:.6f} {yhi:.6f} ylo yhi\n")
            f.write(f"{zlo:.6f} {zhi:.6f} zlo zhi\n\n")

        # Masses
        f.write("Masses\n\n")
        for i, t in enumerate(unique_types):
            m = masses.get(int(t), 72.0)
            f.write(f"{i + 1} {m:.4f}  # CG bead type {t}\n")
        f.write("\n")

        # Atoms section
        f.write("Atoms  # atomic\n\n")
        for i in range(n_atoms):
            atom_type = int(types[i]) + 1  # LAMMPS uses 1-based indexing
            x, y, z = pos[i]
            f.write(f"{i + 1} {atom_type} {x:.6f} {y:.6f} {z:.6f}\n")

    print(f"✓ LAMMPS data file saved to: {output_path}")
    print(f"  - Frame: {frame}/{n_frames - 1}")
    print(f"  - Number of CG beads: {n_atoms}")
    print(f"  - Number of bead types: {n_types}")
    print(f"  - Box: [{xhi:.2f}, {yhi:.2f}, {zhi:.2f}] Å")
    print(f"  - Triclinic: {is_triclinic}")

    return output_path


def main():
    parser = argparse.ArgumentParser(description='Convert CG npz to LAMMPS data format')
    parser.add_argument('npz_path', help='Path to the CG npz file')
    parser.add_argument('output_path', nargs='?', default=None,
                        help='Output path for LAMMPS data file (default: <npz_name>.data)')
    parser.add_argument('--frame', '-f', type=int, default=0,
                        help='Frame index to extract (default: 0)')
    parser.add_argument('--masses', '-m', type=str, default=None,
                        help='Per-type masses as JSON (e.g., \'{"0": 18.015}\') or single float. '
                             'If not provided, reads from NPZ metadata or uses 72.0.')
    parser.add_argument('--comment', '-c', type=str, default=None,
                        help='Comment for the data file header')
    args = parser.parse_args()

    if args.output_path is None:
        args.output_path = args.npz_path.replace('.npz', '.data')

    # Parse masses argument
    masses = None
    if args.masses:
        try:
            masses = float(args.masses)
        except ValueError:
            import json
            raw = json.loads(args.masses)
            masses = {int(k): float(v) for k, v in raw.items()}

    npz_to_lammps_data(args.npz_path, args.output_path, args.frame, masses, args.comment)


if __name__ == '__main__':
    main()
