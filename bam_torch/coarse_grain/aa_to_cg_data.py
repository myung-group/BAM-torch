"""
Convert AA LAMMPS data file to CG LAMMPS data file.

Supports any molecular system with configurable mapping:
- Water: H2O (3 atoms) -> 1 CG bead (default)
- Custom: User-defined atoms_per_molecule and bead definitions

Usage:
    python -m bam_torch.coarse_grain.aa_to_cg_data input.data output.data
    python -m bam_torch.coarse_grain.aa_to_cg_data input.data output.data --atoms-per-mol 5
"""
import os
import json
import numpy as np
import argparse


def read_lammps_data(filename):
    """Read LAMMPS data file."""
    with open(filename) as f:
        lines = f.readlines()

    n_atoms = None
    box = np.zeros((3, 2))
    masses = {}
    atoms = []

    i = 0
    while i < len(lines):
        line = lines[i].strip()

        if 'atoms' in line and n_atoms is None:
            n_atoms = int(line.split()[0])
        elif 'xlo xhi' in line:
            parts = line.split()
            box[0] = [float(parts[0]), float(parts[1])]
        elif 'ylo yhi' in line:
            parts = line.split()
            box[1] = [float(parts[0]), float(parts[1])]
        elif 'zlo zhi' in line:
            parts = line.split()
            box[2] = [float(parts[0]), float(parts[1])]
        elif line == 'Masses':
            i += 2  # skip blank line
            while i < len(lines) and lines[i].strip() and not lines[i].strip().startswith('Atoms'):
                parts = lines[i].split()
                if len(parts) >= 2:
                    type_id = int(parts[0])
                    mass = float(parts[1])
                    masses[type_id] = mass
                i += 1
            continue
        elif line.startswith('Atoms'):
            i += 2  # skip blank line
            while i < len(lines) and lines[i].strip():
                parts = lines[i].split()
                if len(parts) >= 5:
                    atom_id = int(parts[0])
                    atom_type = int(parts[1])
                    x, y, z = float(parts[2]), float(parts[3]), float(parts[4])
                    atoms.append({
                        'id': atom_id,
                        'type': atom_type,
                        'x': x, 'y': y, 'z': z,
                        'mass': masses.get(atom_type, 1.0)
                    })
                i += 1
            continue

        i += 1

    return atoms, box, masses


def cg_mapping(atoms, box, atoms_per_molecule=3):
    """
    CG mapping: group atoms into molecules and compute COM for each CG bead.

    Args:
        atoms: List of atom dicts from read_lammps_data
        box: Box dimensions (3, 2) array
        atoms_per_molecule: Number of atoms per molecule (default: 3 for water)

    Returns:
        List of CG bead dicts with id, type, x, y, z, mass
    """
    n_atoms = len(atoms)
    n_molecules = n_atoms // atoms_per_molecule

    cg_beads = []
    box_lengths = box[:, 1] - box[:, 0]

    for mol_idx in range(n_molecules):
        start = mol_idx * atoms_per_molecule
        mol_atoms = atoms[start:start + atoms_per_molecule]

        # Collect positions and masses
        positions = np.array([[a['x'], a['y'], a['z']] for a in mol_atoms])
        masses_arr = np.array([a['mass'] for a in mol_atoms])
        total_mass = masses_arr.sum()

        # Apply minimum image convention relative to first atom
        ref_pos = positions[0].copy()
        for j in range(1, len(positions)):
            for dim in range(3):
                delta = positions[j, dim] - ref_pos[dim]
                if delta > box_lengths[dim] / 2:
                    positions[j, dim] -= box_lengths[dim]
                elif delta < -box_lengths[dim] / 2:
                    positions[j, dim] += box_lengths[dim]

        # Compute COM
        com = (masses_arr[:, np.newaxis] * positions).sum(axis=0) / total_mass

        # Wrap COM back into box
        for dim in range(3):
            while com[dim] < box[dim, 0]:
                com[dim] += box_lengths[dim]
            while com[dim] >= box[dim, 1]:
                com[dim] -= box_lengths[dim]

        cg_beads.append({
            'id': mol_idx + 1,
            'type': 1,
            'x': com[0],
            'y': com[1],
            'z': com[2],
            'mass': total_mass
        })

    return cg_beads


def write_lammps_data(filename, cg_beads, box, bead_type_masses=None, comment="CG system"):
    """Write CG LAMMPS data file.

    Args:
        filename: Output file path
        cg_beads: List of CG bead dicts
        box: Box dimensions (3, 2) array
        bead_type_masses: Dict mapping type_id -> mass. If None, auto-detect from beads.
        comment: Header comment
    """
    if bead_type_masses is None:
        bead_type_masses = {}
        for bead in cg_beads:
            t = bead['type']
            if t not in bead_type_masses:
                bead_type_masses[t] = bead['mass']

    n_types = len(bead_type_masses)

    with open(filename, 'w') as f:
        f.write(f"# {comment}\n\n")
        f.write(f"{len(cg_beads)} atoms\n")
        f.write(f"{n_types} atom types\n\n")

        f.write(f"{box[0, 0]} {box[0, 1]} xlo xhi\n")
        f.write(f"{box[1, 0]} {box[1, 1]} ylo yhi\n")
        f.write(f"{box[2, 0]} {box[2, 1]} zlo zhi\n\n")

        f.write("Masses\n\n")
        for t in sorted(bead_type_masses.keys()):
            f.write(f"{t} {bead_type_masses[t]:.5f}\n")
        f.write("\n")

        f.write("Atoms  # atomic\n\n")
        for bead in cg_beads:
            f.write(f"{bead['id']} {bead['type']} {bead['x']:.8f} {bead['y']:.8f} {bead['z']:.8f}\n")


def main():
    parser = argparse.ArgumentParser(
        description='Convert AA LAMMPS data file to CG LAMMPS data file'
    )
    parser.add_argument('input', help='Input AA LAMMPS data file')
    parser.add_argument('output', nargs='?', default=None,
                        help='Output CG LAMMPS data file (default: <input>_cg.data)')
    parser.add_argument('--atoms-per-mol', type=int, default=3,
                        help='Atoms per molecule (default: 3 for water)')
    parser.add_argument('--masses', type=str, default=None,
                        help='Per-type masses as JSON (e.g., \'{"1": 18.015}\')')
    parser.add_argument('--comment', type=str, default=None,
                        help='Comment for output file header')

    args = parser.parse_args()

    if args.output is None:
        base = os.path.splitext(args.input)[0]
        args.output = f"{base}_cg.data"

    bead_type_masses = None
    if args.masses:
        raw = json.loads(args.masses)
        bead_type_masses = {int(k): float(v) for k, v in raw.items()}

    print("=" * 60)
    print("AA to CG LAMMPS data conversion")
    print("=" * 60)

    print(f"\nReading: {args.input}")
    atoms, box, masses = read_lammps_data(args.input)
    print(f"  Atoms: {len(atoms)}")
    print(f"  Box: {box[0,1]-box[0,0]:.3f} x {box[1,1]-box[1,0]:.3f} x {box[2,1]-box[2,0]:.3f}")
    print(f"  Masses: {masses}")

    print(f"\nCG mapping ({args.atoms_per_mol} atoms -> 1 bead)...")
    cg_beads = cg_mapping(atoms, box, args.atoms_per_mol)
    print(f"  CG beads: {len(cg_beads)}")

    comment = args.comment or f"CG from {os.path.basename(args.input)} ({args.atoms_per_mol}:1)"

    print(f"\nWriting: {args.output}")
    write_lammps_data(args.output, cg_beads, box, bead_type_masses, comment)

    print(f"\nDone!")
    print(f"\nFirst 5 CG beads:")
    for bead in cg_beads[:5]:
        print(f"  {bead['id']}: type={bead['type']} ({bead['x']:.4f}, {bead['y']:.4f}, {bead['z']:.4f})")


if __name__ == '__main__':
    main()
