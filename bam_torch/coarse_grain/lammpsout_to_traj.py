"""
Convert LAMMPS dump + log output to ASE trajectory.

Handles CG bead element names that ASE cannot recognize by replacing
them with valid element symbols.

Usage:
    python -m bam_torch.coarse_grain.lammpsout_to_traj --dump dump.lammpstrj --log log.lammps -o output.traj
    python -m bam_torch.coarse_grain.lammpsout_to_traj --dump dump.lammpstrj --log log.lammps --element-map '{"water": "Ar", "lipid": "C"}'

Or in Python:
    from bam_torch.coarse_grain.lammpsout_to_traj import lammps_dump_to_traj
    lammps_dump_to_traj('dump.lammpstrj', 'log.lammps', 'output.traj')
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import numpy as np
import argparse
import tempfile
import re

# Known elements that ASE can recognize
_KNOWN_ELEMENTS = {
    'H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne',
    'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca',
    'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
    'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Y', 'Zr',
    'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn',
    'Sb', 'Te', 'I', 'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd',
    'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb',
    'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg',
    'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Th',
    'Pa', 'U', 'Np', 'Pu',
}


def _detect_unknown_elements(content):
    """Detect element names in LAMMPS dump that ASE won't recognize."""
    unknown = set()
    in_atoms = False
    for line in content.splitlines():
        if line.startswith('ITEM: ATOMS'):
            in_atoms = True
            # Parse column names to find element column
            cols = line.split()[2:]  # skip "ITEM:" and "ATOMS"
            if 'element' in cols:
                elem_col = cols.index('element')
            else:
                elem_col = None
            continue
        elif line.startswith('ITEM:'):
            in_atoms = False
            continue

        if in_atoms and elem_col is not None:
            parts = line.split()
            if len(parts) > elem_col:
                elem = parts[elem_col]
                if elem not in _KNOWN_ELEMENTS:
                    unknown.add(elem)
    return unknown


def lammps_dump_to_traj(dump_path, log_path, output_path,
                        element_map=None, energy_column=2,
                        default_element='Ar'):
    """Convert LAMMPS dump + log to ASE trajectory.

    Args:
        dump_path: Path to LAMMPS dump file
        log_path: Path to LAMMPS log file
        output_path: Path for output ASE trajectory
        element_map: Dict mapping unknown element names to valid elements.
            e.g. {'water': 'Ar', 'lipid': 'C'}.
            If None, all unknown elements are mapped to default_element.
        energy_column: Column index (0-based) for energy in log file thermo output (default: 2)
        default_element: Default element symbol for unmapped bead types (default: 'Ar')

    Returns:
        output_path: Path where trajectory was saved
    """
    from ase.io import read
    from ase.io.trajectory import Trajectory
    from ase.calculators.singlepoint import SinglePointCalculator

    if element_map is None:
        element_map = {}

    # Read dump file
    with open(dump_path, 'r') as f:
        content = f.read()

    # Auto-detect unknown elements and apply mapping
    unknown = _detect_unknown_elements(content)
    if unknown:
        replacements = {}
        for elem in unknown:
            if elem in element_map:
                replacements[elem] = element_map[elem]
            else:
                replacements[elem] = default_element
        print(f"Element mapping: {replacements}")
        for old, new in replacements.items():
            content = content.replace(f' {old} ', f' {new} ')

    # Write to temp file and read with ASE
    with tempfile.NamedTemporaryFile(mode='w', suffix='.lammpstrj', delete=False) as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    try:
        atoms_list = read(tmp_path, index=slice(None), format='lammps-dump-text')
    finally:
        os.unlink(tmp_path)

    # Extract energies from log file
    energies = []
    with open(log_path) as log_file:
        collecting = False
        for line in log_file:
            line_split = line.split()
            if not line_split:
                continue

            if 'Step' in line_split:
                collecting = True
                continue
            elif 'Loop' in line_split:
                collecting = False
                continue

            if collecting and len(line_split) > energy_column:
                try:
                    energies.append(float(line_split[energy_column]))
                except (ValueError, IndexError):
                    pass

    # Match frame count
    if len(atoms_list) != len(energies):
        print(f"Warning: frame count ({len(atoms_list)}) != energy count ({len(energies)}). Truncating to shorter.")
        min_length = min(len(atoms_list), len(energies))
        atoms_list = atoms_list[:min_length]
        energies = energies[:min_length]

    # Save ASE trajectory
    with Trajectory(output_path, 'w') as traj:
        for atoms, energy in zip(atoms_list, energies):
            forces = atoms.arrays.get('forces', np.zeros((len(atoms), 3)))
            atoms.calc = SinglePointCalculator(
                atoms,
                energy=energy,
                forces=forces,
                stress=np.zeros(6)
            )
            atoms.info['potential_energy'] = energy
            traj.write(atoms)

    print(f"Saved {len(atoms_list)} frames to: {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description='Convert LAMMPS dump+log to ASE trajectory'
    )
    parser.add_argument('--dump', '-d', type=str, default='dump.lammpstrj',
                        help='Path to LAMMPS dump file (default: dump.lammpstrj)')
    parser.add_argument('--log', '-l', type=str, default='log.lammps',
                        help='Path to LAMMPS log file (default: log.lammps)')
    parser.add_argument('--output', '-o', type=str, default='lammps_out.traj',
                        help='Output ASE trajectory path (default: lammps_out.traj)')
    parser.add_argument('--element-map', '-e', type=str, default=None,
                        help='JSON dict mapping unknown element names to valid ones. '
                             'e.g. \'{"water": "Ar", "lipid": "C"}\'')
    parser.add_argument('--default-element', type=str, default='Ar',
                        help='Default element for unmapped bead types (default: Ar)')
    parser.add_argument('--energy-column', type=int, default=2,
                        help='Column index (0-based) for energy in log thermo output (default: 2)')

    args = parser.parse_args()

    element_map = None
    if args.element_map:
        import json
        element_map = json.loads(args.element_map)

    lammps_dump_to_traj(
        dump_path=args.dump,
        log_path=args.log,
        output_path=args.output,
        element_map=element_map,
        energy_column=args.energy_column,
        default_element=args.default_element
    )


if __name__ == '__main__':
    main()
