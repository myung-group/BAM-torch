"""
QM9star PostgreSQL dump -> xyz conversion preprocessing script.

Extracts snapshot/molecule data from a PostgreSQL binary dump file and
converts it to an extended .xyz file readable by ASE.

Usage:
  1) With pg_restore available (recommended):
     python qm9star_preprocessor.py --dump qm9star_archive_240912.sql --method pg_restore

  2) With an existing plain SQL file (no pg_restore needed):
     python qm9star_preprocessor.py --sql qm9star_plain.sql --method parse_sql

Unit conversions:
  - Energy: Hartree -> eV (x27.2114)
  - Forces: Hartree/bohr -> eV/Ang (x51.4221)
"""

import argparse
import json
import os
import re
import subprocess
import sys
import numpy as np
from pathlib import Path


# Unit conversion constants
HARTREE_TO_EV = 27.211386245988  # eV/Hartree
BOHR_TO_ANGSTROM = 0.529177249   # Ang/bohr
HARTREE_BOHR_TO_EV_ANGSTROM = HARTREE_TO_EV / BOHR_TO_ANGSTROM  # eV/Ang per Hartree/bohr

# Element symbol -> atomic number mapping
ELEMENT_SYMBOLS = [
    'X',  # 0
    'H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne',
    'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar',
    'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
    'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr',
    'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd',
    'In', 'Sn', 'Sb', 'Te', 'I', 'Xe',
    'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy',
    'Ho', 'Er', 'Tm', 'Yb', 'Lu',
    'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg',
    'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn',
]


def dump_to_plain_sql(dump_path, output_sql_path):
    """Convert binary dump to plain SQL using pg_restore."""
    print(f"[1/3] Converting to plain SQL via pg_restore...")
    print(f"  Input: {dump_path}")
    print(f"  Output: {output_sql_path}")

    try:
        result = subprocess.run(
            ['pg_restore', '-f', str(output_sql_path), str(dump_path)],
            capture_output=True, text=True
        )
        if result.returncode != 0:
            # pg_restore can output with -f option without DB connection
            # Some warnings can be ignored
            if "database" in result.stderr.lower():
                print(f"  Warning (ignorable): {result.stderr[:200]}")
            else:
                print(f"  Error: {result.stderr[:500]}")
                return False
        print(f"  Done!")
        return True
    except FileNotFoundError:
        print("  pg_restore not found.")
        print("  Install: sudo apt install postgresql-client")
        return False


def parse_pg_array(s):
    """Convert PostgreSQL array string to Python list.
    e.g.: '{1,2,3}' -> [1, 2, 3]
          '{{1,2,3},{4,5,6}}' -> [[1,2,3],[4,5,6]]
    """
    if s is None or s == '\\N' or s == '':
        return None

    s = s.strip()
    if not s.startswith('{'):
        return None

    # Handle nested arrays
    depth = 0
    for c in s:
        if c == '{':
            depth += 1
        else:
            break

    if depth == 1:
        # 1D array
        inner = s[1:-1]
        if not inner:
            return []
        return [float(x) if '.' in x or 'e' in x.lower()
                else int(x)
                for x in inner.split(',')]
    elif depth == 2:
        # 2D array {{1,2,3},{4,5,6}}
        inner = s[1:-1]  # {1,2,3},{4,5,6}
        rows = []
        current_row = []
        in_row = False
        buf = ""
        for c in inner:
            if c == '{':
                in_row = True
                buf = ""
            elif c == '}':
                if buf:
                    current_row.append(
                        float(buf) if '.' in buf or 'e' in buf.lower()
                        else int(buf)
                    )
                rows.append(current_row)
                current_row = []
                in_row = False
                buf = ""
            elif c == ',':
                if in_row:
                    if buf:
                        current_row.append(
                            float(buf) if '.' in buf or 'e' in buf.lower()
                            else int(buf)
                        )
                    buf = ""
                # Commas between rows are ignored
            else:
                buf += c
        return rows
    return None


def parse_copy_data(lines, table_name, columns):
    """
    Parse data rows following a COPY statement.

    Args:
        lines: data lines after the COPY statement in the SQL file
        table_name: table name
        columns: list of column names

    Returns:
        list of dict, each dict is {column_name: value}
    """
    records = []
    for line in lines:
        line = line.rstrip('\n')
        if line == '\\.':
            break
        values = line.split('\t')
        record = {}
        for col, val in zip(columns, values):
            if val == '\\N':
                record[col] = None
            else:
                record[col] = val
        records.append(record)
    return records


def parse_sql_file(sql_path, charge_type="npa_charges",
                   energy_type="U_0", max_samples=None):
    """
    Read snapshot + molecule data from a plain SQL file.

    Args:
        sql_path: plain SQL file path
        charge_type: charge type to use
        energy_type: energy type to use
        max_samples: maximum number of samples (None for all)

    Returns:
        list of dict with keys:
        - atoms, coords, forces, energy
        - atomic_charges, total_charge
        - smiles, molecule_id
    """
    print(f"[2/3] Parsing SQL file: {sql_path}")
    print(f"  charge_type: {charge_type}")
    print(f"  energy_type: {energy_type}")

    # First read total_charge and total_multiplicity from molecule table
    molecules = {}  # molecule_id -> (total_charge, total_multiplicity)

    snapshot_columns = []
    molecule_columns = []
    snapshots = []

    with open(sql_path, 'r', encoding='utf-8', errors='replace') as f:
        in_copy = False
        current_table = None
        current_columns = []
        data_lines = []

        for line in f:
            # Detect COPY statement start
            if line.startswith('COPY '):
                match = re.match(
                    r'COPY\s+(?:public\.)?(\S+)\s+\((.+?)\)\s+FROM\s+stdin',
                    line
                )
                if match:
                    current_table = match.group(1).strip('"')
                    col_str = match.group(2)
                    current_columns = [
                        c.strip().strip('"')
                        for c in col_str.split(',')
                    ]
                    in_copy = True
                    data_lines = []
                    continue

            if in_copy:
                if line.rstrip('\n') == '\\.':
                    # End of COPY data
                    if current_table == 'molecule':
                        records = parse_copy_data(
                            data_lines, current_table, current_columns
                        )
                        for r in records:
                            mid = r.get('id')
                            tc = r.get('total_charge')
                            tm = r.get('total_multiplicity')
                            if mid and tc is not None:
                                molecules[mid] = (
                                    float(tc),
                                    int(tm) if tm is not None else 1,
                                )
                        print(f"  molecule table: {len(molecules)} records loaded")

                    elif current_table == 'snapshot':
                        records = parse_copy_data(
                            data_lines, current_table, current_columns
                        )
                        snapshots = records
                        print(f"  snapshot table: {len(snapshots)} records loaded")

                    in_copy = False
                    current_table = None
                    continue

                data_lines.append(line)

    # Data conversion
    print(f"\n[3/3] Converting data...")
    results = []
    skipped = 0

    for i, snap in enumerate(snapshots):
        if max_samples and len(results) >= max_samples:
            break

        try:
            # Atom information
            atoms = parse_pg_array(snap.get('atoms'))
            coords = parse_pg_array(snap.get('coords'))
            if atoms is None or coords is None:
                skipped += 1
                continue

            # Energy
            energy_val = snap.get(energy_type)
            if energy_val is None or energy_val == '':
                energy_val = snap.get('single_point_energy')
            if energy_val is None:
                skipped += 1
                continue
            energy = float(energy_val) * HARTREE_TO_EV

            # Forces
            forces_raw = parse_pg_array(snap.get('forces'))
            if forces_raw is not None:
                forces = np.array(forces_raw) * HARTREE_BOHR_TO_EV_ANGSTROM
            else:
                forces = np.zeros((len(atoms), 3))

            # Charges
            charges_raw = parse_pg_array(snap.get(charge_type))
            if charges_raw is not None:
                atomic_charges = np.array(charges_raw, dtype=float)
            else:
                # fallback
                for ct in ['npa_charges', 'mulliken_charge',
                           'hirshfeld_charges', 'formal_charges']:
                    charges_raw = parse_pg_array(snap.get(ct))
                    if charges_raw is not None:
                        atomic_charges = np.array(charges_raw, dtype=float)
                        break
                else:
                    atomic_charges = np.zeros(len(atoms))

            # Total charge & multiplicity (from molecule table)
            mol_id = snap.get('molecule_id')
            mol_info = molecules.get(mol_id, (0.0, 1))
            total_charge = mol_info[0]
            total_multiplicity = mol_info[1]

            result = {
                'atoms': np.array(atoms, dtype=int),
                'coords': np.array(coords, dtype=float),
                'forces': forces.tolist() if isinstance(forces, np.ndarray)
                          else forces,
                'energy': energy,
                'atomic_charges': atomic_charges.tolist()
                    if isinstance(atomic_charges, np.ndarray)
                    else atomic_charges,
                'total_charge': total_charge,
                'total_multiplicity': total_multiplicity,
                'molecule_id': mol_id,
                'filename': snap.get('filename', ''),
            }
            results.append(result)

        except Exception as e:
            skipped += 1
            if skipped <= 5:
                print(f"  Warning: skipping record {i} — {e}")

    print(f"  Conversion complete: {len(results)} succeeded, {skipped} skipped")
    return results


def write_extended_xyz(results, output_path, max_per_file=None):
    """
    Save converted data in extended xyz format.

    Extended xyz readable by ASE:
    - Line 1: number of atoms
    - Line 2: Properties info (Lattice, energy, pbc, etc.)
    - Remaining: element x y z fx fy fz charge
    """
    print(f"\n  Saving xyz file: {output_path}")

    with open(output_path, 'w') as f:
        for i, result in enumerate(results):
            if max_per_file and i >= max_per_file:
                break

            atoms = result['atoms']
            coords = result['coords']
            forces = result['forces']
            energy = result['energy']
            charges = result['atomic_charges']
            total_charge = result['total_charge']
            total_multiplicity = result.get('total_multiplicity', 1)
            n_atoms = len(atoms)

            # Extended XYZ first line
            f.write(f"{n_atoms}\n")

            # Properties line
            lattice = "30.0 0.0 0.0 0.0 30.0 0.0 0.0 0.0 30.0"
            props = (
                f'Lattice="{lattice}" '
                f'Properties=species:S:1:pos:R:3:forces:R:3:charges:R:1 '
                f'energy={energy} '
                f'total_charge={total_charge} '
                f'total_multiplicity={total_multiplicity} '
                f'pbc="F F F"'
            )
            f.write(f"{props}\n")

            # Atom data
            for j in range(n_atoms):
                z = int(atoms[j])
                symbol = ELEMENT_SYMBOLS[z] if z < len(ELEMENT_SYMBOLS) else f"X{z}"
                x, y, zz = coords[j]
                if isinstance(forces, (list, np.ndarray)) and len(forces) > j:
                    fx, fy, fz = forces[j]
                else:
                    fx, fy, fz = 0.0, 0.0, 0.0
                q = charges[j] if j < len(charges) else 0.0
                f.write(
                    f"{symbol:2s} {x:16.8f} {y:16.8f} {zz:16.8f} "
                    f"{fx:16.8f} {fy:16.8f} {fz:16.8f} {q:12.6f}\n"
                )

    print(f"  {min(len(results), max_per_file or len(results))} structures saved")


def _convert_snapshot(snap, molecules, charge_type, energy_type):
    """Convert a single snapshot record to result dict. Returns None on failure."""
    atoms = parse_pg_array(snap.get('atoms'))
    coords = parse_pg_array(snap.get('coords'))
    if atoms is None or coords is None:
        return None

    energy_val = snap.get(energy_type)
    if energy_val is None or energy_val == '':
        energy_val = snap.get('single_point_energy')
    if energy_val is None:
        return None
    energy = float(energy_val) * HARTREE_TO_EV

    forces_raw = parse_pg_array(snap.get('forces'))
    if forces_raw is not None:
        forces = (np.array(forces_raw) * HARTREE_BOHR_TO_EV_ANGSTROM).tolist()
    else:
        forces = [[0.0, 0.0, 0.0]] * len(atoms)

    charges_raw = parse_pg_array(snap.get(charge_type))
    if charges_raw is not None:
        atomic_charges = [float(x) for x in charges_raw]
    else:
        for ct in ['npa_charges', 'mulliken_charge',
                    'hirshfeld_charges', 'formal_charges']:
            charges_raw = parse_pg_array(snap.get(ct))
            if charges_raw is not None:
                atomic_charges = [float(x) for x in charges_raw]
                break
        else:
            atomic_charges = [0.0] * len(atoms)

    mol_id = snap.get('molecule_id')
    mol_info = molecules.get(mol_id, (0.0, 1))

    return {
        'atoms': atoms,
        'coords': coords,
        'forces': forces,
        'energy': energy,
        'atomic_charges': atomic_charges,
        'total_charge': mol_info[0],
        'total_multiplicity': mol_info[1],
    }


def _convert_snapshot_full(snap, molecules, energy_type):
    """Convert a single snapshot with ALL available data columns."""
    atoms = parse_pg_array(snap.get('atoms'))
    coords = parse_pg_array(snap.get('coords'))
    if atoms is None or coords is None:
        return None
    n_atoms = len(atoms)

    # Primary energy
    energy_val = snap.get(energy_type)
    if energy_val is None or energy_val == '':
        energy_val = snap.get('single_point_energy')
    if energy_val is None:
        return None
    energy = float(energy_val) * HARTREE_TO_EV

    # Forces (Hartree/bohr -> eV/Ang)
    forces_raw = parse_pg_array(snap.get('forces'))
    forces = ((np.array(forces_raw) * HARTREE_BOHR_TO_EV_ANGSTROM).tolist()
              if forces_raw is not None else [[0.0, 0.0, 0.0]] * n_atoms)

    # --- Per-atom arrays ---
    per_atom = {}
    for db_col, key in [('npa_charges', 'npa'), ('mulliken_charge', 'mulliken'),
                        ('lowdin_charges', 'lowdin'), ('hirshfeld_charges', 'hirshfeld'),
                        ('formal_charges', 'formal')]:
        parsed = parse_pg_array(snap.get(db_col))
        per_atom[key] = ([float(x) for x in parsed]
                         if parsed and len(parsed) == n_atoms
                         else [0.0] * n_atoms)

    sd = parse_pg_array(snap.get('spin_densities'))
    per_atom['spin_densities'] = ([float(x) for x in sd]
                                  if sd and len(sd) == n_atoms
                                  else [0.0] * n_atoms)

    # --- Per-structure scalar properties ---
    info = {}

    # Energies (Hartree -> eV)
    for key in ['single_point_energy', 'zpve', 'U_0', 'U_T', 'H_T', 'G_T',
                'energy_correction', 'enthalpy_correction',
                'gibbs_free_energy_correction']:
        val = snap.get(key)
        if val and val != '\\N':
            try:
                info[key] = float(val) * HARTREE_TO_EV
            except ValueError:
                pass

    # Orbital energies (Hartree -> eV)
    for key in ['alpha_homo', 'alpha_lumo', 'alpha_gap',
                'beta_homo', 'beta_lumo', 'beta_gap']:
        val = snap.get(key)
        if val and val != '\\N':
            try:
                info[key] = float(val) * HARTREE_TO_EV
            except ValueError:
                pass

    # Properties kept in original units
    for key in ['S', 'Cv', 'isotropic_polarizability',
                'electronic_spatial_extent',
                'spin_quantum_number', 'spin_square']:
        val = snap.get(key)
        if val and val != '\\N':
            try:
                info[key] = float(val)
            except ValueError:
                pass

    # Dipole components (Debye)
    dipole = parse_pg_array(snap.get('dipole'))
    if dipole and len(dipole) >= 3:
        info['dipole_x'] = float(dipole[0])
        info['dipole_y'] = float(dipole[1])
        info['dipole_z'] = float(dipole[2])

    # Boolean flags
    for key in ['is_TS', 'is_optimized', 'is_error']:
        val = snap.get(key)
        if val and val != '\\N':
            info[key] = 1 if val.lower() in ('t', 'true', '1') else 0

    # Molecule info
    mol_id = snap.get('molecule_id')
    mol_info = molecules.get(mol_id, {})

    return {
        'atoms': atoms,
        'coords': coords,
        'forces': forces,
        'energy': energy,
        'per_atom': per_atom,
        'info': info,
        'total_charge': mol_info.get('total_charge', 0.0),
        'total_multiplicity': mol_info.get('total_multiplicity', 1),
        'smiles': mol_info.get('smiles', ''),
    }


# Per-atom Properties header for full mode
_FULL_PROPS = (
    "Properties=species:S:1:pos:R:3:forces:R:3"
    ":npa_charges:R:1:mulliken_charges:R:1"
    ":lowdin_charges:R:1:hirshfeld_charges:R:1"
    ":formal_charges:R:1:spin_densities:R:1"
)


def _write_one_xyz(f, result):
    """Write a single structure to an open XYZ file handle (basic mode)."""
    atoms = result['atoms']
    coords = result['coords']
    forces = result['forces']
    energy = result['energy']
    charges = result['atomic_charges']
    total_charge = result['total_charge']
    total_multiplicity = result.get('total_multiplicity', 1)
    n_atoms = len(atoms)

    f.write(f"{n_atoms}\n")
    lattice = "30.0 0.0 0.0 0.0 30.0 0.0 0.0 0.0 30.0"
    f.write(
        f'Lattice="{lattice}" '
        f'Properties=species:S:1:pos:R:3:forces:R:3:charges:R:1 '
        f'energy={energy} '
        f'total_charge={total_charge} '
        f'total_multiplicity={total_multiplicity} '
        f'pbc="F F F"\n'
    )
    for j in range(n_atoms):
        z = int(atoms[j])
        symbol = ELEMENT_SYMBOLS[z] if z < len(ELEMENT_SYMBOLS) else f"X{z}"
        x, y, zz = coords[j]
        fx, fy, fz = forces[j] if j < len(forces) else (0.0, 0.0, 0.0)
        q = charges[j] if j < len(charges) else 0.0
        f.write(
            f"{symbol:2s} {x:16.8f} {y:16.8f} {zz:16.8f} "
            f"{fx:16.8f} {fy:16.8f} {fz:16.8f} {q:12.6f}\n"
        )


def _write_one_xyz_full(f, result):
    """Write a single structure with ALL data columns."""
    atoms = result['atoms']
    coords = result['coords']
    forces = result['forces']
    pa = result['per_atom']
    info = result['info']
    n_atoms = len(atoms)

    f.write(f"{n_atoms}\n")

    # Info line: lattice + properties header + all scalar key=value pairs
    lattice = "30.0 0.0 0.0 0.0 30.0 0.0 0.0 0.0 30.0"
    parts = [
        f'Lattice="{lattice}"',
        _FULL_PROPS,
        f'energy={result["energy"]}',
        f'total_charge={result["total_charge"]}',
        f'total_multiplicity={result["total_multiplicity"]}',
    ]

    # SMILES (quote to protect special chars)
    smiles = result.get('smiles', '')
    if smiles:
        safe = smiles.replace('"', '\\"')
        parts.append(f'smiles="{safe}"')

    # All scalar info properties
    for key, val in info.items():
        if isinstance(val, float):
            parts.append(f'{key}={val}')
        else:
            parts.append(f'{key}={val}')

    parts.append('pbc="F F F"')
    f.write(' '.join(parts) + '\n')

    # Atom lines
    for j in range(n_atoms):
        z = int(atoms[j])
        sym = ELEMENT_SYMBOLS[z] if z < len(ELEMENT_SYMBOLS) else f"X{z}"
        x, y, zz = coords[j]
        fx, fy, fz = forces[j] if j < len(forces) else (0.0, 0.0, 0.0)
        npa = pa['npa'][j]
        mul = pa['mulliken'][j]
        low = pa['lowdin'][j]
        hir = pa['hirshfeld'][j]
        frm = pa['formal'][j]
        sd = pa['spin_densities'][j]
        f.write(
            f"{sym:2s} {x:16.8f} {y:16.8f} {zz:16.8f} "
            f"{fx:16.8f} {fy:16.8f} {fz:16.8f} "
            f"{npa:12.6f} {mul:12.6f} {low:12.6f} "
            f"{hir:12.6f} {frm:12.6f} {sd:12.6f}\n"
        )


def parse_and_write_streaming(sql_path, output_path, charge_type="npa_charges",
                              energy_type="U_0", max_samples=None,
                              full=False):
    """Memory-efficient 2-pass streaming: parse SQL and write XYZ one record at a time.

    Pass 1: read molecule table only (small, ~100K entries).
    Pass 2: stream snapshot records -> convert -> write XYZ immediately.
    Peak memory: O(molecules) + O(1 snapshot), NOT O(all snapshots).

    Args:
        full: if True, extract ALL DB columns (per-atom charges ×5,
              spin_densities, all energies, orbital props, dipole, flags, smiles).
    """
    # --- Pass 1: molecule table ---
    print(f"[Pass 1/2] Reading molecule table from: {sql_path}")
    molecules = {}

    with open(sql_path, 'r', encoding='utf-8', errors='replace') as f:
        in_copy = False
        current_columns = []

        for line in f:
            if line.startswith('COPY '):
                match = re.match(
                    r'COPY\s+(?:public\.)?(\S+)\s+\((.+?)\)\s+FROM\s+stdin',
                    line,
                )
                if match:
                    table = match.group(1).strip('"')
                    if table == 'molecule':
                        col_str = match.group(2)
                        current_columns = [
                            c.strip().strip('"') for c in col_str.split(',')
                        ]
                        in_copy = True
                    else:
                        in_copy = False
                continue

            if in_copy:
                if line.rstrip('\n') == '\\.':
                    in_copy = False
                    continue
                values = line.rstrip('\n').split('\t')
                record = {}
                for col, val in zip(current_columns, values):
                    record[col] = None if val == '\\N' else val
                mid = record.get('id')
                tc = record.get('total_charge')
                tm = record.get('total_multiplicity')
                if mid and tc is not None:
                    if full:
                        molecules[mid] = {
                            'total_charge': float(tc),
                            'total_multiplicity': int(tm) if tm is not None else 1,
                            'smiles': record.get('smiles', '') or '',
                        }
                    else:
                        molecules[mid] = (
                            float(tc),
                            int(tm) if tm is not None else 1,
                        )

    print(f"  molecule table: {len(molecules)} records loaded")

    # --- Pass 2: stream snapshot -> XYZ ---
    mode_str = "FULL" if full else "basic"
    print(f"\n[Pass 2/2] Streaming snapshot -> XYZ ({mode_str}): {output_path}")
    count = 0
    skipped = 0
    e_min, e_max = float('inf'), float('-inf')
    q_min, q_max = float('inf'), float('-inf')

    with open(sql_path, 'r', encoding='utf-8', errors='replace') as f, \
         open(output_path, 'w') as out:

        in_copy = False
        current_columns = []

        for line in f:
            if line.startswith('COPY '):
                match = re.match(
                    r'COPY\s+(?:public\.)?(\S+)\s+\((.+?)\)\s+FROM\s+stdin',
                    line,
                )
                if match:
                    table = match.group(1).strip('"')
                    if table == 'snapshot':
                        col_str = match.group(2)
                        current_columns = [
                            c.strip().strip('"') for c in col_str.split(',')
                        ]
                        in_copy = True
                    else:
                        in_copy = False
                continue

            if not in_copy:
                continue

            if line.rstrip('\n') == '\\.':
                in_copy = False
                continue

            if max_samples and count >= max_samples:
                continue

            # Parse one snapshot record inline
            values = line.rstrip('\n').split('\t')
            snap = {}
            for col, val in zip(current_columns, values):
                snap[col] = None if val == '\\N' else val

            try:
                if full:
                    result = _convert_snapshot_full(
                        snap, molecules, energy_type)
                else:
                    result = _convert_snapshot(
                        snap, molecules, charge_type, energy_type)
            except Exception as e:
                skipped += 1
                if skipped <= 5:
                    print(f"  Warning: skipping — {e}")
                continue

            if result is None:
                skipped += 1
                continue

            if full:
                _write_one_xyz_full(out, result)
                npa = result['per_atom']['npa']
                for q in npa:
                    if q < q_min:
                        q_min = q
                    if q > q_max:
                        q_max = q
            else:
                _write_one_xyz(out, result)
                for q in result['atomic_charges']:
                    if q < q_min:
                        q_min = q
                    if q > q_max:
                        q_max = q

            e_min = min(e_min, result['energy'])
            e_max = max(e_max, result['energy'])

            count += 1
            if count % 100000 == 0:
                print(f"  ... {count:,} structures written")

    print(f"  Complete: {count:,} succeeded, {skipped:,} skipped")
    print(f"\n=== Data Statistics ===")
    print(f"  Total structures: {count:,}")
    if count > 0:
        print(f"  Energy range: {e_min:.4f} ~ {e_max:.4f} eV")
        print(f"  NPA charge range: {q_min:.4f} ~ {q_max:.4f}")
    print(f"  Output file: {output_path}")
    return count


def extract_stratified_subset(input_path, output_path, n_samples,
                              random_seed=42):
    """Extract a stratified subset from an extended XYZ file.

    Groups structures by (total_charge, total_multiplicity) and samples
    equally from each group. Memory-efficient: only stores metadata per
    structure (~24 bytes each), not the actual atom data.

    Args:
        input_path: source extended XYZ file
        output_path: output XYZ file
        n_samples: total number of samples to extract
        random_seed: random seed for reproducibility
    """
    # --- Pass 1: scan and index structures ---
    print(f"[Pass 1/2] Scanning {input_path}...")
    # Store (n_atoms, group_key) per structure — minimal memory
    meta = []  # [(n_atoms, (charge, mult)), ...]

    with open(input_path, 'r') as f:
        while True:
            line = f.readline()
            if not line:
                break
            n_atoms = int(line.strip())
            info_line = f.readline()

            tc_m = re.search(r'total_charge=([\-\d.]+)', info_line)
            tm_m = re.search(r'total_multiplicity=(\d+)', info_line)
            tc = float(tc_m.group(1)) if tc_m else 0.0
            tm = int(tm_m.group(1)) if tm_m else 1

            meta.append((n_atoms, (tc, tm)))
            for _ in range(n_atoms):
                f.readline()

    total = len(meta)
    print(f"  Total structures: {total:,}")

    # Group indices by (charge, mult)
    groups = {}
    for i, (_, gk) in enumerate(meta):
        groups.setdefault(gk, []).append(i)

    print(f"  Groups found: {len(groups)}")
    for gk in sorted(groups):
        print(f"    charge={gk[0]:+.0f}, mult={gk[1]}: {len(groups[gk]):,}")

    # --- Stratified sampling: equal allocation per group ---
    rng = np.random.RandomState(random_seed)
    n_groups = len(groups)
    per_group = n_samples // n_groups
    remainder = n_samples % n_groups

    selected = set()
    group_counts = {}
    for i, gk in enumerate(sorted(groups)):
        n = per_group + (1 if i < remainder else 0)
        n = min(n, len(groups[gk]))
        chosen = rng.choice(groups[gk], size=n, replace=False)
        selected.update(chosen)
        group_counts[gk] = n

    print(f"\n  Selected: {len(selected):,} structures")
    for gk in sorted(group_counts):
        print(f"    charge={gk[0]:+.0f}, mult={gk[1]}: {group_counts[gk]:,}")

    # --- Pass 2: write selected structures ---
    print(f"\n[Pass 2/2] Writing {output_path}...")
    written = 0
    with open(input_path, 'r') as fin, open(output_path, 'w') as fout:
        idx = 0
        while True:
            line = fin.readline()
            if not line:
                break
            n_atoms = meta[idx][0]
            info_line = fin.readline()
            keep = idx in selected

            if keep:
                fout.write(line)
                fout.write(info_line)

            for _ in range(n_atoms):
                atom_line = fin.readline()
                if keep:
                    fout.write(atom_line)

            if keep:
                written += 1
            idx += 1

    print(f"  Written: {written:,} structures -> {output_path}")
    return written


def main():
    parser = argparse.ArgumentParser(
        description='QM9star PostgreSQL dump -> extended XYZ conversion'
    )
    parser.add_argument(
        '--dump', type=str, default=None,
        help='PostgreSQL binary dump file path'
    )
    parser.add_argument(
        '--sql', type=str, default=None,
        help='Plain SQL file path (after pg_restore)'
    )
    parser.add_argument(
        '--output', type=str, default='qm9star_data.xyz',
        help='Output xyz file path'
    )
    parser.add_argument(
        '--charge-type', type=str, default='npa_charges',
        choices=['formal_charges', 'mulliken_charge', 'npa_charges',
                 'hirshfeld_charges', 'lowdin_charges'],
        help='Charge type to use (default: npa_charges)'
    )
    parser.add_argument(
        '--energy-type', type=str, default='U_0',
        choices=['single_point_energy', 'U_0', 'U_T', 'H_T', 'G_T'],
        help='Energy type to use (default: U_0)'
    )
    parser.add_argument(
        '--max-samples', type=int, default=None,
        help='Maximum number of samples'
    )
    parser.add_argument(
        '--full', action='store_true',
        help='Extract ALL DB columns (charges x5, spin_densities, all energies, '
             'orbital props, dipole, flags, smiles)'
    )
    parser.add_argument(
        '--no-streaming', action='store_true',
        help='Disable streaming mode (load all into memory, for small datasets)'
    )

    # Stratified subset extraction from existing XYZ
    parser.add_argument(
        '--extract-subset', type=str, default=None, metavar='INPUT_XYZ',
        help='Extract stratified subset from an existing XYZ file '
             '(groups by charge×multiplicity, equal allocation per group). '
             'Use with --output and --max-samples.'
    )
    parser.add_argument(
        '--seed', type=int, default=42,
        help='Random seed for stratified sampling (default: 42)'
    )
    args = parser.parse_args()

    # --- Mode: stratified subset extraction ---
    if args.extract_subset:
        if args.max_samples is None:
            print("Error: --max-samples required with --extract-subset")
            sys.exit(1)
        extract_stratified_subset(
            args.extract_subset, args.output,
            n_samples=args.max_samples,
            random_seed=args.seed,
        )
        return

    # Step 1: binary dump -> plain SQL (if needed)
    sql_path = args.sql
    if sql_path is None and args.dump is not None:
        sql_path = args.dump.replace('.sql', '_plain.sql')
        if not os.path.exists(sql_path):
            success = dump_to_plain_sql(args.dump, sql_path)
            if not success:
                print("\nAlternative: cannot proceed without pg_restore.")
                print("  sudo apt install postgresql-client-16")
                sys.exit(1)
        else:
            print(f"  Using existing converted SQL file: {sql_path}")

    if sql_path is None:
        print("Please specify either --dump or --sql.")
        sys.exit(1)

    if args.no_streaming:
        # Legacy: load all into memory (small datasets only)
        results = parse_sql_file(
            sql_path,
            charge_type=args.charge_type,
            energy_type=args.energy_type,
            max_samples=args.max_samples,
        )
        if not results:
            print("No data found.")
            sys.exit(1)
        write_extended_xyz(results, args.output, args.max_samples)
        energies = [r['energy'] for r in results]
        charges_flat = [q for r in results for q in r['atomic_charges']]
        print(f"\n=== Data Statistics ===")
        print(f"  Total structures: {len(results)}")
        print(f"  Energy range: {min(energies):.4f} ~ {max(energies):.4f} eV")
        print(f"  Charge range: {min(charges_flat):.4f} ~ {max(charges_flat):.4f}")
        print(f"  Output file: {args.output}")
    else:
        # Default: memory-efficient streaming
        count = parse_and_write_streaming(
            sql_path, args.output,
            charge_type=args.charge_type,
            energy_type=args.energy_type,
            max_samples=args.max_samples,
            full=args.full,
        )
        if count == 0:
            print("No data found.")
            sys.exit(1)


if __name__ == '__main__':
    main()
