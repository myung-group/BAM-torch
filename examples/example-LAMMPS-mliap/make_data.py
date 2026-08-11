"""Convert an ASE-readable structure to a LAMMPS data file.

Usage: python make_data.py input.xyz output.data H C O
The element list fixes the LAMMPS type order and must match the
`pair_coeff * * <elements>` line in the input script.
"""
import sys
from ase.io import read, write

inp, out, *elements = sys.argv[1:]
atoms = read(inp)
write(out, atoms, format="lammps-data", specorder=elements, masses=True)
print(f"{out}: {len(atoms)} atoms, box {atoms.cell.lengths().round(2)}, types {elements}")
