"""Generate the LAMMPS data file (3bpa_300K.data) from the tutorial dataset.

The first frame of ../dataset/test_300K.xyz (the same structure used in the
ASE MD example) is converted to LAMMPS `data` format. `specorder` fixes the
LAMMPS atom-type order (1=C, 2=H, 3=N, 4=O) - it must match the element list
given to `pair_coeff` in race.in - and `masses=True` writes the matching
Masses section automatically.

Usage:
    python make_data.py
"""
from ase.io import read, write

atoms = read("../dataset/test_300K.xyz", index=0)
write("3bpa_300K.data", atoms, format="lammps-data",
      specorder=["C", "H", "N", "O"], masses=True)
print(f"Wrote 3bpa_300K.data ({len(atoms)} atoms)")
