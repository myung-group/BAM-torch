import numpy as np
from ase import Atoms
from ase.io import read, write
from ase.io.trajectory import Trajectory
from ase.calculators.singlepoint import SinglePointCalculator


atoms_list = read("dump.lammpstrj", index=slice(None))

# Extract from log.lammps file
energies = []
with open("log.lammps") as log_file:
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
            
        if collecting and len(line_split) > 2:
            try:
                energies.append(float(line_split[2]))
            except (ValueError, IndexError):
                pass

# Check sync the number of structures and energies
if len(atoms_list) != len(energies):
    print(f"Warning: Num of frames({len(atoms_list)})and Num of energies({len(energies)}) are not same.")
    # Adjust the length of list
    min_length = min(len(atoms_list), len(energies))
    atoms_list = atoms_list[:min_length]
    energies = energies[:min_length]

# Save as ase.traj file
with Trajectory('lammps_out.traj', 'w') as traj:
    for atoms, energy in zip(atoms_list, energies):
        atoms.calc = SinglePointCalculator(
            atoms,
            energy=energy,
            forces=atoms.get_forces(),
            stress=np.zeros(6)
        )
        atoms.info['potential_energy'] = energy
        traj.write(atoms)
