from ase.build import bulk
from ase.calculators.emt import EMT
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary
from ase.md.npt import NPT
from ase.md import MDLogger
from ase.io.trajectory import Trajectory
from ase import units
from time import perf_counter


from ase.io import read
import numpy as np
from bam_torch.tase.base_calculator import RACECalculator
import json


# 1. Get initial structure
atoms = read('../example-si64/valid.traj', index=slice(None))
energies = [a.get_potential_energy() for a in atoms]
min_index = np.argmin(energies)
stable_atoms = atoms[min_index]

# 2. Assign calculator
with open('input.json') as f:
    json_data = json.load(f)
stable_atoms.calc = RACECalculator(json_data)

# 3. Initialize velocities using Maxwell-Boltzmann distribution at 300 K
target_T = 300  # Kelvin
MaxwellBoltzmannDistribution(stable_atoms, temperature_K=target_T, force_temp=True)
Stationary(stable_atoms)  # remove net linear momentum

# 4. Set up NPT dynamics
timestep = 1.0 * units.fs           # 1 fs time step
sigma = 1.0 * units.bar             # external pressure = 1 bar
ttime = 20.0 * units.fs             # thermostat time constant
pfactor = 2.0e6 * units.GPa * (units.fs ** 2)  # barostat coupling parameter

logfile = f"si_npt_{target_T}K.log"
trajfile = f"si_npt_{target_T}K.traj"
nsteps = 10000   # = 10 ps

# Open trajectory file for writing
traj = Trajectory(trajfile, 'w', stable_atoms)

dyn = NPT(
    stable_atoms,
    timestep,
    temperature_K=target_T,
    externalstress=sigma,     # isotropic external pressure
    ttime=ttime,              # thermostat coupling
    pfactor=pfactor,          # barostat strength
    logfile=logfile,
    trajectory=traj,
    loginterval=100           # write log/trajectory every 100 steps
)

# 5. Simple status printer (energy, temperature, pressure)
start_time = perf_counter()

def print_status():
    step = dyn.get_number_of_steps()
    etot = stable_atoms.get_total_energy()
    temp = stable_atoms.get_temperature()
    stress = stable_atoms.get_stress(include_ideal_gas=True) / units.GPa
    p_mean = (stress[0] + stress[1] + stress[2]) / 3.0
    elapsed = perf_counter() - start_time

    print(f"{step:6d}  Etot = {etot:8.3f} eV  "
          f"T = {temp:7.2f} K  "
          f"P_mean = {p_mean:6.3f} GPa  "
          f"(xx,yy,zz = {stress[0]:6.3f}, {stress[1]:6.3f}, {stress[2]:6.3f})  "
          f"t = {elapsed:6.2f} s")

dyn.attach(print_status, interval=500)

# Attach ASE MD logger for energy, temperature, stress logging
dyn.attach(MDLogger(dyn, stable_atoms, logfile, header=True, stress=True,
                    peratom=False, mode="a"), interval=100)

# 6. Run NPT MD simulation
print("Running NPT MD at 300 K and 1 bar ...")
dyn.run(nsteps)
print("Done. Trajectory saved to:", trajfile)
