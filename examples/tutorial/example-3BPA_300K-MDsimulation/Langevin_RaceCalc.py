import json
from ase import Atoms
from ase import units
from ase.io import read, Trajectory
from ase.md import MDLogger
from ase.md.langevin import Langevin

from bam_torch.tase.base_calculator import RACECalculator


atoms = read("../dataset/test_300K.xyz", index=slice(None))[0]  

# Set calculator
model = 'model.pkl'
atoms.calc = RACECalculator(model=model, device='cuda')
"""
Or, 
with open('input.json') as f:
    json_data = json.load(f)
atoms.calc = RACECalculator(json_data)  
Or, 
atoms.calc = RACECalculator(json_data, device='cuda')  
"""

# Set parameters
T = 300
gamma = 0.01
timestep = 0.5 * units.fs

# Langevin integrator
dyn = Langevin(
    atoms,
    timestep,
    temperature=T * units.kB,
    friction=gamma
)

# Set logger
logger_terminal = MDLogger(dyn, atoms, logfile='-', header=True, stress=False, peratom=False)
dyn.attach(logger_terminal, interval=10)

logger_file = MDLogger(dyn, atoms, logfile='md.log', header=True, stress=False, peratom=False)
dyn.attach(logger_file, interval=10)

# Write trajectory
traj = Trajectory("3BPA_langevin.traj", "w", atoms)
dyn.attach(traj.write, interval=10)

# Run Langevin dynamics
dyn.run(20000)