"""
OpenMM-based Prior Force Field for CG Delta Learning.

Computes prior forces using OpenMM's CustomForce classes.
All forces are defined as energy expressions (strings) and
F = -dV/dr is computed automatically by OpenMM.

Supported potential energy terms:
    Bond (2-body):
        harmonic, morse, fene, quartic
    Angle (3-body):
        harmonic, cosine_harmonic, cosine, urey_bradley, quartic
    Dihedral (4-body):
        cosine, opls, rb, fourier, harmonic
    Improper (4-body):
        harmonic, cvff
    Non-bonded (2-body pair):
        lj, lj_repulsive, wca, buckingham, mie
    Electrostatics (2-body pair):
        coulomb, debye_huckel

Usage:
    from bam_torch.utils.prior_openmm import OpenMMPrior

    prior = OpenMMPrior(
        n_particles=1600,
        box=[60.0, 60.0, 60.0],  # Angstrom
        terms=[
            {
                'type': 'bond',
                'function': 'harmonic',
                'pairs': [[0, 1], [2, 3], ...],
                'params': {'k': 10.0, 'r0': 4.5},
            },
            {
                'type': 'nonbonded',
                'function': 'lj_repulsive',
                'params': {'epsilon': 0.001, 'sigma': 3.5},
                'exclusions': [[0,1], [2,3], ...],
            },
        ]
    )

    F_prior = prior.compute_forces(positions)
    E_prior = prior.compute_energy(positions)
    F_prior, E_prior = prior.compute(positions)

Author: BAM-torch CG Extension
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union

try:
    import openmm
    import openmm.unit as unit
    HAS_OPENMM = True
except ImportError:
    HAS_OPENMM = False


# =============================================================================
# Energy function expressions (string-based, OpenMM evaluates automatically)
# =============================================================================

BOND_FUNCTIONS = {
    'harmonic':  '0.5*k*(r-r0)^2',
    'morse':     'D_e*(1-exp(-a*(r-r0)))^2',
    'fene':      '-0.5*k*R0^2*log(1-(r/R0)^2)',
    'quartic':   'k2*(r-r0)^2 + k3*(r-r0)^3 + k4*(r-r0)^4',
}

ANGLE_FUNCTIONS = {
    'harmonic':         '0.5*k*(theta-theta0)^2',
    'cosine_harmonic':  '0.5*k*(cos(theta)-cos(theta0))^2',
    'cosine':           'k*(1-cos(theta-theta0))',
    'quartic':          'k2*(theta-theta0)^2 + k3*(theta-theta0)^3 + k4*(theta-theta0)^4',
}

# Urey-Bradley is special: angle + 1-3 bond (handled separately)

DIHEDRAL_FUNCTIONS = {
    'cosine':    'k*(1+cos(n*theta-delta))',
    'opls':      'V1/2*(1+cos(theta)) + V2/2*(1-cos(2*theta)) + V3/2*(1+cos(3*theta)) + V4/2*(1-cos(4*theta))',
    'rb':        'C0 + C1*cos(psi) + C2*cos(psi)^2 + C3*cos(psi)^3 + C4*cos(psi)^4 + C5*cos(psi)^5; psi=theta-pi',
    'fourier':   'v0 + a1*cos(theta) + b1*sin(theta) + a2*cos(2*theta) + b2*sin(2*theta) + a3*cos(3*theta) + b3*sin(3*theta)',
    'harmonic':  '0.5*k*(theta-theta0)^2',
}

IMPROPER_FUNCTIONS = {
    'harmonic':  '0.5*k*(theta-theta0)^2',
    'cvff':      'k*(1+d*cos(n*theta))',
}

# Note: CustomNonbondedForce uses param1/param2 for the two particles in a pair.
# Combining rules: eps=sqrt(eps1*eps2), sig=(sig1+sig2)/2 (Lorentz-Berthelot)
NONBONDED_FUNCTIONS = {
    'lj':            '4*eps*((sig/r)^12 - (sig/r)^6); eps=sqrt(eps1*eps2); sig=(sig1+sig2)/2',
    'lj_repulsive':  'eps*(sig/r)^12; eps=sqrt(eps1*eps2); sig=(sig1+sig2)/2',
    'wca':           'step((sig1+sig2)/2*1.122462048309373-r)*(4*sqrt(eps1*eps2)*(((sig1+sig2)/(2*r))^12-((sig1+sig2)/(2*r))^6)+sqrt(eps1*eps2))',
    'buckingham':    'sqrt(A1*A2)*exp(-0.5*(B1+B2)*r) - sqrt(C1*C2)/r^6',
    'mie':           '(n1/(n1-m1))*(n1/m1)^(m1/(n1-m1))*sqrt(eps1*eps2)*(((sig1+sig2)/(2*r))^n1-((sig1+sig2)/(2*r))^m1)',
}

ELECTROSTATIC_FUNCTIONS = {
    'coulomb':       '138.935458*q1*q2/r',  # 138.935458 = 1/(4πε₀) in kJ·nm/(mol·e²); q is per-particle
    'debye_huckel':  '138.935458*q1*q2*exp(-0.5*(kappa1+kappa2)*r)/r',
}

# All function registry
ALL_FUNCTIONS = {
    'bond': BOND_FUNCTIONS,
    'angle': ANGLE_FUNCTIONS,
    'dihedral': DIHEDRAL_FUNCTIONS,
    'improper': IMPROPER_FUNCTIONS,
    'nonbonded': NONBONDED_FUNCTIONS,
    'electrostatic': ELECTROSTATIC_FUNCTIONS,
}


def list_available_functions():
    """Print all available energy functions."""
    print("=" * 60)
    print("Available Prior Energy Functions (OpenMM-based)")
    print("=" * 60)
    for category, funcs in ALL_FUNCTIONS.items():
        print(f"\n  {category.upper()}:")
        for name, expr in funcs.items():
            print(f"    {name:20s}  V = {expr}")
    print()


# =============================================================================
# Parameter name extraction
# =============================================================================

# Map function name → list of parameter names
BOND_PARAMS = {
    'harmonic':  ['k', 'r0'],
    'morse':     ['D_e', 'a', 'r0'],
    'fene':      ['k', 'R0'],
    'quartic':   ['k2', 'k3', 'k4', 'r0'],
}

ANGLE_PARAMS = {
    'harmonic':         ['k', 'theta0'],
    'cosine_harmonic':  ['k', 'theta0'],
    'cosine':           ['k', 'theta0'],
    'quartic':          ['k2', 'k3', 'k4', 'theta0'],
}

DIHEDRAL_PARAMS = {
    'cosine':   ['k', 'n', 'delta'],
    'opls':     ['V1', 'V2', 'V3', 'V4'],
    'rb':       ['C0', 'C1', 'C2', 'C3', 'C4', 'C5'],
    'fourier':  ['v0', 'a1', 'b1', 'a2', 'b2', 'a3', 'b3'],
    'harmonic': ['k', 'theta0'],
}

IMPROPER_PARAMS = {
    'harmonic': ['k', 'theta0'],
    'cvff':     ['k', 'd', 'n'],
}

NONBONDED_PARAMS = {
    'lj':            ['eps', 'sig'],
    'lj_repulsive':  ['eps', 'sig'],
    'wca':           ['eps', 'sig'],
    'buckingham':    ['A', 'B', 'C'],
    'mie':           ['eps', 'sig', 'n', 'm'],
}

ELECTROSTATIC_PARAMS = {
    'coulomb':       ['q'],
    'debye_huckel':  ['q', 'kappa'],
}

ALL_PARAMS = {
    'bond': BOND_PARAMS,
    'angle': ANGLE_PARAMS,
    'dihedral': DIHEDRAL_PARAMS,
    'improper': IMPROPER_PARAMS,
    'nonbonded': NONBONDED_PARAMS,
    'electrostatic': ELECTROSTATIC_PARAMS,
}


# =============================================================================
# Unit conversion constants (our code uses eV/Å, OpenMM uses kJ/mol/nm)
# =============================================================================

# 1 eV = 96.4853 kJ/mol
EV_TO_KJ_MOL = 96.4853
KJ_MOL_TO_EV = 1.0 / EV_TO_KJ_MOL

# 1 Å = 0.1 nm
ANGSTROM_TO_NM = 0.1
NM_TO_ANGSTROM = 10.0


# =============================================================================
# OpenMM Prior Force Field
# =============================================================================

class OpenMMPrior:
    """
    OpenMM-based prior force field for CG delta learning.

    Combines multiple energy terms (bond, angle, dihedral, non-bonded, etc.)
    and computes total prior force and energy using OpenMM.

    All positions/forces are in eV/Å convention (auto-converted to/from OpenMM units).

    Args:
        n_particles: Number of CG beads
        box: Box dimensions [Lx, Ly, Lz] in Angstrom, or None for non-periodic
        terms: List of term configurations (see module docstring)
        masses: Per-particle masses in amu (default: 72.0 for all)
    """

    def __init__(
        self,
        n_particles: int,
        box: Optional[List[float]] = None,
        terms: Optional[List[Dict]] = None,
        masses: Optional[List[float]] = None,
    ):
        if not HAS_OPENMM:
            raise ImportError(
                "OpenMM is required. Install: pip install openmm"
            )

        self.n_particles = n_particles
        self.box = box
        self.terms = terms or []

        # Build OpenMM system
        self.system = openmm.System()

        # Add particles
        default_mass = 72.0  # amu
        for i in range(n_particles):
            m = masses[i] if masses is not None else default_mass
            self.system.addParticle(m)

        # Set periodic box
        if box is not None:
            lx, ly, lz = [b * ANGSTROM_TO_NM for b in box]
            self.system.setDefaultPeriodicBoxVectors(
                openmm.Vec3(lx, 0, 0),
                openmm.Vec3(0, ly, 0),
                openmm.Vec3(0, 0, lz),
            )

        # Add force terms
        for term_config in self.terms:
            self._add_term(term_config)

        # Create context
        integrator = openmm.VerletIntegrator(0.001 * unit.picoseconds)
        self.context = openmm.Context(self.system, integrator)

    def _add_term(self, term_config: Dict):
        """Add a single energy term to the system."""
        term_type = term_config['type']  # bond, angle, dihedral, nonbonded, ...
        func_name = term_config['function']  # harmonic, morse, lj, ...

        if term_type == 'bond':
            self._add_bond(term_config, func_name)
        elif term_type == 'angle':
            self._add_angle(term_config, func_name)
        elif term_type == 'urey_bradley':
            self._add_urey_bradley(term_config)
        elif term_type in ('dihedral', 'improper'):
            self._add_torsion(term_config, func_name, term_type)
        elif term_type == 'nonbonded':
            self._add_nonbonded(term_config, func_name)
        elif term_type == 'electrostatic':
            self._add_nonbonded(term_config, func_name)
        else:
            raise ValueError(f"Unknown term type: {term_type}")

    def _add_bond(self, config: Dict, func_name: str):
        """Add bond force."""
        expr = BOND_FUNCTIONS[func_name]
        param_names = BOND_PARAMS[func_name]

        force = openmm.CustomBondForce(expr)
        # Enable PBC for bond distance calculation
        if self.box is not None:
            force.setUsesPeriodicBoundaryConditions(True)
        for pname in param_names:
            force.addPerBondParameter(pname)

        pairs = config.get('pairs', [])
        params = config.get('params', {})
        per_bond_params = config.get('per_bond_params', None)

        for idx, (i, j) in enumerate(pairs):
            if per_bond_params is not None:
                p_vals = [per_bond_params[idx].get(pn, 0.0) for pn in param_names]
            else:
                p_vals = [params.get(pn, 0.0) for pn in param_names]

            # Unit conversion: r0 Å→nm, k eV/Å²→kJ/mol/nm²
            p_converted = self._convert_bond_params(func_name, param_names, p_vals)
            force.addBond(i, j, p_converted)

        self.system.addForce(force)

    def _add_angle(self, config: Dict, func_name: str):
        """Add angle force."""
        expr = ANGLE_FUNCTIONS[func_name]
        param_names = ANGLE_PARAMS[func_name]

        force = openmm.CustomAngleForce(expr)
        for pname in param_names:
            force.addPerAngleParameter(pname)

        triplets = config.get('triplets', [])
        params = config.get('params', {})
        per_angle_params = config.get('per_angle_params', None)

        for idx, (i, j, k) in enumerate(triplets):
            if per_angle_params is not None:
                p_vals = [per_angle_params[idx].get(pn, 0.0) for pn in param_names]
            else:
                p_vals = [params.get(pn, 0.0) for pn in param_names]

            p_converted = self._convert_angle_params(func_name, param_names, p_vals)
            force.addAngle(i, j, k, p_converted)

        self.system.addForce(force)

    def _add_urey_bradley(self, config: Dict):
        """Add Urey-Bradley: harmonic angle + 1-3 distance term."""
        # Angle part
        angle_force = openmm.CustomAngleForce('0.5*k*(theta-theta0)^2')
        angle_force.addPerAngleParameter('k')
        angle_force.addPerAngleParameter('theta0')

        # 1-3 bond part
        ub_force = openmm.CustomBondForce('0.5*k_ub*(r-r0_ub)^2')
        if self.box is not None:
            ub_force.setUsesPeriodicBoundaryConditions(True)
        ub_force.addPerBondParameter('k_ub')
        ub_force.addPerBondParameter('r0_ub')

        triplets = config.get('triplets', [])
        params = config.get('params', {})

        for i, j, k in triplets:
            k_angle = params.get('k', 0.0) * EV_TO_KJ_MOL  # eV/rad² → kJ/mol/rad²
            theta0 = params.get('theta0', 2.094)  # rad (no conversion)
            angle_force.addAngle(i, j, k, [k_angle, theta0])

            k_ub = params.get('k_ub', 0.0) * EV_TO_KJ_MOL / (NM_TO_ANGSTROM**2)
            r0_ub = params.get('r0_ub', 0.0) * ANGSTROM_TO_NM
            ub_force.addBond(i, k, [k_ub, r0_ub])

        self.system.addForce(angle_force)
        self.system.addForce(ub_force)

    def _add_torsion(self, config: Dict, func_name: str, term_type: str):
        """Add dihedral/improper torsion force."""
        func_dict = DIHEDRAL_FUNCTIONS if term_type == 'dihedral' else IMPROPER_FUNCTIONS
        param_dict = DIHEDRAL_PARAMS if term_type == 'dihedral' else IMPROPER_PARAMS

        expr = func_dict[func_name]
        param_names = param_dict[func_name]

        force = openmm.CustomTorsionForce(expr)
        for pname in param_names:
            force.addPerTorsionParameter(pname)

        quadruplets = config.get('quadruplets', [])
        params = config.get('params', {})
        per_torsion_params = config.get('per_torsion_params', None)

        for idx, (i, j, k, l) in enumerate(quadruplets):
            if per_torsion_params is not None:
                p_vals = [per_torsion_params[idx].get(pn, 0.0) for pn in param_names]
            else:
                p_vals = [params.get(pn, 0.0) for pn in param_names]

            p_converted = self._convert_torsion_params(func_name, param_names, p_vals)
            force.addTorsion(i, j, k, l, p_converted)

        self.system.addForce(force)

    def _add_nonbonded(self, config: Dict, func_name: str):
        """Add non-bonded or electrostatic force."""
        if func_name in NONBONDED_FUNCTIONS:
            expr = NONBONDED_FUNCTIONS[func_name]
            param_names = NONBONDED_PARAMS[func_name]
        elif func_name in ELECTROSTATIC_FUNCTIONS:
            expr = ELECTROSTATIC_FUNCTIONS[func_name]
            param_names = ELECTROSTATIC_PARAMS[func_name]
        else:
            raise ValueError(f"Unknown nonbonded function: {func_name}")

        force = openmm.CustomNonbondedForce(expr)
        for pname in param_names:
            force.addPerParticleParameter(pname)

        cutoff_nm = config.get('cutoff', 10.0) * ANGSTROM_TO_NM
        if self.box is not None:
            force.setNonbondedMethod(openmm.CustomNonbondedForce.CutoffPeriodic)
        else:
            force.setNonbondedMethod(openmm.CustomNonbondedForce.CutoffNonPeriodic)
        force.setCutoffDistance(cutoff_nm)

        # Per-particle parameters
        params = config.get('params', {})
        per_particle_params = config.get('per_particle_params', None)

        for i in range(self.n_particles):
            if per_particle_params is not None:
                p_vals = [per_particle_params[i].get(pn, 0.0) for pn in param_names]
            else:
                p_vals = [params.get(pn, 0.0) for pn in param_names]
            p_converted = self._convert_nonbonded_params(func_name, param_names, p_vals)
            force.addParticle(p_converted)

        # Add exclusions
        exclusions = config.get('exclusions', [])
        for i, j in exclusions:
            force.addExclusion(i, j)

        self.system.addForce(force)

    # ─── Unit conversion helpers ───

    def _convert_bond_params(self, func_name, param_names, values):
        """Convert bond parameters from eV/Å to kJ/mol/nm."""
        converted = []
        for pname, val in zip(param_names, values):
            if pname in ('r0', 'R0'):
                converted.append(val * ANGSTROM_TO_NM)  # Å → nm
            elif pname == 'k' and func_name == 'harmonic':
                converted.append(val * EV_TO_KJ_MOL / (ANGSTROM_TO_NM**2))  # eV/Å² → kJ/mol/nm²
            elif pname == 'k' and func_name == 'fene':
                converted.append(val * EV_TO_KJ_MOL / (ANGSTROM_TO_NM**2))
            elif pname == 'D_e':
                converted.append(val * EV_TO_KJ_MOL)  # eV → kJ/mol
            elif pname == 'a':
                converted.append(val / ANGSTROM_TO_NM)  # 1/Å → 1/nm
            elif pname in ('k2', 'k3', 'k4'):
                # Quartic: k2 eV/Å², k3 eV/Å³, k4 eV/Å⁴
                n = int(pname[1])
                converted.append(val * EV_TO_KJ_MOL / (ANGSTROM_TO_NM**n))
            else:
                converted.append(val)
        return converted

    def _convert_angle_params(self, func_name, param_names, values):
        """Convert angle parameters from eV/rad to kJ/mol/rad."""
        converted = []
        for pname, val in zip(param_names, values):
            if pname == 'theta0':
                converted.append(val)  # rad → rad (no conversion)
            elif pname == 'k':
                converted.append(val * EV_TO_KJ_MOL)  # eV/rad² → kJ/mol/rad²
            elif pname in ('k2', 'k3', 'k4'):
                converted.append(val * EV_TO_KJ_MOL)
            else:
                converted.append(val)
        return converted

    def _convert_torsion_params(self, func_name, param_names, values):
        """Convert torsion parameters from eV to kJ/mol."""
        converted = []
        for pname, val in zip(param_names, values):
            if pname in ('n', 'd'):
                converted.append(val)  # dimensionless integer
            elif pname in ('delta', 'theta0'):
                converted.append(val)  # rad
            else:
                # All energy-like params: eV → kJ/mol
                converted.append(val * EV_TO_KJ_MOL)
        return converted

    def _convert_nonbonded_params(self, func_name, param_names, values):
        """Convert non-bonded parameters from eV/Å to kJ/mol/nm."""
        converted = []
        for pname, val in zip(param_names, values):
            if pname == 'sig':
                converted.append(val * ANGSTROM_TO_NM)  # Å → nm
            elif pname == 'eps':
                converted.append(val * EV_TO_KJ_MOL)  # eV → kJ/mol
            elif pname in ('A', 'C'):
                # Buckingham: A in eV, C in eV·Å⁶
                converted.append(val * EV_TO_KJ_MOL)  # simplified
            elif pname == 'B':
                converted.append(val / ANGSTROM_TO_NM)  # 1/Å → 1/nm
            elif pname == 'kappa':
                converted.append(val / ANGSTROM_TO_NM)  # 1/Å → 1/nm
            elif pname in ('q1', 'q2'):
                converted.append(val)  # elementary charge (dimensionless in our convention)
            elif pname in ('n', 'm'):
                converted.append(val)  # dimensionless
            else:
                converted.append(val)
        return converted

    # ─── Force/Energy computation ───

    def compute(
        self,
        positions: np.ndarray,
        box: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, float]:
        """
        Compute prior forces and energy.

        Args:
            positions: (n_particles, 3) in Angstrom
            box: (3,3) or (3,) cell matrix in Angstrom (overrides init box)

        Returns:
            forces: (n_particles, 3) in eV/Å
            energy: float in eV
        """
        # Convert positions Å → nm
        pos_nm = positions * ANGSTROM_TO_NM
        self.context.setPositions(pos_nm)

        # Update box if provided
        if box is not None:
            if box.ndim == 1:
                box_nm = box * ANGSTROM_TO_NM
                self.context.setPeriodicBoxVectors(
                    openmm.Vec3(box_nm[0], 0, 0),
                    openmm.Vec3(0, box_nm[1], 0),
                    openmm.Vec3(0, 0, box_nm[2]),
                )
            elif box.ndim == 2:
                box_nm = box * ANGSTROM_TO_NM
                self.context.setPeriodicBoxVectors(
                    openmm.Vec3(*box_nm[0]),
                    openmm.Vec3(*box_nm[1]),
                    openmm.Vec3(*box_nm[2]),
                )

        state = self.context.getState(getForces=True, getEnergy=True)

        # Convert forces kJ/mol/nm → eV/Å
        # 1 kJ/(mol·nm) = (1/96.4853 eV) / (10 Å) = 0.001036 eV/Å
        forces_omm = state.getForces(asNumpy=True).value_in_unit(
            unit.kilojoules_per_mole / unit.nanometer
        )
        forces = np.array(forces_omm) * KJ_MOL_TO_EV * ANGSTROM_TO_NM  # kJ/mol/nm → eV/Å

        # Convert energy kJ/mol → eV
        energy = state.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole) * KJ_MOL_TO_EV

        return forces, energy

    def compute_forces(
        self,
        positions: np.ndarray,
        types: Optional[np.ndarray] = None,
        cell: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Compute prior forces. Compatible with PriorForceField interface."""
        box = np.diag(cell) if cell is not None and cell.ndim == 2 else cell
        forces, _ = self.compute(positions, box)
        return forces.astype(positions.dtype)

    def compute_energy(
        self,
        positions: np.ndarray,
        types: Optional[np.ndarray] = None,
        cell: Optional[np.ndarray] = None,
    ) -> float:
        """Compute prior energy. Compatible with PriorForceField interface."""
        box = np.diag(cell) if cell is not None and cell.ndim == 2 else cell
        _, energy = self.compute(positions, box)
        return energy

    def summary(self):
        """Print summary of the prior force field."""
        print(f"OpenMMPrior: {self.n_particles} particles, {len(self.terms)} terms")
        for i, term in enumerate(self.terms):
            print(f"  [{i}] {term['type']}/{term['function']}")


def create_cg_prior(
    n_particles: int,
    box: Optional[List[float]],
    bond_topology: Dict,
    prior_config: Dict,
) -> OpenMMPrior:
    """
    Convenience function to create a CG prior from topology + parameter config.

    Args:
        n_particles: Number of CG beads
        box: [Lx, Ly, Lz] in Angstrom
        bond_topology: {'n_beads_per_mol': int, 'bonds': [[i,j], ...]}
        prior_config: {
            'bond': {'function': 'harmonic', 'k': ..., 'r0': ...},
            'angle': {'function': 'cosine_harmonic', 'k': ..., 'theta0': ...},
            'nonbonded': {'function': 'lj_repulsive', 'epsilon': ..., 'sigma': ...},
        }

    Returns:
        OpenMMPrior instance
    """
    n_beads = bond_topology['n_beads_per_mol']
    bonds_local = bond_topology['bonds']
    n_mol = n_particles // n_beads

    terms = []

    # ─── Bonds ───
    if 'bond' in prior_config:
        bc = prior_config['bond']
        all_pairs = []
        for m in range(n_mol):
            offset = m * n_beads
            for i, j in bonds_local:
                all_pairs.append([offset + i, offset + j])
        terms.append({
            'type': 'bond',
            'function': bc.get('function', 'harmonic'),
            'pairs': all_pairs,
            'params': {k: v for k, v in bc.items() if k != 'function'},
        })

    # ─── Angles ───
    if 'angle' in prior_config:
        ac = prior_config['angle']
        # Auto-generate angles from bonds
        from bam_torch.utils.prior_ff import generate_angles_from_bonds
        angles_local = ac.get('angles', generate_angles_from_bonds(bonds_local, n_beads))
        all_triplets = []
        for m in range(n_mol):
            offset = m * n_beads
            for i, j, k in angles_local:
                all_triplets.append([offset + i, offset + j, offset + k])
        terms.append({
            'type': 'angle',
            'function': ac.get('function', 'harmonic'),
            'triplets': all_triplets,
            'params': {k: v for k, v in ac.items() if k not in ('function', 'angles')},
        })

    # ─── Non-bonded ───
    if 'nonbonded' in prior_config:
        nc = prior_config['nonbonded']
        # Build exclusion list (all intra-molecular pairs)
        exclusions = []
        for m in range(n_mol):
            offset = m * n_beads
            for i in range(n_beads):
                for j in range(i + 1, n_beads):
                    exclusions.append([offset + i, offset + j])
        terms.append({
            'type': 'nonbonded',
            'function': nc.get('function', 'lj_repulsive'),
            'params': {k: v for k, v in nc.items() if k not in ('function', 'cutoff')},
            'cutoff': nc.get('cutoff', 10.0),
            'exclusions': exclusions,
        })

    return OpenMMPrior(
        n_particles=n_particles,
        box=box,
        terms=terms,
    )
