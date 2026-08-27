"""
Prior Force Field Module for Delta Learning in CG Systems

This module provides universal atomic number-based force fields that can be used
as priors for delta learning. The ML model learns to predict F_delta = F_total - F_prior.

Supported Prior Types (All Universal - Atomic Number Based):
- ZBL: Ziegler-Biersack-Littmark screened nuclear repulsion (short-range)
- D2: DFT-D2 Grimme dispersion correction (long-range)
- Universal: Combined ZBL + D2

These priors require only atomic numbers, making them applicable to any molecular system.

Author: BAM-torch CG Extension
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union


# =============================================================================
# Base Prior Force Field Class
# =============================================================================

class PriorForceField:
    """
    Base class for prior force fields in delta learning.

    Usage:
        # Universal prior (any system, just need atomic numbers)
        prior = PriorForceField.from_config({
            'type': 'universal',
            'atomic_numbers': [6, 1, 1, 1, 1],  # CH4
            'zbl_cutoff': 5.0,
            'd2_cutoff': 20.0
        })
        F_prior = prior.compute_forces(positions)
        F_delta = F_total - F_prior
    """

    def __init__(self, config: Dict):
        """
        Initialize prior force field.

        Args:
            config: Configuration dictionary with parameters
        """
        self.config = config
        self.prior_type = config.get('type', 'none')
        self.cutoff = config.get('cutoff', 10.0)

    @classmethod
    def from_config(cls, config: Dict) -> 'PriorForceField':
        """Create prior FF from a configuration dictionary."""
        prior_type = config.get('type', 'none')

        if prior_type == 'zbl':
            return ZBLPrior(config)
        elif prior_type == 'd2':
            return D2Prior(config)
        elif prior_type == 'universal':
            return UniversalPrior(config)
        elif prior_type == 'harmonic_repulsive':
            return HarmonicRepulsivePrior(config)
        elif prior_type == 'harmonic_bond':
            return HarmonicBondPrior(config)
        elif prior_type == 'harmonic_angle':
            return HarmonicAnglePrior(config)
        elif prior_type == 'repulsive_lj':
            return RepulsiveLJPrior(config)
        elif prior_type == 'morse_bond':
            return MorseBondPrior(config)
        elif prior_type == 'fene_bond':
            return FENEBondPrior(config)
        elif prior_type == 'cosine_harmonic_angle':
            return CosineHarmonicAnglePrior(config)
        elif prior_type == 'dihedral':
            return ProperDihedralPrior(config)
        elif prior_type == 'lj':
            return LJPrior(config)
        elif prior_type == 'wca':
            return WCAPrior(config)
        elif prior_type == 'none':
            return NoPrior(config)
        else:
            raise ValueError(f"Unknown prior type: {prior_type}. "
                           f"Available: zbl, d2, universal, harmonic_repulsive, "
                           f"harmonic_bond, harmonic_angle, repulsive_lj, "
                           f"morse_bond, fene_bond, cosine_harmonic_angle, "
                           f"dihedral, lj, wca, none")

    def compute_forces(
        self,
        positions: np.ndarray,
        types: Optional[np.ndarray] = None,
        cell: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Compute prior forces for given positions.

        Args:
            positions: Atomic positions, shape (n_atoms, 3)
            types: Atom/bead types, shape (n_atoms,) - not used for universal priors
            cell: Unit cell for PBC, shape (3, 3) or None

        Returns:
            Prior forces, shape (n_atoms, 3)
        """
        raise NotImplementedError("Subclasses must implement compute_forces")

    def compute_energy(
        self,
        positions: np.ndarray,
        types: Optional[np.ndarray] = None,
        cell: Optional[np.ndarray] = None
    ) -> float:
        """
        Compute prior energy for given positions.

        Args:
            positions: Atomic positions, shape (n_atoms, 3)
            types: Atom/bead types, shape (n_atoms,) - not used for universal priors
            cell: Unit cell for PBC, shape (3, 3) or None

        Returns:
            Prior energy (scalar)
        """
        raise NotImplementedError("Subclasses must implement compute_energy")

    def _apply_pbc(
        self,
        r_vec: np.ndarray,
        cell: Optional[np.ndarray]
    ) -> np.ndarray:
        """Apply minimum image convention for PBC."""
        if cell is None:
            return r_vec

        # Simple orthorhombic PBC
        if cell.ndim == 1 or (cell.ndim == 2 and np.allclose(cell - np.diag(np.diag(cell)), 0)):
            box = np.diag(cell) if cell.ndim == 2 else cell
            r_vec = r_vec - box * np.round(r_vec / box)
        else:
            # General triclinic cell
            inv_cell = np.linalg.inv(cell)
            s = r_vec @ inv_cell
            s = s - np.round(s)
            r_vec = s @ cell

        return r_vec


class NoPrior(PriorForceField):
    """No prior - returns zero forces (equivalent to learning F_total directly)."""

    def compute_forces(self, positions, types=None, cell=None):
        return np.zeros_like(positions)

    def compute_energy(self, positions, types=None, cell=None):
        return 0.0


# =============================================================================
# Universal Atomic Number-Based Priors (like TorchMD-Net)
# =============================================================================

class ZBLPrior(PriorForceField):
    """
    Ziegler-Biersack-Littmark (ZBL) screened nuclear repulsion potential.

    This is a universal prior based on atomic numbers, suitable for any system.
    Particularly useful for short-range repulsion at small interatomic distances.

    Reference:
        Ziegler, J.F., Biersack, J.P., Littmark, U. "The Stopping and Range of Ions in Solids."
        (1985), equations 9 and 10 on page 147.

    V(r) = k_e * Z_i * Z_j / r * phi(r/a)
    where:
        k_e = 14.3996 eV*Å (Coulomb constant)
        a = 0.8854 * a_0 / (Z_i^0.23 + Z_j^0.23)
        a_0 = 0.529177 Å (Bohr radius)
        phi(x) = 0.1818*exp(-3.2*x) + 0.5099*exp(-0.9423*x) +
                 0.2802*exp(-0.4029*x) + 0.02817*exp(-0.2016*x)

    Usage:
        prior = ZBLPrior({
            'atomic_numbers': [6, 6, 8, 1, 1],  # C, C, O, H, H for ethanol
            'cutoff': 5.0
        })
        forces = prior.compute_forces(positions)
    """

    # Bohr radius in Angstrom
    A_0 = 0.529177210903

    # Coulomb constant: k_e = e^2 / (4*pi*eps_0) in eV*Å
    K_E = 14.3996

    def __init__(self, config: Dict):
        super().__init__(config)
        if 'atomic_numbers' not in config or len(config['atomic_numbers']) == 0:
            raise ValueError(
                "ZBLPrior requires 'atomic_numbers' in config. "
                "Please provide atomic numbers for each CG bead type."
            )
        self.atomic_numbers = np.array(config['atomic_numbers'])
        self.cutoff = config.get('cutoff', 5.0)  # ZBL is short-range

    def _screening_function(self, x: np.ndarray) -> np.ndarray:
        """ZBL universal screening function phi(x)."""
        return (0.1818 * np.exp(-3.2 * x) +
                0.5099 * np.exp(-0.9423 * x) +
                0.2802 * np.exp(-0.4029 * x) +
                0.02817 * np.exp(-0.2016 * x))

    def _screening_derivative(self, x: np.ndarray) -> np.ndarray:
        """Derivative of ZBL screening function d(phi)/dx."""
        return (-3.2 * 0.1818 * np.exp(-3.2 * x) +
                -0.9423 * 0.5099 * np.exp(-0.9423 * x) +
                -0.4029 * 0.2802 * np.exp(-0.4029 * x) +
                -0.2016 * 0.02817 * np.exp(-0.2016 * x))

    def _cosine_cutoff(self, r: np.ndarray) -> np.ndarray:
        """Smooth cosine cutoff function."""
        return np.where(
            r < self.cutoff,
            0.5 * (1 + np.cos(np.pi * r / self.cutoff)),
            0.0
        )

    def _cosine_cutoff_derivative(self, r: np.ndarray) -> np.ndarray:
        """Derivative of cosine cutoff."""
        return np.where(
            r < self.cutoff,
            -0.5 * np.pi / self.cutoff * np.sin(np.pi * r / self.cutoff),
            0.0
        )

    def compute_forces(self, positions, types=None, cell=None):
        """
        Compute ZBL forces.

        Args:
            positions: shape (n_atoms, 3)
            types: not used (atomic_numbers are used directly)
            cell: unit cell for PBC
        """
        n_atoms = len(positions)
        forces = np.zeros_like(positions)

        # Get atomic numbers for each atom
        if len(self.atomic_numbers) == 0:
            return forces  # No atomic numbers provided

        Z = self.atomic_numbers
        if len(Z) != n_atoms:
            raise ValueError(f"atomic_numbers length ({len(Z)}) != n_atoms ({n_atoms})")

        for i in range(n_atoms):
            for j in range(i + 1, n_atoms):
                r_vec = positions[j] - positions[i]
                r_vec = self._apply_pbc(r_vec, cell)
                r = np.linalg.norm(r_vec)

                if r > self.cutoff or r < 1e-10:
                    continue

                Z_i, Z_j = Z[i], Z[j]

                # Screening length
                a = 0.8854 * self.A_0 / (Z_i**0.23 + Z_j**0.23)
                x = r / a

                # Screening function and its derivative
                phi = self._screening_function(x)
                dphi_dx = self._screening_derivative(x)

                # Cutoff and its derivative
                fc = self._cosine_cutoff(r)
                dfc_dr = self._cosine_cutoff_derivative(r)

                # Energy: V = k_e * Z_i * Z_j / r * phi(r/a) * fc(r)
                # Force: F = -dV/dr * r_hat

                prefactor = self.K_E * Z_i * Z_j

                # dV/dr = k_e * Z_i * Z_j * [
                #     -phi/r^2 * fc + (1/r) * (dphi/dx) * (1/a) * fc + (phi/r) * dfc/dr
                # ]
                dV_dr = prefactor * (
                    -phi / (r * r) * fc +
                    (1.0 / r) * (dphi_dx / a) * fc +
                    (phi / r) * dfc_dr
                )

                f = -dV_dr * (r_vec / r)
                forces[i] -= f
                forces[j] += f

        return forces

    def compute_energy(self, positions, types=None, cell=None):
        """Compute ZBL energy."""
        n_atoms = len(positions)
        energy = 0.0

        if len(self.atomic_numbers) == 0:
            return energy

        Z = self.atomic_numbers
        if len(Z) != n_atoms:
            raise ValueError(f"atomic_numbers length ({len(Z)}) != n_atoms ({n_atoms})")

        for i in range(n_atoms):
            for j in range(i + 1, n_atoms):
                r_vec = positions[j] - positions[i]
                r_vec = self._apply_pbc(r_vec, cell)
                r = np.linalg.norm(r_vec)

                if r > self.cutoff or r < 1e-10:
                    continue

                Z_i, Z_j = Z[i], Z[j]

                a = 0.8854 * self.A_0 / (Z_i**0.23 + Z_j**0.23)
                x = r / a

                phi = self._screening_function(x)
                fc = self._cosine_cutoff(r)

                energy += self.K_E * Z_i * Z_j / r * phi * fc

        return energy


class D2Prior(PriorForceField):
    """
    DFT-D2 dispersion correction as a prior.

    This is a universal prior based on atomic numbers, providing long-range
    dispersion (van der Waals) interactions.

    Reference:
        Grimme, Stefan. "Semiempirical GGA-type density functional constructed
        with a long‐range dispersion correction."
        Journal of computational chemistry 27.15 (2006): 1787-1799.

    V_disp = -s_6 * sum_{i<j} C_6^{ij} / r_ij^6 * f_damp(r_ij)
    where:
        f_damp(r) = 1 / (1 + exp(-d * (r/R_r - 1)))
        C_6^{ij} = sqrt(C_6^i * C_6^j)
        R_r^{ij} = R_r^i + R_r^j

    Usage:
        prior = D2Prior({
            'atomic_numbers': [6, 6, 8, 1, 1],
            'cutoff': 20.0,
            's_6': 1.0,
            'd': 20
        })
        forces = prior.compute_forces(positions)
    """

    # C_6 parameters (J/mol*nm^6) and van der Waals radii (nm) for elements
    # From Table 1 of Grimme 2006
    # Index = atomic number
    C6_R0_TABLE = {
        1:  (0.14, 0.1001),   # H
        2:  (0.08, 0.1012),   # He
        3:  (1.61, 0.0825),   # Li
        4:  (1.61, 0.1408),   # Be
        5:  (3.13, 0.1485),   # B
        6:  (1.75, 0.1452),   # C
        7:  (1.23, 0.1397),   # N
        8:  (0.70, 0.1342),   # O
        9:  (0.75, 0.1287),   # F
        10: (0.63, 0.1243),   # Ne
        11: (5.71, 0.1144),   # Na
        12: (5.71, 0.1364),   # Mg
        13: (10.79, 0.1639),  # Al
        14: (9.23, 0.1716),   # Si
        15: (7.84, 0.1705),   # P
        16: (5.57, 0.1683),   # S
        17: (5.07, 0.1639),   # Cl
        18: (4.61, 0.1595),   # Ar
        19: (10.80, 0.1485),  # K
        20: (10.80, 0.1474),  # Ca
        21: (10.80, 0.1562),  # Sc
        22: (10.80, 0.1562),  # Ti
        23: (10.80, 0.1562),  # V
        24: (10.80, 0.1562),  # Cr
        25: (10.80, 0.1562),  # Mn
        26: (10.80, 0.1562),  # Fe
        27: (10.80, 0.1562),  # Co
        28: (10.80, 0.1562),  # Ni
        29: (10.80, 0.1562),  # Cu
        30: (10.80, 0.1562),  # Zn
        31: (16.99, 0.1650),  # Ga
        32: (17.10, 0.1727),  # Ge
        33: (16.37, 0.1760),  # As
        34: (12.64, 0.1771),  # Se
        35: (12.47, 0.1749),  # Br
        36: (12.01, 0.1727),  # Kr
        37: (24.67, 0.1628),  # Rb
        38: (24.67, 0.1606),  # Sr
        39: (24.67, 0.1639),  # Y
        40: (24.67, 0.1639),  # Zr
        41: (24.67, 0.1639),  # Nb
        42: (24.67, 0.1639),  # Mo
        43: (24.67, 0.1639),  # Tc
        44: (24.67, 0.1639),  # Ru
        45: (24.67, 0.1639),  # Rh
        46: (24.67, 0.1639),  # Pd
        47: (24.67, 0.1639),  # Ag
        48: (24.67, 0.1639),  # Cd
        49: (37.32, 0.1672),  # In
        50: (38.71, 0.1804),  # Sn
        51: (38.44, 0.1881),  # Sb
        52: (31.74, 0.1892),  # Te
        53: (31.50, 0.1892),  # I
        54: (29.99, 0.1881),  # Xe
    }

    def __init__(self, config: Dict):
        super().__init__(config)
        if 'atomic_numbers' not in config or len(config['atomic_numbers']) == 0:
            raise ValueError(
                "D2Prior requires 'atomic_numbers' in config. "
                "Please provide atomic numbers for each CG bead type."
            )
        self.atomic_numbers = np.array(config['atomic_numbers'])
        self.cutoff = config.get('cutoff', 20.0)  # Å, D2 is longer range
        self.s_6 = config.get('s_6', 1.0)  # Global scaling factor
        self.d = config.get('d', 20)  # Damping steepness

        # Unit conversion:
        # C_6 in J/mol*nm^6 -> eV*Å^6
        # 1 J/mol = 1.0364e-5 eV (per molecule)
        # 1 nm^6 = 1e6 Å^6
        # So: J/mol*nm^6 -> eV*Å^6: multiply by 1.0364e-5 * 1e6 = 10.364
        self.c6_conversion = 1.0364e-5 * 1e6

    def _get_c6_r0(self, Z: int) -> Tuple[float, float]:
        """Get C_6 (eV*Å^6) and R_0 (Å) for atomic number Z."""
        if Z not in self.C6_R0_TABLE:
            raise KeyError(
                f"Atomic number {Z} not found in D2Prior.C6_R0_TABLE. "
                f"D2 dispersion parameters are only available for Z=1-54. "
                f"Please add parameters for Z={Z} to C6_R0_TABLE in prior_ff.py"
            )
        c6_orig, r0_nm = self.C6_R0_TABLE[Z]
        c6_eV_A6 = c6_orig * self.c6_conversion
        r0_A = r0_nm * 10  # nm -> Å
        return c6_eV_A6, r0_A

    def _damping_function(self, r: float, R_r: float) -> float:
        """Fermi damping function."""
        x = -self.d * (r / R_r - 1)
        # Prevent overflow
        if x > 50:
            return 1.0
        elif x < -50:
            return 0.0
        return 1.0 / (1.0 + np.exp(x))

    def _damping_derivative(self, r: float, R_r: float) -> float:
        """Derivative of damping function with respect to r."""
        x = -self.d * (r / R_r - 1)
        if abs(x) > 50:
            return 0.0
        exp_x = np.exp(x)
        f = 1.0 / (1.0 + exp_x)
        # df/dr = df/dx * dx/dr = f^2 * exp(x) * d/R_r
        return f * f * exp_x * self.d / R_r

    def compute_forces(self, positions, types=None, cell=None):
        """Compute D2 dispersion forces."""
        n_atoms = len(positions)
        forces = np.zeros_like(positions)

        if len(self.atomic_numbers) == 0:
            return forces

        Z = self.atomic_numbers
        if len(Z) != n_atoms:
            raise ValueError(f"atomic_numbers length ({len(Z)}) != n_atoms ({n_atoms})")

        for i in range(n_atoms):
            for j in range(i + 1, n_atoms):
                r_vec = positions[j] - positions[i]
                r_vec = self._apply_pbc(r_vec, cell)
                r = np.linalg.norm(r_vec)

                if r > self.cutoff or r < 1e-10:
                    continue

                c6_i, r0_i = self._get_c6_r0(Z[i])
                c6_j, r0_j = self._get_c6_r0(Z[j])

                C_6 = np.sqrt(c6_i * c6_j)
                R_r = r0_i + r0_j

                f_damp = self._damping_function(r, R_r)
                df_damp = self._damping_derivative(r, R_r)

                # V = -s_6 * C_6 / r^6 * f_damp
                # dV/dr = -s_6 * C_6 * [-6/r^7 * f_damp + (1/r^6) * df_damp/dr]
                #       = s_6 * C_6 * [6*f_damp/r^7 - df_damp/r^6]

                r6 = r ** 6
                r7 = r * r6

                dV_dr = self.s_6 * C_6 * (6.0 * f_damp / r7 - df_damp / r6)

                f = -dV_dr * (r_vec / r)
                forces[i] -= f
                forces[j] += f

        return forces

    def compute_energy(self, positions, types=None, cell=None):
        """Compute D2 dispersion energy."""
        n_atoms = len(positions)
        energy = 0.0

        if len(self.atomic_numbers) == 0:
            return energy

        Z = self.atomic_numbers
        if len(Z) != n_atoms:
            raise ValueError(f"atomic_numbers length ({len(Z)}) != n_atoms ({n_atoms})")

        for i in range(n_atoms):
            for j in range(i + 1, n_atoms):
                r_vec = positions[j] - positions[i]
                r_vec = self._apply_pbc(r_vec, cell)
                r = np.linalg.norm(r_vec)

                if r > self.cutoff or r < 1e-10:
                    continue

                c6_i, r0_i = self._get_c6_r0(Z[i])
                c6_j, r0_j = self._get_c6_r0(Z[j])

                C_6 = np.sqrt(c6_i * c6_j)
                R_r = r0_i + r0_j

                f_damp = self._damping_function(r, R_r)

                energy += -self.s_6 * C_6 / (r ** 6) * f_damp

        return energy


class UniversalPrior(PriorForceField):
    """
    Universal prior combining ZBL (short-range repulsion) and D2 (long-range dispersion).

    This provides a reasonable baseline for any molecular system based only on
    atomic numbers, similar to TorchMD-Net's approach.

    Usage:
        prior = UniversalPrior({
            'atomic_numbers': [6, 1, 1, 1, 1],  # Methane: C + 4H
            'zbl_cutoff': 5.0,
            'd2_cutoff': 20.0
        })
    """

    def __init__(self, config: Dict):
        super().__init__(config)

        if 'atomic_numbers' not in config or len(config['atomic_numbers']) == 0:
            raise ValueError(
                "UniversalPrior requires 'atomic_numbers' in config. "
                "Please provide atomic numbers for each CG bead type."
            )

        # Create ZBL prior
        zbl_config = {
            'atomic_numbers': config['atomic_numbers'],
            'cutoff': config.get('zbl_cutoff', 5.0)
        }
        self.zbl = ZBLPrior(zbl_config)

        # Create D2 prior
        d2_config = {
            'atomic_numbers': config['atomic_numbers'],
            'cutoff': config.get('d2_cutoff', 20.0),
            's_6': config.get('s_6', 1.0),
            'd': config.get('d', 20)
        }
        self.d2 = D2Prior(d2_config)

        # Weights for combining
        self.zbl_weight = config.get('zbl_weight', 1.0)
        self.d2_weight = config.get('d2_weight', 1.0)

    def compute_forces(self, positions, types=None, cell=None):
        f_zbl = self.zbl.compute_forces(positions, types, cell)
        f_d2 = self.d2.compute_forces(positions, types, cell)
        return self.zbl_weight * f_zbl + self.d2_weight * f_d2

    def compute_energy(self, positions, types=None, cell=None):
        e_zbl = self.zbl.compute_energy(positions, types, cell)
        e_d2 = self.d2.compute_energy(positions, types, cell)
        return self.zbl_weight * e_zbl + self.d2_weight * e_d2


# =============================================================================
# CG-Specific Priors: Harmonic Bond + Repulsive LJ
# Reference: Majewski et al., Nature Comm. (2023)
# =============================================================================

# DLiPC bond topology: 128 lipids, 21 beads/lipid, 20 bonds/molecule
DLIPC_BOND_TOPOLOGY = {
    "n_beads_per_mol": 21,
    "bonds": [
        [0, 1], [1, 2], [2, 3], [3, 4],               # head chain
        [3, 5], [5, 6], [6, 7], [7, 8],               # sn-2 tail from GL1
        [8, 9], [9, 10], [10, 11], [11, 12],
        [4, 13], [13, 14], [14, 15], [15, 16],         # sn-1 tail from GL2
        [16, 17], [17, 18], [18, 19], [19, 20],
    ]
}


def generate_angles_from_bonds(
    bonds: List[List[int]],
    n_beads_per_mol: int,
) -> List[List[int]]:
    """
    Generate angle triplets (i, j, k) from bond topology, where j is the central atom.

    For each vertex j with neighbors {a, b, c, ...}, generates all angle
    triplets (a, j, b), (a, j, c), (b, j, c), etc.

    Args:
        bonds: List of [i, j] bond pairs (local molecule indices)
        n_beads_per_mol: Number of beads per molecule

    Returns:
        List of [i, j, k] angle triplets where j is the central atom
    """
    # Build adjacency list
    adj: Dict[int, List[int]] = {i: [] for i in range(n_beads_per_mol)}
    for i, j in bonds:
        adj[i].append(j)
        adj[j].append(i)

    # Sort neighbors for reproducibility
    for k in adj:
        adj[k].sort()

    angles = []
    for j in range(n_beads_per_mol):
        neighbors = adj[j]
        for a_idx in range(len(neighbors)):
            for b_idx in range(a_idx + 1, len(neighbors)):
                i, k = neighbors[a_idx], neighbors[b_idx]
                angles.append([i, j, k])

    return angles


def build_exclusion_list(
    bonds_local: List[List[int]],
    n_beads_per_mol: int,
    exclude_13: bool = True,
    exclude_14: bool = True,
) -> set:
    """
    Build set of excluded local pair indices (i, j) with i < j.

    Includes 1-2 (bonded), optionally 1-3 and 1-4 interactions.
    For CG systems, at least 1-3 exclusion is essential because CG beads
    sharing a common bonded neighbor have short center-of-mass distances.

    Args:
        bonds_local: List of [i, j] bond pairs (local molecule indices)
        n_beads_per_mol: Number of beads per molecule
        exclude_13: Exclude 1-3 interactions (default True)
        exclude_14: Exclude 1-4 interactions (default True)

    Returns:
        Set of (i, j) pairs with i < j
    """
    # Build adjacency list
    adj: Dict[int, set] = {i: set() for i in range(n_beads_per_mol)}
    for i, j in bonds_local:
        adj[i].add(j)
        adj[j].add(i)

    excluded = set()

    # 1-2: direct bonds
    for i, j in bonds_local:
        excluded.add((min(i, j), max(i, j)))

    # 1-3: atoms separated by 2 bonds
    if exclude_13:
        for mid in range(n_beads_per_mol):
            neighbors = list(adj[mid])
            for a_idx in range(len(neighbors)):
                for b_idx in range(a_idx + 1, len(neighbors)):
                    a, b = neighbors[a_idx], neighbors[b_idx]
                    excluded.add((min(a, b), max(a, b)))

    # 1-4: atoms separated by 3 bonds
    if exclude_14:
        for i in range(n_beads_per_mol):
            for j in adj[i]:
                for k in adj[j]:
                    if k == i:
                        continue
                    for l in adj[k]:
                        if l == j or l == i:
                            continue
                        excluded.add((min(i, l), max(i, l)))

    return excluded




def build_intra_class_pairs(
    bonds_local: List,
    n_beads_per_mol: int,
    max_graph_distance: int = 4,
) -> Dict:
    """
    Classify intra-molecular bead pairs by bond-graph distance (BFS).

    Graph distance 2 = 1-3 pairs, 3 = 1-4 pairs, 4 = 1-5 pairs, ...
    Used to build class-specific intra-molecular repulsive priors
    (pairs beyond max_graph_distance are left to inter-molecular treatment).

    Args:
        bonds_local: List of [i, j] bond pairs (local molecule indices)
        n_beads_per_mol: Number of beads per molecule
        max_graph_distance: largest graph distance to classify (default 4 = 1-5)

    Returns:
        {graph_distance: [(i, j), ...]} with i < j, sorted
    """
    adj = {i: set() for i in range(n_beads_per_mol)}
    for i, j in bonds_local:
        adj[int(i)].add(int(j))
        adj[int(j)].add(int(i))

    classes: Dict = {}
    for start in range(n_beads_per_mol):
        dist = {start: 0}
        queue = [start]
        while queue:
            cur = queue.pop(0)
            for nb in adj[cur]:
                if nb not in dist:
                    dist[nb] = dist[cur] + 1
                    queue.append(nb)
        for j, d in dist.items():
            if j > start and 2 <= d <= max_graph_distance:
                classes.setdefault(d, []).append((start, j))
    for d in classes:
        classes[d].sort()
    return classes


class HarmonicBondPrior(PriorForceField):
    """
    Harmonic bond prior for CG systems.

    V(r) = 0.5 * k * (r - r_eq)^2 for each bonded pair.

    Config keys:
        bond_topology: dict with 'n_beads_per_mol' and 'bonds' (list of [i,j] pairs)
        k_per_bond: dict mapping 'ti-tj' → k (eV/Å²), or list matching bonds order
        r_eq_per_bond: dict mapping 'ti-tj' → r_eq (Å), or list matching bonds order
        n_molecules: number of molecules (auto-detected if not given)
    """

    def __init__(self, config: Dict):
        super().__init__(config)
        topo = config.get('bond_topology', DLIPC_BOND_TOPOLOGY)
        self.n_beads_per_mol = topo['n_beads_per_mol']
        self.bonds_local = topo['bonds']  # local indices within molecule

        # k and r_eq per bond (in order of bonds_local)
        self.k_per_bond = self._parse_per_bond(config.get('k_per_bond', []))
        self.r_eq_per_bond = self._parse_per_bond(config.get('r_eq_per_bond', []))

        self.n_molecules = config.get('n_molecules', None)

    def _parse_per_bond(self, values) -> np.ndarray:
        """Parse per-bond parameter: list or dict → array of length n_bonds."""
        if isinstance(values, (list, np.ndarray)):
            return np.array(values, dtype=np.float64)
        elif isinstance(values, dict):
            # Dict keyed by 'ti-tj' string
            arr = np.zeros(len(self.bonds_local), dtype=np.float64)
            for idx, (i, j) in enumerate(self.bonds_local):
                key = f"{i}-{j}"
                key_rev = f"{j}-{i}"
                if key in values:
                    arr[idx] = values[key]
                elif key_rev in values:
                    arr[idx] = values[key_rev]
            return arr
        else:
            return np.array([], dtype=np.float64)

    def _expand_bonds(self, n_atoms: int) -> np.ndarray:
        """Expand per-molecule bonds to full system. Returns (n_total_bonds, 2)."""
        n_mol = self.n_molecules or (n_atoms // self.n_beads_per_mol)
        all_bonds = []
        for m in range(n_mol):
            offset = m * self.n_beads_per_mol
            for i, j in self.bonds_local:
                all_bonds.append([offset + i, offset + j])
        return np.array(all_bonds, dtype=np.int64)

    def _expand_params(self, n_atoms: int) -> Tuple[np.ndarray, np.ndarray]:
        """Expand per-bond k, r_eq to full system bonds."""
        n_mol = self.n_molecules or (n_atoms // self.n_beads_per_mol)
        n_bonds_per_mol = len(self.bonds_local)
        k_all = np.tile(self.k_per_bond, n_mol)
        r_eq_all = np.tile(self.r_eq_per_bond, n_mol)
        return k_all, r_eq_all

    def compute_forces(self, positions, types=None, cell=None):
        n_atoms = len(positions)
        forces = np.zeros_like(positions, dtype=np.float64)

        if len(self.k_per_bond) == 0:
            return forces.astype(positions.dtype)

        bonds = self._expand_bonds(n_atoms)
        k_all, r_eq_all = self._expand_params(n_atoms)

        # Vectorized bond computation
        pos_i = positions[bonds[:, 0]]  # (n_bonds, 3)
        pos_j = positions[bonds[:, 1]]  # (n_bonds, 3)
        r_vec = pos_j - pos_i
        r_vec = self._apply_pbc(r_vec, cell)
        r = np.linalg.norm(r_vec, axis=1)  # (n_bonds,)

        # F = -dV/dr * r_hat = -k*(r - r_eq) * r_hat
        mask = r > 1e-10
        f_mag = np.zeros_like(r)
        f_mag[mask] = -k_all[mask] * (r[mask] - r_eq_all[mask])

        f_vec = np.zeros_like(r_vec)
        f_vec[mask] = f_mag[mask, None] * (r_vec[mask] / r[mask, None])

        # Newton's third law: F_i = -f_vec, F_j = +f_vec
        np.add.at(forces, bonds[:, 0], -f_vec)
        np.add.at(forces, bonds[:, 1], f_vec)

        return forces.astype(positions.dtype)

    def compute_energy(self, positions, types=None, cell=None):
        n_atoms = len(positions)
        if len(self.k_per_bond) == 0:
            return 0.0

        bonds = self._expand_bonds(n_atoms)
        k_all, r_eq_all = self._expand_params(n_atoms)

        pos_i = positions[bonds[:, 0]]
        pos_j = positions[bonds[:, 1]]
        r_vec = pos_j - pos_i
        r_vec = self._apply_pbc(r_vec, cell)
        r = np.linalg.norm(r_vec, axis=1)

        # Cap displacement to handle PBC wrapping artifacts in CG coordinates.
        # When CG bead COMs are computed from wrapped atom positions, some bond
        # distances can be ~30 Å instead of ~3 Å, causing energy explosion.
        displacement = r - r_eq_all
        max_displacement = 5.0  # Å — generous cap for PBC outliers
        displacement = np.clip(displacement, -max_displacement, max_displacement)
        energy = np.sum(0.5 * k_all * displacement ** 2)
        return float(energy)


class HarmonicAnglePrior(PriorForceField):
    """
    Harmonic angle prior for CG systems.

    V(θ) = 0.5 * k_θ * (θ - θ_eq)² for each angle triplet (i, j, k)
    where j is the central atom.

    Config keys:
        bond_topology: dict with 'n_beads_per_mol' and 'bonds'
        angle_topology: list of [i, j, k] triplets (auto-generated if not given)
        k_per_angle: list of k values (eV/rad²), matching angle_topology order
        theta_eq_per_angle: list of θ_eq values (radians), matching angle_topology order
        n_molecules: number of molecules (auto-detected if not given)
    """

    def __init__(self, config: Dict):
        super().__init__(config)
        topo = config.get('bond_topology', DLIPC_BOND_TOPOLOGY)
        self.n_beads_per_mol = topo['n_beads_per_mol']
        self.bonds_local = topo['bonds']

        # Angle topology: auto-generate from bonds if not provided
        angle_topo = config.get('angle_topology', None)
        if angle_topo is None:
            self.angles_local = generate_angles_from_bonds(
                self.bonds_local, self.n_beads_per_mol
            )
        else:
            self.angles_local = angle_topo

        # Parameters per angle (in order of angles_local)
        self.k_per_angle = np.array(
            config.get('k_per_angle', []), dtype=np.float64
        )
        self.theta_eq_per_angle = np.array(
            config.get('theta_eq_per_angle', []), dtype=np.float64
        )

        self.n_molecules = config.get('n_molecules', None)

    def _expand_angles(self, n_atoms: int) -> np.ndarray:
        """Expand per-molecule angles to full system. Returns (n_total_angles, 3)."""
        n_mol = self.n_molecules or (n_atoms // self.n_beads_per_mol)
        all_angles = []
        for m in range(n_mol):
            offset = m * self.n_beads_per_mol
            for i, j, k in self.angles_local:
                all_angles.append([offset + i, offset + j, offset + k])
        return np.array(all_angles, dtype=np.int64)

    def _expand_params(self, n_atoms: int) -> Tuple[np.ndarray, np.ndarray]:
        """Expand per-angle k, theta_eq to full system angles."""
        n_mol = self.n_molecules or (n_atoms // self.n_beads_per_mol)
        k_all = np.tile(self.k_per_angle, n_mol)
        theta_eq_all = np.tile(self.theta_eq_per_angle, n_mol)
        return k_all, theta_eq_all

    def compute_forces(self, positions, types=None, cell=None):
        """
        Compute harmonic angle forces.

        F_i = k(θ-θ₀)/sinθ * [r_kj/(d_ij·d_kj) - cosθ·r_ij/d_ij²]
        F_k = k(θ-θ₀)/sinθ * [r_ij/(d_ij·d_kj) - cosθ·r_kj/d_kj²]
        F_j = -(F_i + F_k)
        """
        n_atoms = len(positions)
        forces = np.zeros_like(positions, dtype=np.float64)

        if len(self.k_per_angle) == 0:
            return forces.astype(positions.dtype)

        angles = self._expand_angles(n_atoms)
        k_all, theta_eq_all = self._expand_params(n_atoms)

        pos_i = positions[angles[:, 0]]
        pos_j = positions[angles[:, 1]]
        pos_k = positions[angles[:, 2]]

        r_ij = pos_i - pos_j
        r_kj = pos_k - pos_j

        r_ij = self._apply_pbc(r_ij, cell)
        r_kj = self._apply_pbc(r_kj, cell)

        d_ij = np.linalg.norm(r_ij, axis=1)
        d_kj = np.linalg.norm(r_kj, axis=1)

        # cos(θ) and θ
        dot = np.sum(r_ij * r_kj, axis=1)
        cos_theta = dot / (d_ij * d_kj + 1e-10)
        cos_theta = np.clip(cos_theta, -1.0 + 1e-7, 1.0 - 1e-7)
        theta = np.arccos(cos_theta)
        sin_theta = np.sin(theta)

        # Prefactor = k*(θ - θ_eq) / sin(θ), with guard for near-linear angles
        valid = sin_theta > 1e-6
        prefactor = np.zeros_like(theta)
        prefactor[valid] = (
            k_all[valid] * (theta[valid] - theta_eq_all[valid]) / sin_theta[valid]
        )

        # Cap displacement to handle PBC wrapping artifacts
        max_angular_disp = 5.0  # rad — generous cap
        angular_disp = np.abs(theta - theta_eq_all)
        cap_mask = angular_disp > max_angular_disp
        if np.any(cap_mask):
            prefactor[cap_mask] = np.sign(
                theta[cap_mask] - theta_eq_all[cap_mask]
            ) * k_all[cap_mask] * max_angular_disp / (sin_theta[cap_mask] + 1e-6)

        # F_i = prefactor * [r_kj/(d_ij*d_kj) - cos_theta*r_ij/d_ij²]
        d_ij_dkj = (d_ij * d_kj)[:, None]
        d_ij_sq = (d_ij * d_ij)[:, None]
        d_kj_sq = (d_kj * d_kj)[:, None]

        f_i = prefactor[:, None] * (r_kj / (d_ij_dkj + 1e-10) - cos_theta[:, None] * r_ij / (d_ij_sq + 1e-10))
        f_k = prefactor[:, None] * (r_ij / (d_ij_dkj + 1e-10) - cos_theta[:, None] * r_kj / (d_kj_sq + 1e-10))
        f_j = -(f_i + f_k)

        np.add.at(forces, angles[:, 0], f_i)
        np.add.at(forces, angles[:, 1], f_j)
        np.add.at(forces, angles[:, 2], f_k)

        return forces.astype(positions.dtype)

    def compute_energy(self, positions, types=None, cell=None):
        n_atoms = len(positions)
        if len(self.k_per_angle) == 0:
            return 0.0

        angles = self._expand_angles(n_atoms)
        k_all, theta_eq_all = self._expand_params(n_atoms)

        pos_i = positions[angles[:, 0]]
        pos_j = positions[angles[:, 1]]
        pos_k = positions[angles[:, 2]]

        r_ij = pos_i - pos_j
        r_kj = pos_k - pos_j
        r_ij = self._apply_pbc(r_ij, cell)
        r_kj = self._apply_pbc(r_kj, cell)

        d_ij = np.linalg.norm(r_ij, axis=1)
        d_kj = np.linalg.norm(r_kj, axis=1)

        dot = np.sum(r_ij * r_kj, axis=1)
        cos_theta = dot / (d_ij * d_kj + 1e-10)
        cos_theta = np.clip(cos_theta, -1.0 + 1e-7, 1.0 - 1e-7)
        theta = np.arccos(cos_theta)

        displacement = theta - theta_eq_all
        max_displacement = 5.0  # rad
        displacement = np.clip(displacement, -max_displacement, max_displacement)
        energy = np.sum(0.5 * k_all * displacement ** 2)
        return float(energy)


class RepulsiveLJPrior(PriorForceField):
    """
    Purely repulsive (sigma/r)^12 prior for CG non-bonded interactions.

    V(r) = 4*epsilon*(sigma_ij/r)^12 * fc(r)  for non-bonded pairs.
    Bonded pairs are excluded.

    Config keys:
        sigma_matrix: dict mapping 'ti-tj' → sigma (Å)
        epsilon: float, energy scale (eV), default 0.001
        cutoff: float (Å), default 10.0
        bond_topology: for bond exclusion
        n_molecules: number of molecules
    """

    def __init__(self, config: Dict):
        super().__init__(config)
        self.epsilon = config.get('epsilon', 0.001)
        self.cutoff = config.get('cutoff', 10.0)
        # Maximum force magnitude per pair (eV/Å) to cap rare close contacts
        self.max_force = config.get('max_force', 1.0)

        # sigma matrix: type-pair → sigma
        sigma_dict = config.get('sigma_matrix', {})
        self.sigma_dict = sigma_dict

        # Bond exclusion
        topo = config.get('bond_topology', DLIPC_BOND_TOPOLOGY)
        self.n_beads_per_mol = topo['n_beads_per_mol']
        self.bonds_local = topo['bonds']
        self.n_molecules = config.get('n_molecules', None)

        # Pre-build sigma lookup: (type_i, type_j) → sigma
        self._sigma_lookup: Dict[Tuple[int, int], float] = {}
        for key, val in sigma_dict.items():
            parts = key.split('-')
            ti, tj = int(parts[0]), int(parts[1])
            self._sigma_lookup[(ti, tj)] = val
            self._sigma_lookup[(tj, ti)] = val

        # Default sigma for missing pairs
        if sigma_dict:
            self.default_sigma = np.mean(list(sigma_dict.values()))
        else:
            self.default_sigma = 3.0

    def _get_excluded_set(self, n_atoms: int) -> set:
        """
        Build set of excluded pairs with global indices.

        For CG systems with PBC wrapping artifacts, exclude ALL intra-molecular
        pairs. The repulsive prior only acts on inter-molecular interactions.
        """
        n_mol = self.n_molecules or (n_atoms // self.n_beads_per_mol)
        excluded = set()
        # Exclude ALL intra-molecular pairs (avoids PBC wrapping artifacts)
        for m in range(n_mol):
            offset = m * self.n_beads_per_mol
            for i in range(self.n_beads_per_mol):
                for j in range(i + 1, self.n_beads_per_mol):
                    excluded.add((offset + i, offset + j))
        return excluded

    def _get_sigma(self, ti: int, tj: int) -> float:
        key = (ti, tj)
        if key in self._sigma_lookup:
            return self._sigma_lookup[key]
        return self.default_sigma

    def _cosine_cutoff(self, r: np.ndarray) -> np.ndarray:
        return np.where(
            r < self.cutoff,
            0.5 * (1 + np.cos(np.pi * r / self.cutoff)),
            0.0
        )

    def _cosine_cutoff_derivative(self, r: np.ndarray) -> np.ndarray:
        return np.where(
            r < self.cutoff,
            -0.5 * np.pi / self.cutoff * np.sin(np.pi * r / self.cutoff),
            0.0
        )

    def compute_forces(self, positions, types=None, cell=None):
        """Compute repulsive LJ forces using ASE neighbor list for efficiency."""
        n_atoms = len(positions)
        forces = np.zeros((n_atoms, 3), dtype=np.float64)

        if not self._sigma_lookup:
            return forces.astype(positions.dtype)

        try:
            from ase import Atoms
            from ase.neighborlist import neighbor_list
        except ImportError:
            raise ImportError("ASE is required for RepulsiveLJPrior. Install: pip install ase")

        # Build ASE Atoms for neighbor list
        if cell is not None:
            atoms = Atoms(
                numbers=[1] * n_atoms,  # dummy
                positions=positions,
                cell=cell,
                pbc=True
            )
        else:
            atoms = Atoms(
                numbers=[1] * n_atoms,
                positions=positions,
                pbc=False
            )

        # Get neighbor list
        idx_i, idx_j, r_vec_arr = neighbor_list('ijD', atoms, self.cutoff)

        if len(idx_i) == 0:
            return forces.astype(positions.dtype)

        # Build excluded set
        excluded = self._get_excluded_set(n_atoms)

        # Filter: only i < j to avoid double counting, exclude bonds
        mask_ij = idx_i < idx_j
        idx_i = idx_i[mask_ij]
        idx_j = idx_j[mask_ij]
        r_vec_arr = r_vec_arr[mask_ij]

        # Exclude bonded pairs
        if excluded:
            keep = np.array([
                (idx_i[k], idx_j[k]) not in excluded
                for k in range(len(idx_i))
            ], dtype=bool)
            idx_i = idx_i[keep]
            idx_j = idx_j[keep]
            r_vec_arr = r_vec_arr[keep]

        if len(idx_i) == 0:
            return forces.astype(positions.dtype)

        r = np.linalg.norm(r_vec_arr, axis=1)
        valid = r > 1e-10
        idx_i = idx_i[valid]
        idx_j = idx_j[valid]
        r_vec_arr = r_vec_arr[valid]
        r = r[valid]

        # Get sigma for each pair
        if types is not None:
            sigma_arr = np.array([
                self._get_sigma(int(types[idx_i[k]]), int(types[idx_j[k]]))
                for k in range(len(idx_i))
            ], dtype=np.float64)
        else:
            sigma_arr = np.full(len(idx_i), self.default_sigma, dtype=np.float64)

        # V(r) = 4*eps*(sigma/r)^12 * fc(r)
        sr = sigma_arr / r  # sigma/r
        sr12 = sr ** 12

        fc = self._cosine_cutoff(r)
        dfc_dr = self._cosine_cutoff_derivative(r)

        # dV/dr = 4*eps * [-12*sigma^12/r^13 * fc + (sigma/r)^12 * dfc/dr]
        dV_dr = 4.0 * self.epsilon * (
            -12.0 * sr12 / r * fc + sr12 * dfc_dr
        )

        # Force: F = -dV/dr * r_hat, with magnitude capping
        f_mag = np.abs(dV_dr)
        f_mag = np.minimum(f_mag, self.max_force)
        dV_dr = np.sign(dV_dr) * f_mag

        r_hat = r_vec_arr / r[:, None]
        f_vec = -dV_dr[:, None] * r_hat

        # Newton's third law
        np.add.at(forces, idx_i, -f_vec)
        np.add.at(forces, idx_j, f_vec)

        return forces.astype(positions.dtype)

    def compute_energy(self, positions, types=None, cell=None):
        n_atoms = len(positions)

        if not self._sigma_lookup:
            return 0.0

        try:
            from ase import Atoms
            from ase.neighborlist import neighbor_list
        except ImportError:
            raise ImportError("ASE is required for RepulsiveLJPrior.")

        if cell is not None:
            atoms = Atoms(
                numbers=[1] * n_atoms,
                positions=positions,
                cell=cell,
                pbc=True
            )
        else:
            atoms = Atoms(
                numbers=[1] * n_atoms,
                positions=positions,
                pbc=False
            )

        idx_i, idx_j, r_vec_arr = neighbor_list('ijD', atoms, self.cutoff)

        if len(idx_i) == 0:
            return 0.0

        excluded = self._get_excluded_set(n_atoms)

        mask_ij = idx_i < idx_j
        idx_i = idx_i[mask_ij]
        idx_j = idx_j[mask_ij]
        r_vec_arr = r_vec_arr[mask_ij]

        if excluded:
            keep = np.array([
                (idx_i[k], idx_j[k]) not in excluded
                for k in range(len(idx_i))
            ], dtype=bool)
            idx_i = idx_i[keep]
            idx_j = idx_j[keep]
            r_vec_arr = r_vec_arr[keep]

        if len(idx_i) == 0:
            return 0.0

        r = np.linalg.norm(r_vec_arr, axis=1)
        valid = r > 1e-10
        idx_i = idx_i[valid]
        idx_j = idx_j[valid]
        r = r[valid]

        if types is not None:
            sigma_arr = np.array([
                self._get_sigma(int(types[idx_i[k]]), int(types[idx_j[k]]))
                for k in range(len(idx_i))
            ], dtype=np.float64)
        else:
            sigma_arr = np.full(len(idx_i), self.default_sigma, dtype=np.float64)

        sr12 = (sigma_arr / r) ** 12
        fc = self._cosine_cutoff(r)
        v_per_pair = 4.0 * self.epsilon * sr12 * fc

        # Cap per-pair energy to prevent explosion from close contacts.
        # Consistent with force capping: max V ~ max_force * sigma (generous).
        max_energy_per_pair = self.max_force * 1.0  # eV per pair
        v_per_pair = np.minimum(v_per_pair, max_energy_per_pair)
        energy = np.sum(v_per_pair)
        return float(energy)


class IntraRepulsivePrior(PriorForceField):
    """
    Class-specific intra-molecular repulsive (sigma/r)^12 prior.

    V(r) = 4*epsilon*(sigma_p/r)^12 * fc(r) for an explicit list of
    intra-molecular pairs (typically 1-3/1-4/1-5, i.e. bond-graph
    distance >= 2), each with its own sigma.

    Rationale: the inter-molecular RepulsiveLJPrior excludes ALL
    intra-molecular pairs, and the harmonic bond prior only covers 1-2.
    Without this term, 1-3/1-4/1-5 pairs have zero prior and rely on ML
    extrapolation, which collapses (observed in octanol bulk delta MD).
    Sigma per pair is estimated from AA distance distributions so the
    wall sits strictly below the AA-visited minimum (<= kT at d_min).

    Config keys:
        bond_topology: dict with 'n_beads_per_mol' (for expansion)
        pairs_sigma: list of [i_local, j_local, sigma_A] triplets
        epsilon: float, energy scale (eV), default 0.001
        cutoff: float (A), default 10.0
        max_force: per-pair force cap (eV/A), default 1.0
        n_molecules: number of molecules (auto-detected if not given)
    """

    def __init__(self, config: Dict):
        super().__init__(config)
        topo = config.get('bond_topology', DLIPC_BOND_TOPOLOGY)
        self.n_beads_per_mol = topo['n_beads_per_mol']
        self.n_molecules = config.get('n_molecules', None)
        self.epsilon = config.get('epsilon', 0.001)
        self.cutoff = config.get('cutoff', 10.0)
        self.max_force = config.get('max_force', 1.0)

        pairs_sigma = config.get('pairs_sigma', [])
        self.pairs_local = np.array(
            [[int(p[0]), int(p[1])] for p in pairs_sigma], dtype=np.int64
        ).reshape(-1, 2)
        self.sigma_per_pair = np.array(
            [float(p[2]) for p in pairs_sigma], dtype=np.float64
        )

    def _expand_pairs(self, n_atoms: int):
        """Expand per-molecule pairs/sigma to full system."""
        n_mol = self.n_molecules or (n_atoms // self.n_beads_per_mol)
        offsets = np.arange(n_mol, dtype=np.int64)[:, None, None] * self.n_beads_per_mol
        pairs_all = (self.pairs_local[None, :, :] + offsets).reshape(-1, 2)
        sigma_all = np.tile(self.sigma_per_pair, n_mol)
        return pairs_all, sigma_all

    def _cosine_cutoff(self, r: np.ndarray) -> np.ndarray:
        return np.where(
            r < self.cutoff,
            0.5 * (1 + np.cos(np.pi * r / self.cutoff)),
            0.0
        )

    def _cosine_cutoff_derivative(self, r: np.ndarray) -> np.ndarray:
        return np.where(
            r < self.cutoff,
            -0.5 * np.pi / self.cutoff * np.sin(np.pi * r / self.cutoff),
            0.0
        )

    def compute_forces(self, positions, types=None, cell=None):
        n_atoms = len(positions)
        forces = np.zeros((n_atoms, 3), dtype=np.float64)
        if len(self.sigma_per_pair) == 0:
            return forces.astype(positions.dtype)

        pairs, sigma_arr = self._expand_pairs(n_atoms)
        pos_i = positions[pairs[:, 0]]
        pos_j = positions[pairs[:, 1]]
        r_vec = pos_j - pos_i
        r_vec = self._apply_pbc(r_vec, cell)
        r = np.linalg.norm(r_vec, axis=1)

        valid = (r > 1e-10) & (r < self.cutoff)
        if not valid.any():
            return forces.astype(positions.dtype)
        pairs = pairs[valid]
        sigma_arr = sigma_arr[valid]
        r_vec = r_vec[valid]
        r = r[valid]

        sr12 = (sigma_arr / r) ** 12
        fc = self._cosine_cutoff(r)
        dfc_dr = self._cosine_cutoff_derivative(r)
        dV_dr = 4.0 * self.epsilon * (-12.0 * sr12 / r * fc + sr12 * dfc_dr)

        # Cap force magnitude per pair (consistent with RepulsiveLJPrior)
        f_mag = np.abs(dV_dr)
        f_mag = np.minimum(f_mag, self.max_force)
        dV_dr = np.sign(dV_dr) * f_mag

        r_hat = r_vec / r[:, None]
        f_vec = -dV_dr[:, None] * r_hat
        np.add.at(forces, pairs[:, 0], -f_vec)
        np.add.at(forces, pairs[:, 1], f_vec)
        return forces.astype(positions.dtype)

    def compute_energy(self, positions, types=None, cell=None):
        n_atoms = len(positions)
        if len(self.sigma_per_pair) == 0:
            return 0.0

        pairs, sigma_arr = self._expand_pairs(n_atoms)
        pos_i = positions[pairs[:, 0]]
        pos_j = positions[pairs[:, 1]]
        r_vec = pos_j - pos_i
        r_vec = self._apply_pbc(r_vec, cell)
        r = np.linalg.norm(r_vec, axis=1)

        valid = (r > 1e-10) & (r < self.cutoff)
        if not valid.any():
            return 0.0
        sigma_arr = sigma_arr[valid]
        r = r[valid]

        sr12 = (sigma_arr / r) ** 12
        fc = self._cosine_cutoff(r)
        v_per_pair = 4.0 * self.epsilon * sr12 * fc
        # Cap per-pair energy (consistent with RepulsiveLJPrior)
        v_per_pair = np.minimum(v_per_pair, self.max_force * 1.0)
        return float(np.sum(v_per_pair))


class HarmonicRepulsivePrior(PriorForceField):
    """
    Combined Harmonic Bond + Harmonic Angle + Repulsive LJ prior for CG delta learning.

    This is the recommended prior for CG lipid/protein systems,
    following Majewski et al., Nature Comm. (2023).

    Config keys:
        bond_topology: dict with 'n_beads_per_mol' and 'bonds'
        harmonic: dict with 'k_per_bond', 'r_eq_per_bond'
        angle: dict with 'k_per_angle', 'theta_eq_per_angle' (optional)
        repulsive: dict with 'sigma_matrix', 'epsilon', 'cutoff'
        n_molecules: number of molecules
    """

    def __init__(self, config: Dict):
        super().__init__(config)
        topo = config.get('bond_topology', DLIPC_BOND_TOPOLOGY)
        n_mol = config.get('n_molecules', None)

        harmonic_cfg = config.get('harmonic', {})
        angle_cfg = config.get('angle', None)
        repulsive_cfg = config.get('repulsive', {})

        # Build HarmonicBondPrior config
        harm_config = {
            'type': 'harmonic_bond',
            'bond_topology': topo,
            'k_per_bond': harmonic_cfg.get('k_per_bond', harmonic_cfg.get('k', [])),
            'r_eq_per_bond': harmonic_cfg.get('r_eq_per_bond', harmonic_cfg.get('r_eq', [])),
            'n_molecules': n_mol,
        }
        self.harmonic = HarmonicBondPrior(harm_config)

        # Build HarmonicAnglePrior (optional, backward compatible)
        if angle_cfg is not None:
            angle_config = {
                'type': 'harmonic_angle',
                'bond_topology': topo,
                'angle_topology': angle_cfg.get('angle_topology', None),
                'k_per_angle': angle_cfg.get('k_per_angle', angle_cfg.get('k', [])),
                'theta_eq_per_angle': angle_cfg.get('theta_eq_per_angle', angle_cfg.get('theta_eq', [])),
                'n_molecules': n_mol,
            }
            self.angle = HarmonicAnglePrior(angle_config)
        else:
            self.angle = None

        # Build RepulsiveLJPrior config
        rep_config = {
            'type': 'repulsive_lj',
            'bond_topology': topo,
            'sigma_matrix': repulsive_cfg.get('sigma_matrix', repulsive_cfg.get('sigma', {})),
            'epsilon': repulsive_cfg.get('epsilon', 0.001),
            'cutoff': repulsive_cfg.get('cutoff', 10.0),
            'n_molecules': n_mol,
        }
        self.repulsive = RepulsiveLJPrior(rep_config)

        # Build IntraRepulsivePrior (optional, backward compatible).
        # Covers 1-3/1-4/1-5 pairs that the inter-molecular repulsive
        # excludes and the harmonic bond prior does not reach.
        intra_cfg = config.get('repulsive_intra', None)
        if intra_cfg is not None:
            intra_config = {
                'type': 'intra_repulsive',
                'bond_topology': topo,
                'pairs_sigma': intra_cfg.get('pairs_sigma', []),
                'epsilon': intra_cfg.get('epsilon', repulsive_cfg.get('epsilon', 0.001)),
                'cutoff': intra_cfg.get('cutoff', repulsive_cfg.get('cutoff', 10.0)),
                'max_force': intra_cfg.get('max_force', repulsive_cfg.get('max_force', 1.0)),
                'n_molecules': n_mol,
            }
            self.intra = IntraRepulsivePrior(intra_config)
        else:
            self.intra = None

    def compute_forces(self, positions, types=None, cell=None):
        f_harm = self.harmonic.compute_forces(positions, types, cell)
        f_rep = self.repulsive.compute_forces(positions, types, cell)
        f_total = f_harm + f_rep
        if self.angle is not None:
            f_total = f_total + self.angle.compute_forces(positions, types, cell)
        if self.intra is not None:
            f_total = f_total + self.intra.compute_forces(positions, types, cell)
        return f_total

    def compute_energy(self, positions, types=None, cell=None):
        e_harm = self.harmonic.compute_energy(positions, types, cell)
        e_rep = self.repulsive.compute_energy(positions, types, cell)
        e_total = e_harm + e_rep
        if self.angle is not None:
            e_total = e_total + self.angle.compute_energy(positions, types, cell)
        if self.intra is not None:
            e_total = e_total + self.intra.compute_energy(positions, types, cell)
        return e_total


# =============================================================================
# Additional Bond Priors
# =============================================================================

class MorseBondPrior(PriorForceField):
    """
    Morse bond prior: V(r) = D_e * [1 - exp(-a*(r - r_eq))]^2

    Captures bond dissociation (anharmonic). Reduces to harmonic near r_eq.

    Config keys:
        bond_topology: dict with 'n_beads_per_mol' and 'bonds'
        D_e_per_bond: list of D_e values (eV)
        a_per_bond: list of a values (1/Å)
        r_eq_per_bond: list of r_eq values (Å)
        n_molecules: number of molecules
    """

    def __init__(self, config: Dict):
        super().__init__(config)
        topo = config.get('bond_topology', {})
        self.n_beads_per_mol = topo['n_beads_per_mol']
        self.bonds_local = topo['bonds']
        self.D_e = np.array(config.get('D_e_per_bond', []), dtype=np.float64)
        self.a = np.array(config.get('a_per_bond', []), dtype=np.float64)
        self.r_eq = np.array(config.get('r_eq_per_bond', []), dtype=np.float64)
        self.n_molecules = config.get('n_molecules', None)

    def _expand_bonds(self, n_atoms):
        n_mol = self.n_molecules or (n_atoms // self.n_beads_per_mol)
        all_bonds = []
        for m in range(n_mol):
            offset = m * self.n_beads_per_mol
            for i, j in self.bonds_local:
                all_bonds.append([offset + i, offset + j])
        return np.array(all_bonds, dtype=np.int64)

    def compute_forces(self, positions, types=None, cell=None):
        n_atoms = len(positions)
        forces = np.zeros_like(positions, dtype=np.float64)
        if len(self.D_e) == 0:
            return forces.astype(positions.dtype)

        bonds = self._expand_bonds(n_atoms)
        n_mol = self.n_molecules or (n_atoms // self.n_beads_per_mol)
        D_e = np.tile(self.D_e, n_mol)
        a = np.tile(self.a, n_mol)
        r_eq = np.tile(self.r_eq, n_mol)

        r_vec = positions[bonds[:, 1]] - positions[bonds[:, 0]]
        r_vec = self._apply_pbc(r_vec, cell)
        r = np.linalg.norm(r_vec, axis=1)

        mask = r > 1e-10
        exp_term = np.exp(-a * (r - r_eq))
        # F = -dV/dr = -2*D_e*a*(1-exp)*exp
        f_mag = np.zeros_like(r)
        f_mag[mask] = -2.0 * D_e[mask] * a[mask] * (1.0 - exp_term[mask]) * exp_term[mask]

        f_vec = np.zeros_like(r_vec)
        f_vec[mask] = f_mag[mask, None] * (r_vec[mask] / r[mask, None])

        np.add.at(forces, bonds[:, 0], -f_vec)
        np.add.at(forces, bonds[:, 1], f_vec)
        return forces.astype(positions.dtype)

    def compute_energy(self, positions, types=None, cell=None):
        n_atoms = len(positions)
        if len(self.D_e) == 0:
            return 0.0
        bonds = self._expand_bonds(n_atoms)
        n_mol = self.n_molecules or (n_atoms // self.n_beads_per_mol)
        D_e = np.tile(self.D_e, n_mol)
        a = np.tile(self.a, n_mol)
        r_eq = np.tile(self.r_eq, n_mol)

        r_vec = positions[bonds[:, 1]] - positions[bonds[:, 0]]
        r_vec = self._apply_pbc(r_vec, cell)
        r = np.linalg.norm(r_vec, axis=1)
        return float(np.sum(D_e * (1.0 - np.exp(-a * (r - r_eq)))**2))


class FENEBondPrior(PriorForceField):
    """
    FENE bond prior: V(r) = -0.5 * k * R0^2 * ln[1 - (r/R0)^2]

    Finite extensibility — bond cannot stretch beyond R0.

    Config keys:
        bond_topology, k_per_bond, R0_per_bond, n_molecules
    """

    def __init__(self, config: Dict):
        super().__init__(config)
        topo = config.get('bond_topology', {})
        self.n_beads_per_mol = topo['n_beads_per_mol']
        self.bonds_local = topo['bonds']
        self.k = np.array(config.get('k_per_bond', []), dtype=np.float64)
        self.R0 = np.array(config.get('R0_per_bond', []), dtype=np.float64)
        self.n_molecules = config.get('n_molecules', None)

    def _expand_bonds(self, n_atoms):
        n_mol = self.n_molecules or (n_atoms // self.n_beads_per_mol)
        all_bonds = []
        for m in range(n_mol):
            offset = m * self.n_beads_per_mol
            for i, j in self.bonds_local:
                all_bonds.append([offset + i, offset + j])
        return np.array(all_bonds, dtype=np.int64)

    def compute_forces(self, positions, types=None, cell=None):
        n_atoms = len(positions)
        forces = np.zeros_like(positions, dtype=np.float64)
        if len(self.k) == 0:
            return forces.astype(positions.dtype)

        bonds = self._expand_bonds(n_atoms)
        n_mol = self.n_molecules or (n_atoms // self.n_beads_per_mol)
        k = np.tile(self.k, n_mol)
        R0 = np.tile(self.R0, n_mol)

        r_vec = positions[bonds[:, 1]] - positions[bonds[:, 0]]
        r_vec = self._apply_pbc(r_vec, cell)
        r = np.linalg.norm(r_vec, axis=1)

        ratio2 = (r / R0)**2
        mask = (r > 1e-10) & (ratio2 < 1.0)
        # F = -dV/dr = -k*r/(1-(r/R0)^2)
        f_mag = np.zeros_like(r)
        f_mag[mask] = -k[mask] * r[mask] / (1.0 - ratio2[mask])

        f_vec = np.zeros_like(r_vec)
        f_vec[mask] = f_mag[mask, None] * (r_vec[mask] / r[mask, None])

        np.add.at(forces, bonds[:, 0], -f_vec)
        np.add.at(forces, bonds[:, 1], f_vec)
        return forces.astype(positions.dtype)

    def compute_energy(self, positions, types=None, cell=None):
        n_atoms = len(positions)
        if len(self.k) == 0:
            return 0.0
        bonds = self._expand_bonds(n_atoms)
        n_mol = self.n_molecules or (n_atoms // self.n_beads_per_mol)
        k = np.tile(self.k, n_mol)
        R0 = np.tile(self.R0, n_mol)

        r_vec = positions[bonds[:, 1]] - positions[bonds[:, 0]]
        r_vec = self._apply_pbc(r_vec, cell)
        r = np.linalg.norm(r_vec, axis=1)
        ratio2 = (r / R0)**2
        valid = ratio2 < 1.0
        energy = np.zeros_like(r)
        energy[valid] = -0.5 * k[valid] * R0[valid]**2 * np.log(1.0 - ratio2[valid])
        return float(np.sum(energy))


# =============================================================================
# Additional Angle Priors
# =============================================================================

class CosineHarmonicAnglePrior(PriorForceField):
    """
    Cosine harmonic angle prior: V(θ) = 0.5 * k * (cos(θ) - cos(θ_eq))^2

    Numerically stable at θ=180° (linear angles). Preferred for CG systems.

    Config keys:
        bond_topology, angle_topology, k_per_angle, theta_eq_per_angle, n_molecules
    """

    def __init__(self, config: Dict):
        super().__init__(config)
        topo = config.get('bond_topology', {})
        self.n_beads_per_mol = topo['n_beads_per_mol']
        self.bonds_local = topo['bonds']

        angle_topo = config.get('angle_topology', None)
        if angle_topo is None:
            self.angles_local = generate_angles_from_bonds(
                self.bonds_local, self.n_beads_per_mol
            )
        else:
            self.angles_local = angle_topo

        self.k_per_angle = np.array(config.get('k_per_angle', []), dtype=np.float64)
        self.theta_eq_per_angle = np.array(config.get('theta_eq_per_angle', []), dtype=np.float64)
        self.n_molecules = config.get('n_molecules', None)

    def _expand_angles(self, n_atoms):
        n_mol = self.n_molecules or (n_atoms // self.n_beads_per_mol)
        all_angles = []
        for m in range(n_mol):
            offset = m * self.n_beads_per_mol
            for i, j, k in self.angles_local:
                all_angles.append([offset + i, offset + j, offset + k])
        return np.array(all_angles, dtype=np.int64)

    def compute_forces(self, positions, types=None, cell=None):
        """
        F = -dV/dθ * dθ/dr
        dV/dθ = -k*(cosθ - cosθ₀)*sinθ
        Combined: dV/d(cosθ) = k*(cosθ - cosθ₀)
        """
        n_atoms = len(positions)
        forces = np.zeros_like(positions, dtype=np.float64)
        if len(self.k_per_angle) == 0:
            return forces.astype(positions.dtype)

        angles = self._expand_angles(n_atoms)
        n_mol = self.n_molecules or (n_atoms // self.n_beads_per_mol)
        k_all = np.tile(self.k_per_angle, n_mol)
        theta_eq_all = np.tile(self.theta_eq_per_angle, n_mol)
        cos_eq = np.cos(theta_eq_all)

        pos_i = positions[angles[:, 0]]
        pos_j = positions[angles[:, 1]]
        pos_k = positions[angles[:, 2]]

        r_ij = pos_i - pos_j
        r_kj = pos_k - pos_j
        r_ij = self._apply_pbc(r_ij, cell)
        r_kj = self._apply_pbc(r_kj, cell)

        d_ij = np.linalg.norm(r_ij, axis=1)
        d_kj = np.linalg.norm(r_kj, axis=1)

        dot = np.sum(r_ij * r_kj, axis=1)
        cos_theta = dot / (d_ij * d_kj + 1e-10)
        cos_theta = np.clip(cos_theta, -1.0 + 1e-7, 1.0 - 1e-7)

        # dV/d(cosθ) = k*(cosθ - cosθ₀)
        # d(cosθ)/dr_i = r_kj/(d_ij*d_kj) - cosθ*r_ij/d_ij²
        prefactor = k_all * (cos_theta - cos_eq)

        d_ij_dkj = (d_ij * d_kj)[:, None]
        d_ij_sq = (d_ij * d_ij)[:, None]
        d_kj_sq = (d_kj * d_kj)[:, None]

        f_i = -prefactor[:, None] * (r_kj / (d_ij_dkj + 1e-10) - cos_theta[:, None] * r_ij / (d_ij_sq + 1e-10))
        f_k = -prefactor[:, None] * (r_ij / (d_ij_dkj + 1e-10) - cos_theta[:, None] * r_kj / (d_kj_sq + 1e-10))
        f_j = -(f_i + f_k)

        np.add.at(forces, angles[:, 0], f_i)
        np.add.at(forces, angles[:, 1], f_j)
        np.add.at(forces, angles[:, 2], f_k)
        return forces.astype(positions.dtype)

    def compute_energy(self, positions, types=None, cell=None):
        n_atoms = len(positions)
        if len(self.k_per_angle) == 0:
            return 0.0
        angles = self._expand_angles(n_atoms)
        n_mol = self.n_molecules or (n_atoms // self.n_beads_per_mol)
        k_all = np.tile(self.k_per_angle, n_mol)
        theta_eq_all = np.tile(self.theta_eq_per_angle, n_mol)
        cos_eq = np.cos(theta_eq_all)

        pos_i = positions[angles[:, 0]]
        pos_j = positions[angles[:, 1]]
        pos_k = positions[angles[:, 2]]
        r_ij = pos_i - pos_j
        r_kj = pos_k - pos_j
        r_ij = self._apply_pbc(r_ij, cell)
        r_kj = self._apply_pbc(r_kj, cell)
        d_ij = np.linalg.norm(r_ij, axis=1)
        d_kj = np.linalg.norm(r_kj, axis=1)
        cos_theta = np.sum(r_ij * r_kj, axis=1) / (d_ij * d_kj + 1e-10)
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        return float(np.sum(0.5 * k_all * (cos_theta - cos_eq)**2))


# =============================================================================
# Dihedral Priors
# =============================================================================

class ProperDihedralPrior(PriorForceField):
    """
    Proper dihedral prior with multiple functional forms.

    Supported types:
        'cosine':  V(φ) = k * [1 + cos(n*φ - δ)]            (CHARMM/AMBER)
        'opls':    V(φ) = Σ Vi/2 * [1 + (-1)^(i+1) cos(iφ)]  (OPLS-AA)
        'fourier': V(φ) = v0 + Σ [a_n*cos(nφ) + b_n*sin(nφ)] (General Fourier)
        'rb':      V(ψ) = Σ C_n * cos(ψ)^n, ψ = φ - π        (Ryckaert-Bellemans)

    Config keys:
        bond_topology: dict with 'n_beads_per_mol' and 'bonds'
        dihedral_topology: list of [i,j,k,l] quadruplets
        dihedral_type: 'cosine', 'opls', 'fourier', or 'rb'
        params_per_dihedral: list of param dicts per dihedral
        n_molecules: number of molecules
    """

    def __init__(self, config: Dict):
        super().__init__(config)
        topo = config.get('bond_topology', {})
        self.n_beads_per_mol = topo['n_beads_per_mol']
        self.dihedrals_local = config.get('dihedral_topology', [])
        self.dihedral_type = config.get('dihedral_type', 'cosine')
        self.params = config.get('params_per_dihedral', [])
        self.n_molecules = config.get('n_molecules', None)

    def _expand_dihedrals(self, n_atoms):
        n_mol = self.n_molecules or (n_atoms // self.n_beads_per_mol)
        all_dihs = []
        for m in range(n_mol):
            offset = m * self.n_beads_per_mol
            for i, j, k, l in self.dihedrals_local:
                all_dihs.append([offset+i, offset+j, offset+k, offset+l])
        return np.array(all_dihs, dtype=np.int64) if all_dihs else np.zeros((0, 4), dtype=np.int64)

    @staticmethod
    def _compute_dihedral_angle(p1, p2, p3, p4, cell=None):
        """Compute dihedral angle φ for quadruplet (p1-p2-p3-p4)."""
        b1 = p2 - p1
        b2 = p3 - p2
        b3 = p4 - p3
        n1 = np.cross(b1, b2)
        n2 = np.cross(b2, b3)
        n1_norm = np.linalg.norm(n1, axis=1, keepdims=True)
        n2_norm = np.linalg.norm(n2, axis=1, keepdims=True)
        n1_norm = np.where(n1_norm < 1e-10, 1.0, n1_norm)
        n2_norm = np.where(n2_norm < 1e-10, 1.0, n2_norm)
        n1 = n1 / n1_norm
        n2 = n2 / n2_norm
        m1 = np.cross(n1, b2 / (np.linalg.norm(b2, axis=1, keepdims=True) + 1e-10))
        cos_phi = np.sum(n1 * n2, axis=1)
        sin_phi = np.sum(m1 * n2, axis=1)
        phi = np.arctan2(sin_phi, cos_phi)
        return phi

    def _compute_V_dV(self, phi, params):
        """Compute V and dV/dφ for given dihedral angles."""
        V = np.zeros_like(phi)
        dV = np.zeros_like(phi)

        if self.dihedral_type == 'cosine':
            # V = k * [1 + cos(n*φ - δ)]
            for idx, p in enumerate(params):
                k = p.get('k', 0.0)
                n = p.get('n', 1)
                delta = p.get('delta', 0.0)
                V[idx] = k * (1.0 + np.cos(n * phi[idx] - delta))
                dV[idx] = -k * n * np.sin(n * phi[idx] - delta)

        elif self.dihedral_type == 'opls':
            # V = V1/2(1+cosφ) + V2/2(1-cos2φ) + V3/2(1+cos3φ) + V4/2(1-cos4φ)
            for idx, p in enumerate(params):
                V_coeffs = [p.get('V1', 0), p.get('V2', 0), p.get('V3', 0), p.get('V4', 0)]
                signs = [1, -1, 1, -1]
                for i, (Vi, si) in enumerate(zip(V_coeffs, signs)):
                    n = i + 1
                    V[idx] += Vi / 2.0 * (1.0 + si * np.cos(n * phi[idx]))
                    dV[idx] += -Vi / 2.0 * si * n * np.sin(n * phi[idx])

        elif self.dihedral_type == 'rb':
            # V(ψ) = Σ C_n cos(ψ)^n, ψ = φ - π
            for idx, p in enumerate(params):
                C = [p.get(f'C{n}', 0.0) for n in range(6)]
                psi = phi[idx] - np.pi
                cos_psi = np.cos(psi)
                for n, cn in enumerate(C):
                    V[idx] += cn * cos_psi**n
                    if n > 0:
                        dV[idx] += -cn * n * cos_psi**(n-1) * np.sin(psi)

        elif self.dihedral_type == 'fourier':
            # V = v0 + Σ [a_n cos(nφ) + b_n sin(nφ)]
            for idx, p in enumerate(params):
                v0 = p.get('v0', 0.0)
                V[idx] = v0
                for n in range(1, 7):
                    an = p.get(f'a{n}', 0.0)
                    bn = p.get(f'b{n}', 0.0)
                    V[idx] += an * np.cos(n * phi[idx]) + bn * np.sin(n * phi[idx])
                    dV[idx] += -an * n * np.sin(n * phi[idx]) + bn * n * np.cos(n * phi[idx])

        return V, dV

    def compute_forces(self, positions, types=None, cell=None):
        n_atoms = len(positions)
        forces = np.zeros_like(positions, dtype=np.float64)
        if len(self.dihedrals_local) == 0 or len(self.params) == 0:
            return forces.astype(positions.dtype)

        dihedrals = self._expand_dihedrals(n_atoms)
        n_mol = self.n_molecules or (n_atoms // self.n_beads_per_mol)

        for d_idx in range(len(dihedrals)):
            i, j, k, l = dihedrals[d_idx]
            p_idx = d_idx % len(self.params)

            p1, p2, p3, p4 = positions[i], positions[j], positions[k], positions[l]
            b1 = p2 - p1
            b2 = p3 - p2
            b3 = p4 - p3

            n1 = np.cross(b1, b2)
            n2 = np.cross(b2, b3)
            n1_len = np.linalg.norm(n1)
            n2_len = np.linalg.norm(n2)
            if n1_len < 1e-10 or n2_len < 1e-10:
                continue
            n1 /= n1_len
            n2 /= n2_len
            b2_norm = b2 / (np.linalg.norm(b2) + 1e-10)
            m1 = np.cross(n1, b2_norm)
            phi = np.arctan2(np.dot(m1, n2), np.dot(n1, n2))

            _, dV = self._compute_V_dV(np.array([phi]), [self.params[p_idx]])
            dV_dphi = dV[0]

            # Numerical force via central difference (robust for all dihedral types)
            dphi = 1e-7
            # dφ/dr using chain rule through Cartesian coordinates
            # Use the standard LAMMPS dihedral force decomposition
            b2_len = np.linalg.norm(b2)
            n1_sq = np.dot(n1, n1) * n1_len**2
            n2_sq = np.dot(n2, n2) * n2_len**2

            # Forces on each atom (Allen & Tildesley formulation)
            f1 = -dV_dphi * b2_len / (n1_len**2 + 1e-10) * n1 * n1_len
            f4 = dV_dphi * b2_len / (n2_len**2 + 1e-10) * n2 * n2_len
            f2_contrib1 = (np.dot(b1, b2) / (b2_len**2 + 1e-10) - 1.0) * f1
            f2_contrib2 = -(np.dot(b3, b2) / (b2_len**2 + 1e-10)) * f4
            f2 = f2_contrib1 + f2_contrib2
            f3 = -(f1 + f2 + f4)

            forces[i] += f1
            forces[j] += f2
            forces[k] += f3
            forces[l] += f4

        return forces.astype(positions.dtype)

    def compute_energy(self, positions, types=None, cell=None):
        n_atoms = len(positions)
        if len(self.dihedrals_local) == 0 or len(self.params) == 0:
            return 0.0

        dihedrals = self._expand_dihedrals(n_atoms)
        total_energy = 0.0

        for d_idx in range(len(dihedrals)):
            i, j, k, l = dihedrals[d_idx]
            p_idx = d_idx % len(self.params)
            phi = self._compute_dihedral_angle(
                positions[i:i+1], positions[j:j+1],
                positions[k:k+1], positions[l:l+1]
            )[0]
            V, _ = self._compute_V_dV(np.array([phi]), [self.params[p_idx]])
            total_energy += V[0]
        return float(total_energy)


# =============================================================================
# Additional Non-bonded Priors
# =============================================================================

class LJPrior(PriorForceField):
    """
    Full Lennard-Jones 12-6 prior: V(r) = 4ε[(σ/r)^12 - (σ/r)^6]

    Config keys:
        sigma_matrix: dict 'ti-tj' → σ (Å)
        epsilon_matrix: dict 'ti-tj' → ε (eV)
        cutoff: float (Å)
        bond_topology, n_molecules: for exclusion
    """

    def __init__(self, config: Dict):
        super().__init__(config)
        self.cutoff = config.get('cutoff', 10.0)
        self.sigma_dict = config.get('sigma_matrix', {})
        self.epsilon_dict = config.get('epsilon_matrix', {})
        topo = config.get('bond_topology', {'n_beads_per_mol': 1, 'bonds': []})
        self.n_beads_per_mol = topo['n_beads_per_mol']
        self.bonds_local = topo['bonds']
        self.n_molecules = config.get('n_molecules', None)

    def _get_params(self, ti, tj):
        for key_fmt in [f"{ti}-{tj}", f"{tj}-{ti}"]:
            if key_fmt in self.sigma_dict:
                sig = self.sigma_dict[key_fmt]
                eps = self.epsilon_dict.get(key_fmt, 0.001)
                return sig, eps
        return 3.0, 0.001

    def compute_forces(self, positions, types=None, cell=None):
        from ase import Atoms
        from ase.neighborlist import neighbor_list

        n_atoms = len(positions)
        forces = np.zeros((n_atoms, 3), dtype=np.float64)
        if not self.sigma_dict:
            return forces.astype(positions.dtype)

        use_pbc = cell is not None and np.abs(cell).sum() > 1e-6
        if use_pbc:
            atoms = Atoms(symbols=['X']*n_atoms, positions=positions, cell=cell, pbc=True)
        else:
            atoms = Atoms(symbols=['X']*n_atoms, positions=positions, pbc=False)

        ii, jj, Sij = neighbor_list('ijS', atoms, self.cutoff)

        # Build exclusion set
        n_mol = self.n_molecules or (n_atoms // self.n_beads_per_mol)
        excluded = set()
        for m in range(n_mol):
            offset = m * self.n_beads_per_mol
            for bi, bj in self.bonds_local:
                excluded.add((offset+bi, offset+bj))
                excluded.add((offset+bj, offset+bi))

        for idx in range(len(ii)):
            i, j = ii[idx], jj[idx]
            if i >= j:  # avoid double counting
                continue
            if (i, j) in excluded or (j, i) in excluded:
                continue
            r_vec = positions[j] - positions[i]
            if use_pbc:
                r_vec += Sij[idx] @ cell
            r = np.linalg.norm(r_vec)
            if r < 1e-10 or r > self.cutoff:
                continue

            ti = types[i] if types is not None else 0
            tj = types[j] if types is not None else 0
            sig, eps = self._get_params(ti, tj)

            sr6 = (sig / r)**6
            sr12 = sr6**2
            # F = -dV/dr * r_hat = 4ε(12σ¹²/r¹³ - 6σ⁶/r⁷) * r_hat
            f_mag = 4.0 * eps * (12.0 * sr12 / r - 6.0 * sr6 / r)
            f_vec = f_mag * r_vec / r
            forces[i] -= f_vec
            forces[j] += f_vec

        return forces.astype(positions.dtype)

    def compute_energy(self, positions, types=None, cell=None):
        from ase import Atoms
        from ase.neighborlist import neighbor_list

        n_atoms = len(positions)
        if not self.sigma_dict:
            return 0.0

        use_pbc = cell is not None and np.abs(cell).sum() > 1e-6
        if use_pbc:
            atoms = Atoms(symbols=['X']*n_atoms, positions=positions, cell=cell, pbc=True)
        else:
            atoms = Atoms(symbols=['X']*n_atoms, positions=positions, pbc=False)

        ii, jj, Sij = neighbor_list('ijS', atoms, self.cutoff)
        energy = 0.0

        n_mol = self.n_molecules or (n_atoms // self.n_beads_per_mol)
        excluded = set()
        for m in range(n_mol):
            offset = m * self.n_beads_per_mol
            for bi, bj in self.bonds_local:
                excluded.add((offset+bi, offset+bj))

        for idx in range(len(ii)):
            i, j = ii[idx], jj[idx]
            if i >= j:
                continue
            if (i, j) in excluded or (j, i) in excluded:
                continue
            r_vec = positions[j] - positions[i]
            if use_pbc:
                r_vec += Sij[idx] @ cell
            r = np.linalg.norm(r_vec)
            if r < 1e-10 or r > self.cutoff:
                continue

            ti = types[i] if types is not None else 0
            tj = types[j] if types is not None else 0
            sig, eps = self._get_params(ti, tj)
            sr6 = (sig / r)**6
            energy += 4.0 * eps * (sr6**2 - sr6)

        return float(energy)


class WCAPrior(PriorForceField):
    """
    WCA (Weeks-Chandler-Andersen) prior — purely repulsive LJ.

    V(r) = 4ε[(σ/r)^12 - (σ/r)^6] + ε   for r < 2^(1/6)*σ
    V(r) = 0                                for r >= 2^(1/6)*σ

    Config keys:
        sigma_matrix, epsilon_matrix, bond_topology, n_molecules
    """

    def __init__(self, config: Dict):
        super().__init__(config)
        self.sigma_dict = config.get('sigma_matrix', {})
        self.epsilon_dict = config.get('epsilon_matrix', {})
        topo = config.get('bond_topology', {'n_beads_per_mol': 1, 'bonds': []})
        self.n_beads_per_mol = topo['n_beads_per_mol']
        self.bonds_local = topo['bonds']
        self.n_molecules = config.get('n_molecules', None)
        # WCA cutoff is max(2^(1/6)*σ) over all pairs
        if self.sigma_dict:
            max_sig = max(self.sigma_dict.values())
        else:
            max_sig = 3.0
        self.cutoff = 2.0**(1.0/6.0) * max_sig + 0.1

    def _get_params(self, ti, tj):
        for key_fmt in [f"{ti}-{tj}", f"{tj}-{ti}"]:
            if key_fmt in self.sigma_dict:
                sig = self.sigma_dict[key_fmt]
                eps = self.epsilon_dict.get(key_fmt, 0.001)
                return sig, eps
        return 3.0, 0.001

    def compute_forces(self, positions, types=None, cell=None):
        from ase import Atoms
        from ase.neighborlist import neighbor_list

        n_atoms = len(positions)
        forces = np.zeros((n_atoms, 3), dtype=np.float64)
        if not self.sigma_dict:
            return forces.astype(positions.dtype)

        use_pbc = cell is not None and np.abs(cell).sum() > 1e-6
        if use_pbc:
            atoms = Atoms(symbols=['X']*n_atoms, positions=positions, cell=cell, pbc=True)
        else:
            atoms = Atoms(symbols=['X']*n_atoms, positions=positions, pbc=False)

        ii, jj, Sij = neighbor_list('ijS', atoms, self.cutoff)

        n_mol = self.n_molecules or (n_atoms // self.n_beads_per_mol)
        excluded = set()
        for m in range(n_mol):
            offset = m * self.n_beads_per_mol
            for bi, bj in self.bonds_local:
                excluded.add((offset+bi, offset+bj))
                excluded.add((offset+bj, offset+bi))

        for idx in range(len(ii)):
            i, j = ii[idx], jj[idx]
            if i >= j:  # avoid double counting (neighbor_list returns both i->j and j->i)
                continue
            if (i, j) in excluded or (j, i) in excluded:
                continue
            r_vec = positions[j] - positions[i]
            if use_pbc:
                r_vec += Sij[idx] @ cell
            r = np.linalg.norm(r_vec)
            ti = types[i] if types is not None else 0
            tj = types[j] if types is not None else 0
            sig, eps = self._get_params(ti, tj)
            r_cut = 2.0**(1.0/6.0) * sig

            if r < 1e-10 or r >= r_cut:
                continue

            sr6 = (sig / r)**6
            sr12 = sr6**2
            f_mag = 4.0 * eps * (12.0 * sr12 / r - 6.0 * sr6 / r)
            f_vec = f_mag * r_vec / r
            forces[i] -= f_vec
            forces[j] += f_vec

        return forces.astype(positions.dtype)

    def compute_energy(self, positions, types=None, cell=None):
        from ase import Atoms
        from ase.neighborlist import neighbor_list

        n_atoms = len(positions)
        if not self.sigma_dict:
            return 0.0

        use_pbc = cell is not None and np.abs(cell).sum() > 1e-6
        if use_pbc:
            atoms = Atoms(symbols=['X']*n_atoms, positions=positions, cell=cell, pbc=True)
        else:
            atoms = Atoms(symbols=['X']*n_atoms, positions=positions, pbc=False)

        ii, jj, Sij = neighbor_list('ijS', atoms, self.cutoff)
        energy = 0.0

        n_mol = self.n_molecules or (n_atoms // self.n_beads_per_mol)
        excluded = set()
        for m in range(n_mol):
            offset = m * self.n_beads_per_mol
            for bi, bj in self.bonds_local:
                excluded.add((offset+bi, offset+bj))

        for idx in range(len(ii)):
            i, j = ii[idx], jj[idx]
            if i >= j:
                continue
            if (i, j) in excluded or (j, i) in excluded:
                continue
            r_vec = positions[j] - positions[i]
            if use_pbc:
                r_vec += Sij[idx] @ cell
            r = np.linalg.norm(r_vec)
            ti = types[i] if types is not None else 0
            tj = types[j] if types is not None else 0
            sig, eps = self._get_params(ti, tj)
            r_cut = 2.0**(1.0/6.0) * sig

            if r < 1e-10 or r >= r_cut:
                continue

            sr6 = (sig / r)**6
            energy += 4.0 * eps * (sr6**2 - sr6) + eps

        return float(energy)


def estimate_prior_params_from_npz(
    npz_path: str,
    bond_topology: Dict,
    n_sample: int = 100,
    temperature: float = 300.0,
    sigma_scale: float = 0.9,
    repulsive_cutoff: float = 10.0,
    subsample: int = 1,
) -> Dict:
    """
    Estimate harmonic bond and repulsive LJ parameters from CG trajectory.

    For each bond type-pair:
        r_eq = mean(bond distance)
        k = kT / var(bond distance)  (equipartition theorem)

    For each non-bonded type-pair:
        sigma = sigma_scale * min(pair distance)

    Args:
        npz_path: Path to CG NPZ file
        bond_topology: dict with 'n_beads_per_mol' and 'bonds'
        n_sample: Number of frames to sample for estimation
        temperature: Temperature in K for kT estimation
        sigma_scale: Scale factor for sigma (< 1 to avoid overlap)
        repulsive_cutoff: Cutoff for repulsive interaction (Å)
        subsample: Subsample every N frames from the NPZ

    Returns:
        Config dict suitable for HarmonicRepulsivePrior
    """
    kB = 8.617333262e-5  # eV/K
    kT = kB * temperature

    data = np.load(npz_path, allow_pickle=True)
    positions = data['positions']  # (n_frames, n_atoms, 3)
    types = data['types']          # (n_atoms,)
    cells = data['cells']          # (n_frames, 3, 3)

    n_frames_total = positions.shape[0]
    n_atoms = positions.shape[1]

    # Subsample if needed
    if subsample > 1:
        indices = np.arange(0, n_frames_total, subsample)
    else:
        indices = np.arange(n_frames_total)

    # Sample frames
    if len(indices) > n_sample:
        rng = np.random.default_rng(42)
        sample_idx = rng.choice(indices, size=n_sample, replace=False)
        sample_idx.sort()
    else:
        sample_idx = indices

    n_beads_per_mol = bond_topology['n_beads_per_mol']
    bonds_local = bond_topology['bonds']
    n_mol = n_atoms // n_beads_per_mol

    print(f"\nEstimating prior parameters from {len(sample_idx)} frames...")
    print(f"  System: {n_atoms} beads, {n_mol} molecules, {n_beads_per_mol} beads/mol")
    print(f"  Bonds per molecule: {len(bonds_local)}")

    # --- 1) Bond parameter estimation ---
    # Collect bond distances per bond-type
    bond_type_distances: Dict[str, List[float]] = {}
    for b_idx, (bi, bj) in enumerate(bonds_local):
        ti, tj = int(types[bi]), int(types[bj])
        key = f"{min(ti,tj)}-{max(ti,tj)}"
        if key not in bond_type_distances:
            bond_type_distances[key] = []

    def _apply_pbc_vec(r_vec, cell):
        if cell is None:
            return r_vec
        if cell.ndim == 2 and np.allclose(cell - np.diag(np.diag(cell)), 0):
            box = np.diag(cell)
            return r_vec - box * np.round(r_vec / box)
        inv_cell = np.linalg.inv(cell)
        s = r_vec @ inv_cell
        s = s - np.round(s)
        return s @ cell

    for fi in sample_idx:
        pos = positions[fi]
        cell = cells[fi] if cells.ndim == 3 else None

        for m in range(n_mol):
            offset = m * n_beads_per_mol
            for bi, bj in bonds_local:
                ai, aj = offset + bi, offset + bj
                ti, tj = int(types[ai]), int(types[aj])
                r_vec = pos[aj] - pos[ai]
                r_vec = _apply_pbc_vec(r_vec, cell)
                r = np.linalg.norm(r_vec)
                key = f"{min(ti,tj)}-{max(ti,tj)}"
                bond_type_distances[key].append(r)

    # Compute r_eq, k per bond type
    # Use robust statistics: filter outliers from PBC wrapping artifacts
    # CG bond distances should be < max_bond_dist (beads within same molecule)
    max_bond_dist = 15.0  # Å, generous upper bound for CG bonds
    bond_params: Dict[str, Dict[str, float]] = {}
    for key, dists in bond_type_distances.items():
        dists_arr = np.array(dists)
        # Filter outliers: distances > max_bond_dist are PBC artifacts
        valid_mask = dists_arr < max_bond_dist
        dists_clean = dists_arr[valid_mask]
        n_outliers = len(dists_arr) - len(dists_clean)

        if len(dists_clean) < 10:
            # Not enough clean data; use median of all
            r_eq = float(np.median(dists_arr))
            k = 1.0  # default spring constant
        else:
            r_eq = float(np.mean(dists_clean))
            var_r = float(np.var(dists_clean))
            k = kT / var_r if var_r > 1e-6 else 100.0

        bond_params[key] = {'r_eq': r_eq, 'k': k}
        clean_std = np.std(dists_clean) if len(dists_clean) > 0 else 0.0
        outlier_pct = n_outliers / len(dists_arr) * 100 if len(dists_arr) > 0 else 0.0
        print(f"  Bond {key}: r_eq={r_eq:.3f} Å, k={k:.3f} eV/Å², "
              f"std={clean_std:.4f} Å"
              f"{f', outliers={n_outliers} ({outlier_pct:.0f}%)' if n_outliers > 0 else ''}")

    # Map back to per-bond lists (in order of bonds_local)
    k_per_bond = []
    r_eq_per_bond = []
    for bi, bj in bonds_local:
        ti, tj = int(types[bi]), int(types[bj])
        key = f"{min(ti,tj)}-{max(ti,tj)}"
        k_per_bond.append(bond_params[key]['k'])
        r_eq_per_bond.append(bond_params[key]['r_eq'])

    # --- 2) Angle parameter estimation ---
    angles_local = generate_angles_from_bonds(bonds_local, n_beads_per_mol)
    print(f"\n  Angle topology: {len(angles_local)} angles from {len(bonds_local)} bonds")

    # Collect angles per angle-type
    angle_type_values: Dict[str, List[float]] = {}
    for a_idx, (ai, aj, ak) in enumerate(angles_local):
        ti, tj, tk = int(types[ai]), int(types[aj]), int(types[ak])
        # Key includes central atom type; endpoints sorted
        key = f"{min(ti,tk)}-{tj}-{max(ti,tk)}"
        if key not in angle_type_values:
            angle_type_values[key] = []

    for fi in sample_idx:
        pos = positions[fi]
        cell = cells[fi] if cells.ndim == 3 else None

        for m in range(n_mol):
            offset = m * n_beads_per_mol
            for ai, aj, ak in angles_local:
                gi, gj, gk = offset + ai, offset + aj, offset + ak
                r_ij = pos[gi] - pos[gj]
                r_kj = pos[gk] - pos[gj]
                r_ij = _apply_pbc_vec(r_ij, cell)
                r_kj = _apply_pbc_vec(r_kj, cell)
                d_ij = np.linalg.norm(r_ij)
                d_kj = np.linalg.norm(r_kj)
                if d_ij < 1e-10 or d_kj < 1e-10:
                    continue
                cos_theta = np.dot(r_ij, r_kj) / (d_ij * d_kj)
                cos_theta = np.clip(cos_theta, -1.0, 1.0)
                theta = np.arccos(cos_theta)

                ti, tj, tk = int(types[gi]), int(types[gj]), int(types[gk])
                key = f"{min(ti,tk)}-{tj}-{max(ti,tk)}"
                angle_type_values[key].append(theta)

    # Compute theta_eq, k_theta per angle type (equipartition: k = kT / var(θ))
    max_angle_dist = np.pi  # radian, no outlier filtering needed for angles
    angle_params: Dict[str, Dict[str, float]] = {}
    for key, vals in angle_type_values.items():
        vals_arr = np.array(vals)
        # Filter extreme outliers (e.g., PBC artifacts giving near-0 or near-pi)
        median_val = np.median(vals_arr)
        valid_mask = np.abs(vals_arr - median_val) < 1.0  # within ~57° of median
        vals_clean = vals_arr[valid_mask] if np.sum(valid_mask) > 10 else vals_arr

        theta_eq = float(np.mean(vals_clean))
        var_theta = float(np.var(vals_clean))
        k_theta = kT / var_theta if var_theta > 1e-8 else 100.0

        angle_params[key] = {'theta_eq': theta_eq, 'k': k_theta}
        theta_eq_deg = np.degrees(theta_eq)
        std_deg = np.degrees(np.std(vals_clean))
        print(f"  Angle {key}: θ_eq={theta_eq_deg:.1f}°, k={k_theta:.4f} eV/rad², "
              f"std={std_deg:.1f}°")

    # Map back to per-angle lists (in order of angles_local)
    k_per_angle = []
    theta_eq_per_angle = []
    for ai, aj, ak in angles_local:
        ti, tj, tk = int(types[ai]), int(types[aj]), int(types[ak])
        key = f"{min(ti,tk)}-{tj}-{max(ti,tk)}"
        k_per_angle.append(angle_params[key]['k'])
        theta_eq_per_angle.append(angle_params[key]['theta_eq'])

    # --- 3) Non-bonded sigma estimation ---
    print(f"\n  Estimating non-bonded sigma (from {len(sample_idx)} frames)...")

    # Build excluded set: 1-2, 1-3, 1-4 (local indices)
    excluded_local = build_exclusion_list(
        bonds_local, n_beads_per_mol,
        exclude_13=True, exclude_14=True,
    )
    print(f"  Excluded pairs per molecule: {len(excluded_local)} "
          f"(1-2: {len(bonds_local)}, +1-3/1-4)")

    # Sample minimum distances per type-pair using ASE neighbor list
    try:
        from ase import Atoms
        from ase.neighborlist import neighbor_list
    except ImportError:
        raise ImportError("ASE is required for estimate_prior_params_from_npz")

    # Collect distances per type-pair across frames
    # Use 5th percentile (not minimum) to be robust to PBC wrapping artifacts
    dist_per_type: Dict[str, List[float]] = {}
    n_sigma_frames = min(30, len(sample_idx))
    sigma_idx = sample_idx[:n_sigma_frames]

    # Minimum physical distance for CG beads (filter PBC artifacts)
    min_physical_dist = 1.0  # Å — beads represent atom groups, can't be < 1 Å

    for fi in sigma_idx:
        pos = positions[fi]
        cell = cells[fi] if cells.ndim == 3 else None

        if cell is not None:
            atoms = Atoms(numbers=[1] * n_atoms, positions=pos, cell=cell, pbc=True)
        else:
            atoms = Atoms(numbers=[1] * n_atoms, positions=pos, pbc=False)

        idx_i, idx_j, dists = neighbor_list('ijd', atoms, repulsive_cutoff)

        # Filter i < j and valid distances
        mask = (idx_i < idx_j) & (dists > min_physical_dist)
        idx_i = idx_i[mask]
        idx_j = idx_j[mask]
        dists = dists[mask]

        # Only inter-molecular pairs (exclude ALL intra-molecular)
        for k_idx in range(len(idx_i)):
            ai, aj = int(idx_i[k_idx]), int(idx_j[k_idx])
            mol_i = ai // n_beads_per_mol
            mol_j = aj // n_beads_per_mol
            if mol_i == mol_j:
                continue  # skip intra-molecular pairs

            ti, tj = int(types[ai]), int(types[aj])
            key = f"{min(ti,tj)}-{max(ti,tj)}"
            if key not in dist_per_type:
                dist_per_type[key] = []
            dist_per_type[key].append(float(dists[k_idx]))

    # Compute sigma from 1st percentile of inter-molecular distance distribution
    # Use conservative scaling: sigma should be well below any observed distance
    # so that (sigma/r)^12 is small for most pairs.
    sigma_matrix = {}
    for key, dists_list in dist_per_type.items():
        dists_arr = np.array(dists_list)
        p1 = float(np.percentile(dists_arr, 1))
        sigma = sigma_scale * p1
        sigma_matrix[key] = sigma

    print(f"  Found {len(sigma_matrix)} non-bonded type-pairs (inter-molecular only)")
    sorted_keys = sorted(sigma_matrix.keys(), key=lambda x: sigma_matrix[x])
    for key in sorted_keys[:5]:
        p1 = sigma_matrix[key] / sigma_scale
        print(f"    sigma[{key}] = {sigma_matrix[key]:.3f} Å "
              f"(p1_dist = {p1:.3f} Å)")
    if len(sorted_keys) > 5:
        print(f"    ... and {len(sorted_keys)-5} more pairs")
    if sorted_keys:
        median_sigma = np.median([sigma_matrix[k] for k in sorted_keys])
        print(f"  Median sigma: {median_sigma:.3f} Å")

    # Build full config
    config = {
        'type': 'harmonic_repulsive',
        'bond_topology': bond_topology,
        'harmonic': {
            'k_per_bond': k_per_bond,
            'r_eq_per_bond': r_eq_per_bond,
        },
        'angle': {
            'angle_topology': angles_local,
            'k_per_angle': k_per_angle,
            'theta_eq_per_angle': theta_eq_per_angle,
        },
        'repulsive': {
            'sigma_matrix': sigma_matrix,
            'epsilon': 0.001,
            'cutoff': repulsive_cutoff,
        },
    }

    return config


# =============================================================================
# Utility Functions
# =============================================================================

def compute_delta_forces(
    total_forces: np.ndarray,
    positions: np.ndarray,
    prior: Union[Dict, PriorForceField],
    types: Optional[np.ndarray] = None,
    cell: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Compute delta forces: F_delta = F_total - F_prior

    Args:
        total_forces: Total forces from AA mapping, shape (n_atoms, 3)
        positions: CG positions, shape (n_atoms, 3)
        prior: Prior FF config dict or PriorForceField instance
        types: Bead types, shape (n_atoms,) - not used for universal priors
        cell: Unit cell, shape (3, 3) or None

    Returns:
        Delta forces, shape (n_atoms, 3)
    """
    if isinstance(prior, dict):
        prior_ff = PriorForceField.from_config(prior)
    else:
        prior_ff = prior

    prior_forces = prior_ff.compute_forces(positions, types, cell)
    delta_forces = total_forces - prior_forces

    return delta_forces


def print_available_priors():
    """Print available prior FF types."""
    print("\n" + "="*60)
    print("Available Prior Force Field Types")
    print("(All Universal - Atomic Number Based)")
    print("="*60)
    print("""
  zbl (ZBLPrior):
    Type: Universal short-range repulsion
    Description: Ziegler-Biersack-Littmark screened nuclear repulsion
    Input: atomic_numbers (list of atomic numbers), cutoff
    Reference: Ziegler et al., "The Stopping and Range of Ions in Solids" (1985)

  d2 (D2Prior):
    Type: Universal long-range dispersion
    Description: DFT-D2 dispersion correction (Grimme)
    Input: atomic_numbers, cutoff, s_6 (scaling), d (damping steepness)
    Reference: Grimme, J. Comput. Chem. 27 (2006) 1787-1799

  universal (UniversalPrior):
    Type: Combined ZBL + D2
    Description: Short-range repulsion + long-range dispersion
    Input: atomic_numbers, zbl_cutoff, d2_cutoff, zbl_weight, d2_weight

  none (NoPrior):
    Type: No prior (zero forces)
    Description: Equivalent to direct learning (F_delta = F_total)

Usage Examples:
    # ZBL only (short-range repulsion)
    prior = PriorForceField.from_config({
        'type': 'zbl',
        'atomic_numbers': [8, 1, 1],  # H2O
        'cutoff': 5.0
    })

    # D2 only (long-range dispersion)
    prior = PriorForceField.from_config({
        'type': 'd2',
        'atomic_numbers': [6, 1, 1, 1, 1],  # CH4
        'cutoff': 20.0,
        's_6': 1.0
    })

    # Universal (ZBL + D2 combined) - Recommended
    prior = PriorForceField.from_config({
        'type': 'universal',
        'atomic_numbers': [6, 1, 1, 1, 1],  # CH4
        'zbl_cutoff': 5.0,
        'd2_cutoff': 20.0
    })
""")
    print("="*60)


# =============================================================================
# Test
# =============================================================================

if __name__ == '__main__':
    print_available_priors()

    # Test ZBL Prior
    print("\n\n" + "="*60)
    print("Testing ZBL Prior (Universal, atomic number based)...")
    print("="*60)

    # Methane: C (Z=6) + 4 H (Z=1)
    zbl_prior = ZBLPrior({
        'atomic_numbers': [6, 1, 1, 1, 1],
        'cutoff': 5.0
    })

    # Tetrahedral methane-like positions (Å)
    positions_ch4 = np.array([
        [0.0, 0.0, 0.0],      # C
        [1.09, 0.0, 0.0],     # H
        [-0.36, 1.03, 0.0],   # H
        [-0.36, -0.51, 0.89], # H
        [-0.36, -0.51, -0.89] # H
    ])

    forces_zbl = zbl_prior.compute_forces(positions_ch4)
    energy_zbl = zbl_prior.compute_energy(positions_ch4)

    print(f"System: CH4 (Methane)")
    print(f"Atomic numbers: C=6, H=1,1,1,1")
    print(f"Forces:\n{forces_zbl}")
    print(f"Energy: {energy_zbl:.6f} eV")

    # Test D2 Prior
    print("\n\n" + "="*60)
    print("Testing D2 Prior (Universal, atomic number based)...")
    print("="*60)

    d2_prior = D2Prior({
        'atomic_numbers': [6, 1, 1, 1, 1],
        'cutoff': 20.0
    })

    forces_d2 = d2_prior.compute_forces(positions_ch4)
    energy_d2 = d2_prior.compute_energy(positions_ch4)

    print(f"System: CH4 (Methane)")
    print(f"Forces:\n{forces_d2}")
    print(f"Energy: {energy_d2:.6f} eV")

    # Test Universal Prior (ZBL + D2)
    print("\n\n" + "="*60)
    print("Testing Universal Prior (ZBL + D2 combined)...")
    print("="*60)

    universal_prior = UniversalPrior({
        'atomic_numbers': [6, 1, 1, 1, 1],
        'zbl_cutoff': 5.0,
        'd2_cutoff': 20.0
    })

    forces_universal = universal_prior.compute_forces(positions_ch4)
    energy_universal = universal_prior.compute_energy(positions_ch4)

    print(f"System: CH4 (Methane)")
    print(f"Forces:\n{forces_universal}")
    print(f"Energy: {energy_universal:.6f} eV (ZBL: {energy_zbl:.6f} + D2: {energy_d2:.6f})")

    # Test with water molecules (O=8, H=1)
    print("\n\n" + "="*60)
    print("Testing Universal Prior with Water (H2O)...")
    print("="*60)

    water_prior = UniversalPrior({
        'atomic_numbers': [8, 1, 1],  # O, H, H
        'zbl_cutoff': 5.0,
        'd2_cutoff': 20.0
    })

    # Water geometry
    positions_h2o = np.array([
        [0.0, 0.0, 0.0],      # O
        [0.96, 0.0, 0.0],     # H
        [-0.24, 0.93, 0.0],   # H
    ])

    forces_h2o = water_prior.compute_forces(positions_h2o)
    energy_h2o = water_prior.compute_energy(positions_h2o)

    print(f"System: H2O (Water)")
    print(f"Atomic numbers: O=8, H=1, H=1")
    print(f"Forces:\n{forces_h2o}")
    print(f"Energy: {energy_h2o:.6f} eV")

    # Test via from_config
    print("\n\n" + "="*60)
    print("Testing PriorForceField.from_config()...")
    print("="*60)

    prior = PriorForceField.from_config({
        'type': 'universal',
        'atomic_numbers': [7, 1, 1, 1],  # NH3
        'zbl_cutoff': 5.0,
        'd2_cutoff': 20.0
    })

    positions_nh3 = np.array([
        [0.0, 0.0, 0.0],      # N
        [1.01, 0.0, 0.0],     # H
        [-0.34, 0.95, 0.0],   # H
        [-0.34, -0.48, 0.83], # H
    ])

    forces_nh3 = prior.compute_forces(positions_nh3)
    energy_nh3 = prior.compute_energy(positions_nh3)

    print(f"System: NH3 (Ammonia)")
    print(f"Atomic numbers: N=7, H=1, H=1, H=1")
    print(f"Energy: {energy_nh3:.6f} eV")

    print("\n" + "="*60)
    print("All Prior FF tests passed!")
    print("="*60)
