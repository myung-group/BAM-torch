"""
LAMMPS wrapper for Delta Learning BAM models.

This wrapper includes prior force calculation (ZBL + D2) so that:
    F_total = F_prior + F_delta (model output)

The prior is computed analytically at each step, ensuring physically
reasonable behavior even outside the training data distribution.
"""

from typing import Dict, List, Optional
import math

import torch
from e3nn.util.jit import compile_mode

from bam_torch.utils.scatter import scatter_sum


# =============================================================================
# PyTorch Prior Force Field Implementation (JIT-compatible)
# =============================================================================

@compile_mode("script")
class ZBLPriorTorch(torch.nn.Module):
    """
    Ziegler-Biersack-Littmark (ZBL) screened nuclear repulsion in PyTorch.

    V(r) = k_e * Z_i * Z_j / r * phi(r/a) * fc(r)
    """

    def __init__(self, atomic_numbers: List[int], cutoff: float = 5.0):
        super().__init__()
        self.register_buffer("atomic_numbers", torch.tensor(atomic_numbers, dtype=torch.float32))
        self.register_buffer("cutoff", torch.tensor(cutoff, dtype=torch.float32))
        # Physical constants (must be buffers for JIT compatibility)
        self.register_buffer("A_0", torch.tensor(0.529177210903, dtype=torch.float32))  # Bohr radius in Å
        self.register_buffer("K_E", torch.tensor(14.3996, dtype=torch.float32))  # Coulomb constant in eV*Å

    def _screening_function(self, x: torch.Tensor) -> torch.Tensor:
        """ZBL universal screening function phi(x)."""
        return (0.1818 * torch.exp(-3.2 * x) +
                0.5099 * torch.exp(-0.9423 * x) +
                0.2802 * torch.exp(-0.4029 * x) +
                0.02817 * torch.exp(-0.2016 * x))

    def _screening_derivative(self, x: torch.Tensor) -> torch.Tensor:
        """Derivative of ZBL screening function d(phi)/dx."""
        return (-3.2 * 0.1818 * torch.exp(-3.2 * x) +
                -0.9423 * 0.5099 * torch.exp(-0.9423 * x) +
                -0.4029 * 0.2802 * torch.exp(-0.4029 * x) +
                -0.2016 * 0.02817 * torch.exp(-0.2016 * x))

    def _cosine_cutoff(self, r: torch.Tensor) -> torch.Tensor:
        """Smooth cosine cutoff function."""
        pi = 3.141592653589793
        return torch.where(
            r < self.cutoff,
            0.5 * (1.0 + torch.cos(pi * r / self.cutoff)),
            torch.zeros_like(r)
        )

    def _cosine_cutoff_derivative(self, r: torch.Tensor) -> torch.Tensor:
        """Derivative of cosine cutoff."""
        pi = 3.141592653589793
        return torch.where(
            r < self.cutoff,
            -0.5 * pi / self.cutoff * torch.sin(pi * r / self.cutoff),
            torch.zeros_like(r)
        )

    def forward(
        self,
        positions: torch.Tensor,
        edge_index: torch.Tensor,
        shifts: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute ZBL forces using edge list.

        Args:
            positions: (n_atoms, 3)
            edge_index: (2, n_edges) - [src, dst] pairs
            shifts: (n_edges, 3) - periodic shifts

        Returns:
            forces: (n_atoms, 3)
        """
        n_atoms = positions.shape[0]
        device = positions.device
        forces = torch.zeros((n_atoms, 3), device=device, dtype=positions.dtype)

        if edge_index.shape[1] == 0:
            return forces

        src, dst = edge_index[0], edge_index[1]

        # Compute displacement vectors
        r_vec = positions[dst] - positions[src] + shifts
        r = torch.norm(r_vec, dim=1)

        # Mask for valid distances
        valid = (r > 1e-10) & (r < self.cutoff)

        if not valid.any():
            return forces

        # Get atomic numbers
        Z_src = self.atomic_numbers[src[valid]]
        Z_dst = self.atomic_numbers[dst[valid]]

        r_valid = r[valid]
        r_vec_valid = r_vec[valid]

        # Screening length
        a = 0.8854 * self.A_0 / (Z_src.pow(0.23) + Z_dst.pow(0.23))
        x = r_valid / a

        # Screening function and derivative
        phi = self._screening_function(x)
        dphi_dx = self._screening_derivative(x)

        # Cutoff and derivative
        fc = self._cosine_cutoff(r_valid)
        dfc_dr = self._cosine_cutoff_derivative(r_valid)

        # Prefactor
        prefactor = self.K_E * Z_src * Z_dst

        # dV/dr
        dV_dr = prefactor * (
            -phi / (r_valid * r_valid) * fc +
            (1.0 / r_valid) * (dphi_dx / a) * fc +
            (phi / r_valid) * dfc_dr
        )

        # Force vectors: F = -dV/dr * r_hat
        f_vec = -dV_dr.unsqueeze(1) * (r_vec_valid / r_valid.unsqueeze(1))

        # Accumulate forces (Newton's third law)
        src_valid = src[valid]
        dst_valid = dst[valid]

        # Use scatter_add for accumulation
        forces.index_add_(0, src_valid, -f_vec)
        forces.index_add_(0, dst_valid, f_vec)

        return forces


@compile_mode("script")
class D2PriorTorch(torch.nn.Module):
    """
    DFT-D2 dispersion correction in PyTorch.

    V_disp = -s_6 * sum_{i<j} C_6^{ij} / r_ij^6 * f_damp(r_ij)
    """

    def __init__(
        self,
        atomic_numbers: List[int],
        cutoff: float = 20.0,
        s_6: float = 1.0,
        d: float = 20.0
    ):
        super().__init__()
        self.register_buffer("cutoff", torch.tensor(cutoff, dtype=torch.float32))
        self.s_6 = s_6
        self.d = d

        # Pre-compute C6 and R0 for each atom
        # C6 in eV*Å^6, R0 in Å
        c6_list = []
        r0_list = []
        for Z in atomic_numbers:
            c6, r0 = self._get_c6_r0(Z)
            c6_list.append(c6)
            r0_list.append(r0)

        self.register_buffer("c6_per_atom", torch.tensor(c6_list, dtype=torch.float32))
        self.register_buffer("r0_per_atom", torch.tensor(r0_list, dtype=torch.float32))

    def _get_c6_r0(self, Z: int) -> tuple:
        """Get C6 (eV*Å^6) and R0 (Å) for atomic number Z."""
        # C6 in J/mol*nm^6, R0 in nm (from Grimme 2006)
        C6_R0_TABLE = {
            1:  (0.14, 0.1001),   # H
            2:  (0.08, 0.1012),   # He
            6:  (1.75, 0.1452),   # C
            7:  (1.23, 0.1397),   # N
            8:  (0.70, 0.1342),   # O
            9:  (0.75, 0.1287),   # F
            14: (9.23, 0.1716),   # Si
            15: (7.84, 0.1705),   # P
            16: (5.57, 0.1683),   # S
            17: (5.07, 0.1639),   # Cl
        }

        if Z not in C6_R0_TABLE:
            # Default to carbon-like values for unknown elements
            c6_orig, r0_nm = 1.75, 0.1452
        else:
            c6_orig, r0_nm = C6_R0_TABLE[Z]

        # Unit conversion: J/mol*nm^6 -> eV*Å^6
        c6_conversion = 1.0364e-5 * 1e6  # = 10.364
        c6_eV_A6 = c6_orig * c6_conversion
        r0_A = r0_nm * 10  # nm -> Å

        return c6_eV_A6, r0_A

    def _damping_function(self, r: torch.Tensor, R_r: torch.Tensor) -> torch.Tensor:
        """Fermi damping function."""
        x = -self.d * (r / R_r - 1.0)
        # Clamp to avoid overflow
        x = torch.clamp(x, -50.0, 50.0)
        return 1.0 / (1.0 + torch.exp(x))

    def _damping_derivative(self, r: torch.Tensor, R_r: torch.Tensor) -> torch.Tensor:
        """Derivative of damping function with respect to r."""
        x = -self.d * (r / R_r - 1.0)
        x = torch.clamp(x, -50.0, 50.0)
        exp_x = torch.exp(x)
        f = 1.0 / (1.0 + exp_x)
        # df/dr = f^2 * exp(x) * d/R_r
        return f * f * exp_x * self.d / R_r

    def forward(
        self,
        positions: torch.Tensor,
        edge_index: torch.Tensor,
        shifts: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute D2 dispersion forces using edge list.

        Args:
            positions: (n_atoms, 3)
            edge_index: (2, n_edges) - [src, dst] pairs
            shifts: (n_edges, 3) - periodic shifts

        Returns:
            forces: (n_atoms, 3)
        """
        n_atoms = positions.shape[0]
        device = positions.device
        forces = torch.zeros((n_atoms, 3), device=device, dtype=positions.dtype)

        if edge_index.shape[1] == 0:
            return forces

        src, dst = edge_index[0], edge_index[1]

        # Compute displacement vectors
        r_vec = positions[dst] - positions[src] + shifts
        r = torch.norm(r_vec, dim=1)

        # Mask for valid distances
        valid = (r > 1e-10) & (r < self.cutoff)

        if not valid.any():
            return forces

        r_valid = r[valid]
        r_vec_valid = r_vec[valid]
        src_valid = src[valid]
        dst_valid = dst[valid]

        # Get C6 and R0 for pairs
        c6_src = self.c6_per_atom[src_valid]
        c6_dst = self.c6_per_atom[dst_valid]
        r0_src = self.r0_per_atom[src_valid]
        r0_dst = self.r0_per_atom[dst_valid]

        # Combination rules
        C_6 = torch.sqrt(c6_src * c6_dst)
        R_r = r0_src + r0_dst

        # Damping function and derivative
        f_damp = self._damping_function(r_valid, R_r)
        df_damp = self._damping_derivative(r_valid, R_r)

        # V = -s_6 * C_6 / r^6 * f_damp
        # dV/dr = s_6 * C_6 * [6*f_damp/r^7 - df_damp/r^6]
        r6 = r_valid.pow(6)
        r7 = r_valid * r6

        dV_dr = self.s_6 * C_6 * (6.0 * f_damp / r7 - df_damp / r6)

        # Force vectors
        f_vec = -dV_dr.unsqueeze(1) * (r_vec_valid / r_valid.unsqueeze(1))

        # Accumulate forces
        forces.index_add_(0, src_valid, -f_vec)
        forces.index_add_(0, dst_valid, f_vec)

        return forces


@compile_mode("script")
class UniversalPriorTorch(torch.nn.Module):
    """Combined ZBL + D2 prior in PyTorch."""

    def __init__(
        self,
        atomic_numbers: List[int],
        zbl_cutoff: float = 5.0,
        d2_cutoff: float = 20.0,
        s_6: float = 1.0,
        d: float = 20.0
    ):
        super().__init__()
        self.zbl = ZBLPriorTorch(atomic_numbers, cutoff=zbl_cutoff)
        self.d2 = D2PriorTorch(atomic_numbers, cutoff=d2_cutoff, s_6=s_6, d=d)
        self.register_buffer("max_cutoff", torch.tensor(max(zbl_cutoff, d2_cutoff)))

    def forward(
        self,
        positions: torch.Tensor,
        edge_index: torch.Tensor,
        shifts: torch.Tensor
    ) -> torch.Tensor:
        """Compute combined ZBL + D2 forces."""
        f_zbl = self.zbl(positions, edge_index, shifts)
        f_d2 = self.d2(positions, edge_index, shifts)
        return f_zbl + f_d2


# =============================================================================
# CG-Specific Priors: Harmonic Bond + Repulsive LJ (JIT-compatible)
# =============================================================================

@compile_mode("script")
class HarmonicBondPriorTorch(torch.nn.Module):
    """
    Harmonic bond prior in PyTorch (JIT-compilable).

    V(r) = 0.5 * k * (r - r_eq)^2 per bonded pair.
    Bond pairs are pre-expanded to the full system at init time.
    """

    def __init__(
        self,
        bond_pairs: torch.Tensor,
        k_per_bond: torch.Tensor,
        r_eq_per_bond: torch.Tensor,
    ):
        """
        Args:
            bond_pairs: (n_total_bonds, 2) global atom indices
            k_per_bond: (n_total_bonds,) spring constants in eV/Å²
            r_eq_per_bond: (n_total_bonds,) equilibrium distances in Å
        """
        super().__init__()
        self.register_buffer("bond_pairs", bond_pairs.long())
        self.register_buffer("k_per_bond", k_per_bond.float())
        self.register_buffer("r_eq_per_bond", r_eq_per_bond.float())

    def forward(
        self,
        positions: torch.Tensor,
        edge_index: torch.Tensor,
        shifts: torch.Tensor,
        cell: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute harmonic bond forces.

        Args:
            positions: (n_atoms, 3)
            edge_index: unused (bonds are pre-defined)
            shifts: unused
            cell: (3, 3) unit cell for PBC

        Returns:
            forces: (n_atoms, 3)
        """
        n_atoms = positions.shape[0]
        device = positions.device
        forces = torch.zeros((n_atoms, 3), device=device, dtype=positions.dtype)

        if self.bond_pairs.shape[0] == 0:
            return forces

        idx_i = self.bond_pairs[:, 0]
        idx_j = self.bond_pairs[:, 1]

        r_vec = positions[idx_j] - positions[idx_i]

        # Minimum image PBC
        if cell.numel() > 0:
            inv_cell = torch.linalg.inv(cell)
            s = r_vec @ inv_cell
            s = s - torch.round(s)
            r_vec = s @ cell

        r = torch.norm(r_vec, dim=1)

        # F = -k*(r - r_eq) * r_hat
        valid = r > 1e-10
        f_mag = torch.zeros_like(r)
        f_mag[valid] = -self.k_per_bond[valid] * (r[valid] - self.r_eq_per_bond[valid])

        f_vec = torch.zeros_like(r_vec)
        f_vec[valid] = f_mag[valid].unsqueeze(1) * (r_vec[valid] / r[valid].unsqueeze(1))

        # Newton's third law
        forces.index_add_(0, idx_i, -f_vec)
        forces.index_add_(0, idx_j, f_vec)

        return forces


@compile_mode("script")
class RepulsiveLJPriorTorch(torch.nn.Module):
    """
    Purely repulsive (sigma/r)^12 prior in PyTorch (JIT-compilable).

    V(r) = 4*epsilon*(sigma_ij/r)^12 * fc(r) for non-bonded pairs.
    Bonded pairs are excluded using a pre-built exclusion set.
    """

    def __init__(
        self,
        sigma_matrix: torch.Tensor,
        species_to_type: torch.Tensor,
        epsilon: float = 0.001,
        cutoff: float = 10.0,
        max_force: float = 1.0,
        excluded_bonds: torch.Tensor = torch.empty(0, 2, dtype=torch.long),
    ):
        """
        Args:
            sigma_matrix: (n_types, n_types) sigma values
            species_to_type: (n_atoms,) maps atom index → type index
            epsilon: energy scale (eV)
            cutoff: cutoff distance (Å)
            max_force: Maximum force magnitude per pair (eV/Å)
            excluded_bonds: (n_bonds, 2) bonded pair indices to exclude
        """
        super().__init__()
        self.register_buffer("sigma_matrix", sigma_matrix.float())
        self.register_buffer("species_to_type", species_to_type.long())
        self.register_buffer("cutoff_val", torch.tensor(cutoff, dtype=torch.float32))
        self.register_buffer("max_force_val", torch.tensor(max_force, dtype=torch.float32))
        self.epsilon = epsilon

        # Build exclusion hash: (min(i,j) * N + max(i,j)) for fast lookup
        if excluded_bonds.numel() > 0:
            n_max = species_to_type.shape[0]
            min_ij = torch.minimum(excluded_bonds[:, 0], excluded_bonds[:, 1])
            max_ij = torch.maximum(excluded_bonds[:, 0], excluded_bonds[:, 1])
            excl_hash = min_ij * n_max + max_ij
            self.register_buffer("excl_hash", excl_hash)
            self.register_buffer("n_max", torch.tensor(n_max, dtype=torch.long))
        else:
            self.register_buffer("excl_hash", torch.empty(0, dtype=torch.long))
            self.register_buffer("n_max", torch.tensor(0, dtype=torch.long))

    def _is_excluded(self, src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
        """Check if pairs are in exclusion list."""
        if self.excl_hash.numel() == 0:
            return torch.zeros(src.shape[0], dtype=torch.bool, device=src.device)

        n = self.n_max.item()
        min_ij = torch.minimum(src, dst)
        max_ij = torch.maximum(src, dst)
        pair_hash = min_ij * n + max_ij

        # Check membership via broadcasting
        excluded = (pair_hash.unsqueeze(1) == self.excl_hash.unsqueeze(0)).any(dim=1)
        return excluded

    def _cosine_cutoff(self, r: torch.Tensor) -> torch.Tensor:
        pi = 3.141592653589793
        return torch.where(
            r < self.cutoff_val,
            0.5 * (1.0 + torch.cos(pi * r / self.cutoff_val)),
            torch.zeros_like(r)
        )

    def _cosine_cutoff_derivative(self, r: torch.Tensor) -> torch.Tensor:
        pi = 3.141592653589793
        return torch.where(
            r < self.cutoff_val,
            -0.5 * pi / self.cutoff_val * torch.sin(pi * r / self.cutoff_val),
            torch.zeros_like(r)
        )

    def forward(
        self,
        positions: torch.Tensor,
        edge_index: torch.Tensor,
        shifts: torch.Tensor,
        cell: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute repulsive LJ forces using edge list from LAMMPS.
        """
        n_atoms = positions.shape[0]
        device = positions.device
        forces = torch.zeros((n_atoms, 3), device=device, dtype=positions.dtype)

        if edge_index.shape[1] == 0:
            return forces

        src, dst = edge_index[0], edge_index[1]

        # Only i < j to avoid double counting
        mask_ij = src < dst
        src = src[mask_ij]
        dst = dst[mask_ij]
        shifts_filt = shifts[mask_ij]

        if src.shape[0] == 0:
            return forces

        # Exclude bonded pairs
        if self.excl_hash.numel() > 0:
            not_excluded = ~self._is_excluded(src, dst)
            src = src[not_excluded]
            dst = dst[not_excluded]
            shifts_filt = shifts_filt[not_excluded]

        if src.shape[0] == 0:
            return forces

        # Compute distances
        r_vec = positions[dst] - positions[src] + shifts_filt
        r = torch.norm(r_vec, dim=1)

        valid = (r > 1e-10) & (r < self.cutoff_val)
        if not valid.any():
            return forces

        src_v = src[valid]
        dst_v = dst[valid]
        r_v = r[valid]
        r_vec_v = r_vec[valid]

        # Get sigma for each pair
        ti = self.species_to_type[src_v]
        tj = self.species_to_type[dst_v]
        sigma = self.sigma_matrix[ti, tj]

        # V(r) = 4*eps*(sigma/r)^12 * fc(r)
        sr = sigma / r_v
        sr12 = sr.pow(12)

        fc = self._cosine_cutoff(r_v)
        dfc_dr = self._cosine_cutoff_derivative(r_v)

        # dV/dr = 4*eps * [-12*sr^12/r * fc + sr^12 * dfc/dr]
        dV_dr = 4.0 * self.epsilon * (-12.0 * sr12 / r_v * fc + sr12 * dfc_dr)

        # Cap force magnitude per pair
        dV_dr = torch.clamp(dV_dr, -self.max_force_val, self.max_force_val)

        # Force
        r_hat = r_vec_v / r_v.unsqueeze(1)
        f_vec = -dV_dr.unsqueeze(1) * r_hat

        # Newton's third law
        forces.index_add_(0, src_v, -f_vec)
        forces.index_add_(0, dst_v, f_vec)

        return forces


@compile_mode("script")
class HarmonicAnglePriorTorch(torch.nn.Module):
    """
    Harmonic angle prior in PyTorch (JIT-compilable).

    V(θ) = 0.5 * k_θ * (θ - θ_eq)² for each angle triplet (i, j, k).
    Angle triplets are pre-expanded to the full system at init time.
    """

    def __init__(
        self,
        angle_triplets: torch.Tensor,
        k_per_angle: torch.Tensor,
        theta_eq_per_angle: torch.Tensor,
    ):
        """
        Args:
            angle_triplets: (n_total_angles, 3) global atom indices [i, j, k]
            k_per_angle: (n_total_angles,) spring constants in eV/rad²
            theta_eq_per_angle: (n_total_angles,) equilibrium angles in radians
        """
        super().__init__()
        self.register_buffer("angle_triplets", angle_triplets.long())
        self.register_buffer("k_per_angle", k_per_angle.float())
        self.register_buffer("theta_eq_per_angle", theta_eq_per_angle.float())

    def forward(
        self,
        positions: torch.Tensor,
        edge_index: torch.Tensor,
        shifts: torch.Tensor,
        cell: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute harmonic angle forces.

        Args:
            positions: (n_atoms, 3)
            edge_index: unused (angles are pre-defined)
            shifts: unused
            cell: (3, 3) unit cell for PBC

        Returns:
            forces: (n_atoms, 3)
        """
        n_atoms = positions.shape[0]
        device = positions.device
        forces = torch.zeros((n_atoms, 3), device=device, dtype=positions.dtype)

        if self.angle_triplets.shape[0] == 0:
            return forces

        idx_i = self.angle_triplets[:, 0]
        idx_j = self.angle_triplets[:, 1]
        idx_k = self.angle_triplets[:, 2]

        # Vectors from central atom j
        r_ij = positions[idx_i] - positions[idx_j]
        r_kj = positions[idx_k] - positions[idx_j]

        # Minimum image PBC
        if cell.numel() > 0:
            inv_cell = torch.linalg.inv(cell)
            s_ij = r_ij @ inv_cell
            s_ij = s_ij - torch.round(s_ij)
            r_ij = s_ij @ cell
            s_kj = r_kj @ inv_cell
            s_kj = s_kj - torch.round(s_kj)
            r_kj = s_kj @ cell

        d_ij = torch.norm(r_ij, dim=1)
        d_kj = torch.norm(r_kj, dim=1)

        # cos(θ) and θ
        dot = torch.sum(r_ij * r_kj, dim=1)
        cos_theta = dot / (d_ij * d_kj + 1e-10)
        cos_theta = torch.clamp(cos_theta, -1.0 + 1e-7, 1.0 - 1e-7)
        theta = torch.acos(cos_theta)
        sin_theta = torch.sin(theta)

        # Prefactor = k*(θ - θ_eq) / sin(θ)
        valid = sin_theta > 1e-6
        prefactor = torch.zeros_like(theta)
        prefactor[valid] = (
            self.k_per_angle[valid] * (theta[valid] - self.theta_eq_per_angle[valid])
            / sin_theta[valid]
        )

        # F_i = prefactor * [r_kj/(d_ij*d_kj) - cos_theta*r_ij/d_ij²]
        d_ij_dkj = (d_ij * d_kj).unsqueeze(1)
        d_ij_sq = (d_ij * d_ij).unsqueeze(1)
        d_kj_sq = (d_kj * d_kj).unsqueeze(1)
        cos_theta_col = cos_theta.unsqueeze(1)
        prefactor_col = prefactor.unsqueeze(1)

        f_i = prefactor_col * (r_kj / (d_ij_dkj + 1e-10) - cos_theta_col * r_ij / (d_ij_sq + 1e-10))
        f_k = prefactor_col * (r_ij / (d_ij_dkj + 1e-10) - cos_theta_col * r_kj / (d_kj_sq + 1e-10))
        f_j = -(f_i + f_k)

        forces.index_add_(0, idx_i, f_i)
        forces.index_add_(0, idx_j, f_j)
        forces.index_add_(0, idx_k, f_k)

        return forces


@compile_mode("script")
class HarmonicRepulsivePriorTorch(torch.nn.Module):
    """Combined Harmonic Bond + Harmonic Angle + Repulsive LJ prior (JIT-compilable)."""

    def __init__(
        self,
        harmonic: HarmonicBondPriorTorch,
        repulsive: RepulsiveLJPriorTorch,
        angle: Optional[HarmonicAnglePriorTorch] = None,
    ):
        super().__init__()
        self.harmonic = harmonic
        self.repulsive = repulsive
        self.angle = angle

    def forward(
        self,
        positions: torch.Tensor,
        edge_index: torch.Tensor,
        shifts: torch.Tensor,
        cell: torch.Tensor,
    ) -> torch.Tensor:
        f_harm = self.harmonic(positions, edge_index, shifts, cell)
        f_rep = self.repulsive(positions, edge_index, shifts, cell)
        f_total = f_harm + f_rep
        if self.angle is not None:
            f_total = f_total + self.angle(positions, edge_index, shifts, cell)
        return f_total


def build_harmonic_repulsive_prior_torch(
    prior_config: Dict,
    n_atoms: int,
    types_array: List[int],
) -> HarmonicRepulsivePriorTorch:
    """
    Factory to build HarmonicRepulsivePriorTorch from a prior config dict.

    Args:
        prior_config: Config dict from training (includes bond_topology, harmonic, repulsive)
        n_atoms: Total number of atoms in the system
        types_array: List of type indices for each atom

    Returns:
        HarmonicRepulsivePriorTorch module
    """
    topo = prior_config.get('bond_topology', {})
    n_beads_per_mol = topo.get('n_beads_per_mol', 21)
    bonds_local = topo.get('bonds', [])
    n_mol = n_atoms // n_beads_per_mol

    harmonic_cfg = prior_config.get('harmonic', {})
    repulsive_cfg = prior_config.get('repulsive', {})

    # --- Build bond pairs and parameters ---
    k_list = harmonic_cfg.get('k_per_bond', harmonic_cfg.get('k', []))
    r_eq_list = harmonic_cfg.get('r_eq_per_bond', harmonic_cfg.get('r_eq', []))

    if isinstance(k_list, list) and len(k_list) > 0:
        all_bonds = []
        all_k = []
        all_r_eq = []
        for m in range(n_mol):
            offset = m * n_beads_per_mol
            for b_idx, (bi, bj) in enumerate(bonds_local):
                all_bonds.append([offset + bi, offset + bj])
                all_k.append(k_list[b_idx])
                all_r_eq.append(r_eq_list[b_idx])

        bond_pairs = torch.tensor(all_bonds, dtype=torch.long)
        k_tensor = torch.tensor(all_k, dtype=torch.float32)
        r_eq_tensor = torch.tensor(all_r_eq, dtype=torch.float32)
    else:
        bond_pairs = torch.empty(0, 2, dtype=torch.long)
        k_tensor = torch.empty(0, dtype=torch.float32)
        r_eq_tensor = torch.empty(0, dtype=torch.float32)

    harmonic_prior = HarmonicBondPriorTorch(bond_pairs, k_tensor, r_eq_tensor)

    # --- Build exclusion list: ALL intra-molecular pairs ---
    # For CG systems, repulsive prior acts only on inter-molecular interactions.
    # Intra-molecular interactions are handled by the harmonic bond prior.
    all_excluded = []
    for m in range(n_mol):
        offset = m * n_beads_per_mol
        for i in range(n_beads_per_mol):
            for j in range(i + 1, n_beads_per_mol):
                all_excluded.append([offset + i, offset + j])

    if all_excluded:
        excluded_tensor = torch.tensor(all_excluded, dtype=torch.long)
    else:
        excluded_tensor = torch.empty(0, 2, dtype=torch.long)

    # --- Build sigma matrix ---
    sigma_dict = repulsive_cfg.get('sigma_matrix', repulsive_cfg.get('sigma', {}))
    epsilon = repulsive_cfg.get('epsilon', 0.001)
    cutoff = repulsive_cfg.get('cutoff', 10.0)

    n_types = max(types_array) + 1 if types_array else 1
    sigma_matrix = torch.zeros((n_types, n_types), dtype=torch.float32)

    # Default sigma
    if sigma_dict:
        default_sigma = sum(sigma_dict.values()) / len(sigma_dict)
        sigma_matrix.fill_(default_sigma)
        for key, val in sigma_dict.items():
            parts = key.split('-')
            ti, tj = int(parts[0]), int(parts[1])
            if ti < n_types and tj < n_types:
                sigma_matrix[ti, tj] = val
                sigma_matrix[tj, ti] = val
    else:
        sigma_matrix.fill_(3.0)

    species_to_type = torch.tensor(types_array, dtype=torch.long)

    max_force = repulsive_cfg.get('max_force', 1.0)

    repulsive_prior = RepulsiveLJPriorTorch(
        sigma_matrix=sigma_matrix,
        species_to_type=species_to_type,
        epsilon=epsilon,
        cutoff=cutoff,
        max_force=max_force,
        excluded_bonds=excluded_tensor,
    )

    # --- Build angle prior (optional) ---
    angle_cfg = prior_config.get('angle', None)
    angle_prior: Optional[HarmonicAnglePriorTorch] = None

    if angle_cfg is not None:
        angles_local = angle_cfg.get('angle_topology', None)
        if angles_local is None:
            # Auto-generate from bonds
            from bam_torch.utils.prior_ff import generate_angles_from_bonds
            angles_local = generate_angles_from_bonds(bonds_local, n_beads_per_mol)

        k_angle_list = angle_cfg.get('k_per_angle', angle_cfg.get('k', []))
        theta_eq_list = angle_cfg.get('theta_eq_per_angle', angle_cfg.get('theta_eq', []))

        if isinstance(k_angle_list, list) and len(k_angle_list) > 0:
            all_angles = []
            all_k_angle = []
            all_theta_eq = []
            for m in range(n_mol):
                offset = m * n_beads_per_mol
                for a_idx, (ai, aj, ak) in enumerate(angles_local):
                    all_angles.append([offset + ai, offset + aj, offset + ak])
                    all_k_angle.append(k_angle_list[a_idx])
                    all_theta_eq.append(theta_eq_list[a_idx])

            angle_triplets = torch.tensor(all_angles, dtype=torch.long)
            k_angle_tensor = torch.tensor(all_k_angle, dtype=torch.float32)
            theta_eq_tensor = torch.tensor(all_theta_eq, dtype=torch.float32)
        else:
            angle_triplets = torch.empty(0, 3, dtype=torch.long)
            k_angle_tensor = torch.empty(0, dtype=torch.float32)
            theta_eq_tensor = torch.empty(0, dtype=torch.float32)

        angle_prior = HarmonicAnglePriorTorch(
            angle_triplets, k_angle_tensor, theta_eq_tensor
        )

    return HarmonicRepulsivePriorTorch(harmonic_prior, repulsive_prior, angle_prior)


# =============================================================================
# LAMMPS BAM Delta Wrapper
# =============================================================================

@compile_mode("script")
class LAMMPS_BAM_Delta(torch.nn.Module):
    """
    LAMMPS wrapper for Delta Learning BAM models.

    At inference:
        F_total = F_prior (ZBL + D2) + F_delta (ML model)
    """

    def __init__(
        self,
        model,
        enr_avg_per_element: Dict[int, float],
        e_corr: float = 0.0,
        prior_config: Optional[Dict] = None,
        **kwargs
    ):
        super().__init__()
        self.model = model
        self.register_buffer("atomic_numbers", model.atomic_numbers)
        self.register_buffer("r_max", model.r_max)
        self.register_buffer("num_interactions", model.num_interactions)

        # Energy offset per element
        if enr_avg_per_element is not None:
            max_mapped_val = max(enr_avg_per_element.keys())
            enr_avg_tensor = torch.zeros(max_mapped_val + 1, dtype=torch.float32)
            for mapped_val, avg_energy in enr_avg_per_element.items():
                enr_avg_tensor[mapped_val] = avg_energy
            self.register_buffer("enr_avg_per_element", enr_avg_tensor)
        else:
            self.register_buffer("enr_avg_per_element", torch.empty(0))

        self.register_buffer("e_corr", torch.tensor(e_corr, dtype=torch.float32))

        # Initialize prior force field
        self.prior_type = ""
        if prior_config is not None:
            prior_type = prior_config.get('type', 'universal')
            self.prior_type = prior_type

            if prior_type == 'harmonic_repulsive':
                # Harmonic + Repulsive prior (needs n_atoms and types from kwargs)
                n_atoms = kwargs.get('n_atoms', 0)
                types_array = kwargs.get('types_array', [])
                self.prior_hr: Optional[HarmonicRepulsivePriorTorch] = \
                    build_harmonic_repulsive_prior_torch(
                        prior_config, n_atoms, types_array
                    )
                self.prior = None
                self.has_prior = True
            else:
                # Universal (ZBL + D2) prior
                atomic_numbers = prior_config.get('atomic_numbers', [8])
                zbl_cutoff = prior_config.get('zbl_cutoff', 5.0)
                d2_cutoff = prior_config.get('d2_cutoff', 20.0)
                s_6 = prior_config.get('s_6', 1.0)
                d = prior_config.get('d', 20.0)

                self.prior = UniversalPriorTorch(
                    atomic_numbers=atomic_numbers,
                    zbl_cutoff=zbl_cutoff,
                    d2_cutoff=d2_cutoff,
                    s_6=s_6,
                    d=d
                )
                self.prior_hr = None
                self.has_prior = True
        else:
            self.prior = None
            self.prior_hr = None
            self.has_prior = False

        # Head selection
        if not hasattr(model, "heads"):
            model.heads = [None]
        self.register_buffer(
            "head",
            torch.tensor(
                self.model.heads.index(kwargs.get("head", self.model.heads[-1])),
                dtype=torch.long,
            ).unsqueeze(0),
        )

        for param in self.model.parameters():
            param.requires_grad = False

    def forward(
        self,
        data: Dict[str, torch.Tensor],
        local_or_ghost: torch.Tensor,
        compute_virials: bool = True
    ) -> Dict[str, Optional[torch.Tensor]]:

        num_graphs = data["ptr"].numel() - 1
        data["head"] = self.head
        data["num_nodes"] = torch.tensor(
            data["positions"].shape[0],
            dtype=torch.long,
            device=data["positions"].device
        )

        compute_displacement = compute_virials

        # Get ML model output (F_delta)
        out = self.model(data, backprop=False, compute_displacement=compute_displacement)

        node_energy = out["node_energy"]
        assert node_energy is not None
        forces_delta = out["forces"]
        assert forces_delta is not None

        # Compute prior forces if available
        if self.has_prior:
            positions = data["positions"]
            edge_index = data["edge_index"]
            shifts = data.get("shifts", torch.zeros((edge_index.shape[1], 3),
                                                     device=positions.device,
                                                     dtype=positions.dtype))
            if self.prior_hr is not None:
                cell = data.get("cell", torch.zeros((3, 3),
                                                     device=positions.device,
                                                     dtype=positions.dtype))
                forces_prior = self.prior_hr(positions, edge_index, shifts, cell)
            elif self.prior is not None:
                forces_prior = self.prior(positions, edge_index, shifts)
            else:
                forces_prior = torch.zeros_like(forces_delta)
        else:
            forces_prior = torch.zeros_like(forces_delta)

        # Total forces = prior + delta
        forces_total = forces_prior + forces_delta

        # Energy calculation (same as original)
        species = data["species"]
        local_species = species[local_or_ghost]
        local_node_avg_energies = self.enr_avg_per_element[local_species]

        node_energy[local_or_ghost] = node_energy[local_or_ghost] + local_node_avg_energies
        energy = node_energy.sum()

        if self.e_corr != 0.0:
            local_count = local_or_ghost.sum().item()
            if "total_local_atoms" in data:
                total_system_atoms = data["total_local_atoms"].item()
            else:
                total_system_atoms = local_count
            e_corr_per_local = self.e_corr / total_system_atoms
            node_energy[local_or_ghost] = node_energy[local_or_ghost] + e_corr_per_local
            energy = energy + (e_corr_per_local * local_count)

        positions = data["positions"]
        displacement = out["displacement"]

        node_energy_local = node_energy * local_or_ghost
        total_energy_local = scatter_sum(
            src=node_energy_local,
            index=data["batch"],
            dim=0,
            dim_size=num_graphs,
        )

        # Virial calculation
        grad_outputs: List[Optional[torch.Tensor]] = [torch.ones_like(total_energy_local)]

        if compute_virials and displacement is not None:
            forces, virials = torch.autograd.grad(
                outputs=[total_energy_local],
                inputs=[positions, displacement],
                grad_outputs=grad_outputs,
                retain_graph=False,
                create_graph=False,
                allow_unused=True,
            )
            if forces is not None:
                forces = -1 * forces
            else:
                forces = torch.zeros_like(positions)
            if virials is not None:
                virials = -1 * virials
            else:
                virials = torch.zeros_like(displacement)
        else:
            forces = torch.autograd.grad(
                outputs=[total_energy_local],
                inputs=[data["positions"]],
                grad_outputs=grad_outputs,
                retain_graph=False,
                create_graph=False,
                allow_unused=True,
            )[0]
            if forces is not None:
                forces = -1 * forces
            else:
                forces = torch.zeros_like(positions)
            virials = None

        return {
            "total_energy_local": total_energy_local,
            "node_energy": node_energy,
            "forces": forces_total,  # Use total forces (prior + delta)
            "virials": virials,
        }
