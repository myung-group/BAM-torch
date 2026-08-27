"""
Protein CG Prior Energy Module for Delta Learning

Reproduces the prior energy from Charron et al. 2023 (Nature Chemistry)
for coarse-grained protein models. The prior includes:
- Harmonic bond potentials
- Harmonic angle potentials (in cos(theta) space)
- Fourier dihedral potentials (phi, psi, omega, gamma_1, gamma_2)
- Excluded volume repulsion ((sigma/r)^12)

Bead type convention (1-indexed, matching cg_embeds in the H5 dataset):
    1-20: Sidechain beads (ALA, CYS, ASP, ..., TYR)
    21: N (backbone nitrogen)
    22: CA (backbone alpha carbon)
    23: C (backbone carbonyl carbon)
    24: O (backbone carbonyl oxygen)

Residue structures:
    Standard (18 amino acids): [N(21), CA(22), SC(1-20), C(23), O(24)] = 5 beads
    GLY:                       [N(21), SC(6), C(23), O(24)]            = 4 beads
    PRO:                       [N(21), CA(22), SC(13), C(23), O(24)]   = 5 beads

Reference:
    Charron et al., "Navigating protein landscapes with a machine-learned
    transferable coarse-grained model", Nature Chemistry (2023).

Author: BAM-torch CG Extension
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# =============================================================================
# Constants
# =============================================================================

CHARRON_BEAD_TYPES: Dict[int, str] = {
    1: "ALA", 2: "CYS", 3: "ASP", 4: "GLU", 5: "PHE",
    6: "GLY", 7: "HIS", 8: "ILE", 9: "LYS", 10: "LEU",
    11: "MET", 12: "ASN", 13: "PRO", 14: "GLN", 15: "ARG",
    16: "SER", 17: "THR", 18: "VAL", 19: "TRP", 20: "TYR",
    21: "N", 22: "CA", 23: "C", 24: "O",
}

AA_NAME_TO_TYPE: Dict[str, int] = {v: k for k, v in CHARRON_BEAD_TYPES.items()}

# Sidechain types (1-20); GLY=6 has no CA bead
SC_TYPES: Set[int] = set(range(1, 21))
GLY_TYPE: int = 6
PRO_TYPE: int = 13
N_TYPE: int = 21
CA_TYPE: int = 22
C_TYPE: int = 23
O_TYPE: int = 24

# Amino acid names for dihedral lookup (matching NPZ key prefixes)
AA_NAMES: List[str] = [
    "ALA", "CYS", "ASP", "GLU", "PHE", "GLY", "HIS", "ILE",
    "LYS", "LEU", "MET", "ASN", "PRO", "GLN", "ARG", "SER",
    "THR", "VAL", "TRP", "TYR",
]


# =============================================================================
# Residue Data Structure
# =============================================================================

@dataclass
class Residue:
    """Represents one CG residue parsed from the bead type array."""

    index: int                  # Residue sequential index (0-based)
    bead_indices: List[int]     # Global bead indices for all beads in this residue
    bead_types: List[int]       # Bead type IDs
    sc_type: int                # Sidechain bead type (1-20)
    aa_name: str                # Three-letter amino acid name
    is_gly: bool                # True if GLY (4-bead, no CA)
    context: int = 0            # 0=bulk, 1=n_term, 2=c_term

    @property
    def n_idx(self) -> int:
        """Global index of N bead."""
        return self.bead_indices[0]

    @property
    def ca_idx(self) -> Optional[int]:
        """Global index of CA bead (None for GLY)."""
        if self.is_gly:
            return None
        return self.bead_indices[1]

    @property
    def sc_idx(self) -> int:
        """Global index of sidechain bead."""
        if self.is_gly:
            return self.bead_indices[1]  # GLY: [N, SC, C, O]
        return self.bead_indices[2]      # Standard: [N, CA, SC, C, O]

    @property
    def c_idx(self) -> int:
        """Global index of C bead."""
        if self.is_gly:
            return self.bead_indices[2]
        return self.bead_indices[3]

    @property
    def o_idx(self) -> int:
        """Global index of O bead."""
        if self.is_gly:
            return self.bead_indices[3]
        return self.bead_indices[4]

    @property
    def second_bead_idx(self) -> int:
        """Index of the second bead (CA for standard, SC for GLY).

        This is the bead used in phi/psi/omega dihedrals at the
        '[CA or SC]' position.
        """
        if self.is_gly:
            return self.sc_idx
        return self.ca_idx


# =============================================================================
# Topology Builder
# =============================================================================

class ProteinTopology:
    """Parses bead type arrays and generates bond/angle/dihedral topology.

    The topology is built by scanning the bead type array sequentially.
    N(21) marks the start of each residue. If the next bead is CA(22),
    it is a standard 5-bead residue; otherwise (GLY), it is 4-bead.

    Attributes:
        residues: List of Residue objects.
        bond_list: (n_bonds, 2) array of bead index pairs.
        bond_types: (n_bonds, 2) array of bead type pairs.
        bond_context: (n_bonds,) array of context labels.
        angle_list: (n_angles, 3) array of bead index triples (i, j_center, k).
        angle_types: (n_angles, 3) array of bead type triples.
        angle_context: (n_angles,) array of context labels.
        dihedral_groups: Dict mapping dihedral category to list of
            (indices_4, types_4) tuples.
        exclusion_pairs: Set of (i, j) pairs excluded from repulsion.
    """

    def __init__(self, types: np.ndarray) -> None:
        self.types = np.asarray(types, dtype=np.int64)
        self.n_beads = len(self.types)

        self.residues: List[Residue] = []
        self._parse_residues()
        self._assign_contexts()

        # Build topology
        self.bonds: List[Tuple[int, int]] = []
        self.bond_type_list: List[Tuple[int, int]] = []
        self.bond_ctx: List[int] = []
        self._build_bonds()

        self.angles: List[Tuple[int, int, int]] = []
        self.angle_type_list: List[Tuple[int, int, int]] = []
        self.angle_ctx: List[int] = []
        self._build_angles()

        self.dihedral_groups: Dict[str, List[Tuple[List[int], List[int]]]] = {}
        self._build_dihedrals()

        self.exclusion_pairs: Set[Tuple[int, int]] = set()
        self._build_exclusions()

        # Convert to arrays
        self.bond_list = np.array(self.bonds, dtype=np.int64) if self.bonds else np.zeros((0, 2), dtype=np.int64)
        self.bond_types = np.array(self.bond_type_list, dtype=np.int64) if self.bond_type_list else np.zeros((0, 2), dtype=np.int64)
        self.bond_context = np.array(self.bond_ctx, dtype=np.int64) if self.bond_ctx else np.zeros(0, dtype=np.int64)
        self.angle_list = np.array(self.angles, dtype=np.int64) if self.angles else np.zeros((0, 3), dtype=np.int64)
        self.angle_types = np.array(self.angle_type_list, dtype=np.int64) if self.angle_type_list else np.zeros((0, 3), dtype=np.int64)
        self.angle_context = np.array(self.angle_ctx, dtype=np.int64) if self.angle_ctx else np.zeros(0, dtype=np.int64)

        logger.info(
            "ProteinTopology: %d residues, %d bonds, %d angles, %d dihedrals",
            len(self.residues),
            len(self.bonds),
            len(self.angles),
            sum(len(v) for v in self.dihedral_groups.values()),
        )

    # -----------------------------------------------------------------
    # Residue parsing
    # -----------------------------------------------------------------

    def _parse_residues(self) -> None:
        """Scan bead type array to identify residues."""
        i = 0
        res_idx = 0
        while i < self.n_beads:
            if self.types[i] != N_TYPE:
                raise ValueError(
                    f"Expected N(21) at bead index {i}, got type {self.types[i]}. "
                    "Bead type array does not follow expected protein CG topology."
                )
            # Check if next bead is CA (standard) or SC (GLY)
            if i + 1 >= self.n_beads:
                raise ValueError(f"Unexpected end of bead array after N at index {i}")

            next_type = self.types[i + 1]
            if next_type == CA_TYPE:
                # Standard 5-bead residue: N, CA, SC, C, O
                if i + 4 >= self.n_beads:
                    raise ValueError(
                        f"Incomplete standard residue starting at index {i}"
                    )
                sc_type = int(self.types[i + 2])
                if sc_type not in SC_TYPES:
                    raise ValueError(
                        f"Expected SC type (1-20) at index {i+2}, got {sc_type}"
                    )
                if self.types[i + 3] != C_TYPE:
                    raise ValueError(
                        f"Expected C(23) at index {i+3}, got {self.types[i+3]}"
                    )
                if self.types[i + 4] != O_TYPE:
                    raise ValueError(
                        f"Expected O(24) at index {i+4}, got {self.types[i+4]}"
                    )
                bead_indices = [i, i + 1, i + 2, i + 3, i + 4]
                bead_types = [int(self.types[j]) for j in bead_indices]
                self.residues.append(Residue(
                    index=res_idx,
                    bead_indices=bead_indices,
                    bead_types=bead_types,
                    sc_type=sc_type,
                    aa_name=CHARRON_BEAD_TYPES[sc_type],
                    is_gly=False,
                ))
                i += 5
            elif next_type in SC_TYPES:
                # GLY 4-bead residue: N, SC(6), C, O
                sc_type = int(next_type)
                if sc_type != GLY_TYPE:
                    raise ValueError(
                        f"Only GLY (type 6) should have 4-bead layout, "
                        f"but found SC type {sc_type} without CA at index {i+1}"
                    )
                if i + 3 >= self.n_beads:
                    raise ValueError(
                        f"Incomplete GLY residue starting at index {i}"
                    )
                if self.types[i + 2] != C_TYPE:
                    raise ValueError(
                        f"Expected C(23) at index {i+2}, got {self.types[i+2]}"
                    )
                if self.types[i + 3] != O_TYPE:
                    raise ValueError(
                        f"Expected O(24) at index {i+3}, got {self.types[i+3]}"
                    )
                bead_indices = [i, i + 1, i + 2, i + 3]
                bead_types = [int(self.types[j]) for j in bead_indices]
                self.residues.append(Residue(
                    index=res_idx,
                    bead_indices=bead_indices,
                    bead_types=bead_types,
                    sc_type=sc_type,
                    aa_name="GLY",
                    is_gly=True,
                ))
                i += 4
            else:
                raise ValueError(
                    f"Unexpected bead type {next_type} at index {i+1} "
                    f"(expected CA=22 or GLY_SC=6)"
                )
            res_idx += 1

    def _assign_contexts(self) -> None:
        """Assign terminal context labels to residues."""
        if len(self.residues) == 0:
            return
        if len(self.residues) == 1:
            # Single residue: both N- and C-terminal (use n_term)
            self.residues[0].context = 1
            return
        self.residues[0].context = 1   # n_term
        self.residues[-1].context = 2  # c_term
        for res in self.residues[1:-1]:
            res.context = 0  # bulk

    # -----------------------------------------------------------------
    # Bond generation
    # -----------------------------------------------------------------

    def _add_bond(self, idx_a: int, idx_b: int, context: int) -> None:
        """Register a bond between two beads."""
        t_a = int(self.types[idx_a])
        t_b = int(self.types[idx_b])
        self.bonds.append((idx_a, idx_b))
        self.bond_type_list.append((t_a, t_b))
        self.bond_ctx.append(context)

    def _build_bonds(self) -> None:
        """Generate intra- and inter-residue bonds."""
        for res in self.residues:
            ctx = res.context
            if res.is_gly:
                # GLY: N-SC, SC-C, C-O
                self._add_bond(res.n_idx, res.sc_idx, ctx)
                self._add_bond(res.sc_idx, res.c_idx, ctx)
                self._add_bond(res.c_idx, res.o_idx, ctx)
            else:
                # Standard: N-CA, CA-SC, CA-C, C-O
                self._add_bond(res.n_idx, res.ca_idx, ctx)
                self._add_bond(res.ca_idx, res.sc_idx, ctx)
                self._add_bond(res.ca_idx, res.c_idx, ctx)
                self._add_bond(res.c_idx, res.o_idx, ctx)

        # Inter-residue bonds: C(i)-N(i+1)
        for k in range(len(self.residues) - 1):
            res_i = self.residues[k]
            res_j = self.residues[k + 1]
            # Use the context of the bond endpoint in the later residue's context
            # Convention: inter-residue bond context follows the earlier residue
            ctx = res_i.context
            self._add_bond(res_i.c_idx, res_j.n_idx, ctx)

    # -----------------------------------------------------------------
    # Angle generation
    # -----------------------------------------------------------------

    def _build_angles(self) -> None:
        """Generate all angles from the bond graph.

        An angle (i, j, k) exists when bonds i-j and j-k both exist,
        with j as the central atom.
        """
        # Build adjacency from bonds
        adjacency: Dict[int, List[int]] = {}
        for a, b in self.bonds:
            adjacency.setdefault(a, []).append(b)
            adjacency.setdefault(b, []).append(a)

        # Also need a quick context lookup: for each bond, what context?
        bond_ctx_map: Dict[Tuple[int, int], int] = {}
        for (a, b), ctx in zip(self.bonds, self.bond_ctx):
            bond_ctx_map[(a, b)] = ctx
            bond_ctx_map[(b, a)] = ctx

        # Bead -> residue context
        bead_to_context: Dict[int, int] = {}
        for res in self.residues:
            for bi in res.bead_indices:
                bead_to_context[bi] = res.context

        seen_angles: Set[Tuple[int, int, int]] = set()

        for j, neighbors in adjacency.items():
            for ii, ni in enumerate(neighbors):
                for nk in neighbors[ii + 1:]:
                    # Angle: ni - j - nk
                    canon = (min(ni, nk), j, max(ni, nk))
                    if canon in seen_angles:
                        continue
                    seen_angles.add(canon)

                    t_i = int(self.types[ni])
                    t_j = int(self.types[j])
                    t_k = int(self.types[nk])

                    # Context: use the center atom's residue context
                    ctx = bead_to_context.get(j, 0)

                    self.angles.append((ni, j, nk))
                    self.angle_type_list.append((t_i, t_j, t_k))
                    self.angle_ctx.append(ctx)

    # -----------------------------------------------------------------
    # Dihedral generation
    # -----------------------------------------------------------------

    def _build_dihedrals(self) -> None:
        """Generate phi, psi, omega, gamma_1, and gamma_2 dihedrals."""
        n_res = len(self.residues)

        for k, res in enumerate(self.residues):
            # ------ phi(i): C(i-1) - N(i) - [CA/SC](i) - C(i) ------
            if k > 0:
                prev = self.residues[k - 1]
                indices = [prev.c_idx, res.n_idx, res.second_bead_idx, res.c_idx]
                types_ = [int(self.types[idx]) for idx in indices]
                # Determine dihedral type name: {AA}_phi
                dih_name = f"{res.aa_name}_phi"
                self.dihedral_groups.setdefault(dih_name, []).append(
                    (indices, types_)
                )

            # ------ psi(i): N(i) - [CA/SC](i) - C(i) - N(i+1) ------
            if k < n_res - 1:
                next_res = self.residues[k + 1]
                indices = [res.n_idx, res.second_bead_idx, res.c_idx, next_res.n_idx]
                types_ = [int(self.types[idx]) for idx in indices]
                dih_name = f"{res.aa_name}_psi"
                self.dihedral_groups.setdefault(dih_name, []).append(
                    (indices, types_)
                )

            # ------ omega: [CA/SC](i) - C(i) - N(i+1) - [CA/SC](i+1) ------
            if k < n_res - 1:
                next_res = self.residues[k + 1]
                indices = [
                    res.second_bead_idx,
                    res.c_idx,
                    next_res.n_idx,
                    next_res.second_bead_idx,
                ]
                types_ = [int(self.types[idx]) for idx in indices]
                # PRO omega if residue i+1 is PRO
                if next_res.sc_type == PRO_TYPE:
                    dih_name = "pro_omega"
                else:
                    dih_name = "non_pro_omega"
                self.dihedral_groups.setdefault(dih_name, []).append(
                    (indices, types_)
                )

            # ------ gamma_1: N(i) - SC(i) - C(i) - CA(i) ------
            # Only for standard residues (not GLY): requires both CA and SC
            if not res.is_gly:
                indices = [res.n_idx, res.sc_idx, res.c_idx, res.ca_idx]
                types_ = [int(self.types[idx]) for idx in indices]
                self.dihedral_groups.setdefault("gamma_1", []).append(
                    (indices, types_)
                )

            # ------ gamma_2: [CA/SC](i) - O(i) - N(i+1) - C(i) ------
            # Improper at C: involves CA/SC(i), O(i), N(i+1), C(i)
            # From NPZ: non-zero at (22,24,21,23) = CA-O-N-C
            #                    and (6,24,21,23)  = GLY_SC-O-N-C
            if k < n_res - 1:
                next_res = self.residues[k + 1]
                indices = [
                    res.second_bead_idx,  # CA or SC(GLY)
                    res.o_idx,            # O
                    next_res.n_idx,       # N(i+1)
                    res.c_idx,            # C
                ]
                types_ = [int(self.types[idx]) for idx in indices]
                self.dihedral_groups.setdefault("gamma_2", []).append(
                    (indices, types_)
                )

    # -----------------------------------------------------------------
    # Exclusion pairs (1-2 and 1-3 bonded)
    # -----------------------------------------------------------------

    def _build_exclusions(self) -> None:
        """Build set of excluded pairs for repulsion (1-2 and 1-3 bonded)."""
        # 1-2 pairs (direct bonds)
        for a, b in self.bonds:
            pair = (min(a, b), max(a, b))
            self.exclusion_pairs.add(pair)

        # 1-3 pairs (separated by exactly 2 bonds)
        adjacency: Dict[int, Set[int]] = {}
        for a, b in self.bonds:
            adjacency.setdefault(a, set()).add(b)
            adjacency.setdefault(b, set()).add(a)

        for j, neighbors in adjacency.items():
            nbr_list = list(neighbors)
            for ii in range(len(nbr_list)):
                for kk in range(ii + 1, len(nbr_list)):
                    pair = (min(nbr_list[ii], nbr_list[kk]),
                            max(nbr_list[ii], nbr_list[kk]))
                    self.exclusion_pairs.add(pair)


# =============================================================================
# Sub-modules: Bond, Angle, Dihedral, Repulsion
# =============================================================================

class HarmonicBondPrior(nn.Module):
    """Harmonic bond prior: E = 0.5 * k * (r - r_0)^2.

    Parameters are stored as (3, 25, 25) tensors for 3 contexts
    (bulk, n_term, c_term), indexed by bead type IDs.
    """

    def __init__(
        self,
        k_bulk: np.ndarray,
        x0_bulk: np.ndarray,
        k_nterm: np.ndarray,
        x0_nterm: np.ndarray,
        k_cterm: np.ndarray,
        x0_cterm: np.ndarray,
    ) -> None:
        super().__init__()
        # Stack into (3, 25, 25): [bulk, n_term, c_term]
        k_all = np.stack([k_bulk, k_nterm, k_cterm], axis=0)
        x0_all = np.stack([x0_bulk, x0_nterm, x0_cterm], axis=0)
        self.register_buffer("k", torch.tensor(k_all, dtype=torch.float32))
        self.register_buffer("x0", torch.tensor(x0_all, dtype=torch.float32))

    def forward(
        self,
        positions: torch.Tensor,
        bond_indices: torch.Tensor,
        bond_types: torch.Tensor,
        bond_contexts: torch.Tensor,
    ) -> torch.Tensor:
        """Compute total harmonic bond energy.

        Args:
            positions: (n_beads, 3) with requires_grad=True.
            bond_indices: (n_bonds, 2) long tensor of bead index pairs.
            bond_types: (n_bonds, 2) long tensor of bead type pairs.
            bond_contexts: (n_bonds,) long tensor, 0=bulk, 1=nterm, 2=cterm.

        Returns:
            Scalar total bond energy.
        """
        if bond_indices.shape[0] == 0:
            return torch.zeros(1, device=positions.device, dtype=positions.dtype)

        r_ij = positions[bond_indices[:, 1]] - positions[bond_indices[:, 0]]
        dist = torch.norm(r_ij, dim=-1)  # (n_bonds,)

        t0 = bond_types[:, 0]
        t1 = bond_types[:, 1]
        k_val = self.k[bond_contexts, t0, t1]
        x0_val = self.x0[bond_contexts, t0, t1]

        energy = 0.5 * k_val * (dist - x0_val) ** 2
        return energy.sum()


class HarmonicAnglePrior(nn.Module):
    """Harmonic angle prior in cos(theta) space.

    E = 0.5 * k * (cos(theta) - cos(theta_0))^2

    The equilibrium values cos(theta_0) are stored directly in the NPZ
    as the ``x_0`` arrays.
    """

    def __init__(
        self,
        k_bulk: np.ndarray,
        x0_bulk: np.ndarray,
        k_nterm: np.ndarray,
        x0_nterm: np.ndarray,
        k_cterm: np.ndarray,
        x0_cterm: np.ndarray,
    ) -> None:
        super().__init__()
        k_all = np.stack([k_bulk, k_nterm, k_cterm], axis=0)
        x0_all = np.stack([x0_bulk, x0_nterm, x0_cterm], axis=0)
        self.register_buffer("k", torch.tensor(k_all, dtype=torch.float32))
        self.register_buffer("cos_theta0", torch.tensor(x0_all, dtype=torch.float32))

    def forward(
        self,
        positions: torch.Tensor,
        angle_indices: torch.Tensor,
        angle_types: torch.Tensor,
        angle_contexts: torch.Tensor,
    ) -> torch.Tensor:
        """Compute total harmonic angle energy.

        Args:
            positions: (n_beads, 3).
            angle_indices: (n_angles, 3) long tensor (i, j_center, k).
            angle_types: (n_angles, 3) long tensor.
            angle_contexts: (n_angles,) long tensor.

        Returns:
            Scalar total angle energy.
        """
        if angle_indices.shape[0] == 0:
            return torch.zeros(1, device=positions.device, dtype=positions.dtype)

        r_ji = positions[angle_indices[:, 0]] - positions[angle_indices[:, 1]]
        r_jk = positions[angle_indices[:, 2]] - positions[angle_indices[:, 1]]

        # cos(theta) via dot product
        norm_ji = torch.norm(r_ji, dim=-1, keepdim=True).clamp(min=1e-8)
        norm_jk = torch.norm(r_jk, dim=-1, keepdim=True).clamp(min=1e-8)
        cos_theta = (r_ji * r_jk).sum(dim=-1) / (norm_ji.squeeze(-1) * norm_jk.squeeze(-1))
        cos_theta = cos_theta.clamp(-1.0 + 1e-7, 1.0 - 1e-7)

        t0 = angle_types[:, 0]
        t1 = angle_types[:, 1]
        t2 = angle_types[:, 2]
        k_val = self.k[angle_contexts, t0, t1, t2]
        cos_eq = self.cos_theta0[angle_contexts, t0, t1, t2]

        energy = 0.5 * k_val * (cos_theta - cos_eq) ** 2
        return energy.sum()


class FourierDihedralPrior(nn.Module):
    """Fourier dihedral prior for protein CG models.

    E = v_0 + sum_{n=1}^{N} [k1s_n * cos(n * phi) + k2s_n * sin(n * phi)]

    At initialization, the full 5D parameter tensors are pre-indexed for
    each dihedral in the topology to create compact per-dihedral arrays,
    avoiding expensive runtime lookups.
    """

    def __init__(
        self,
        dihedral_groups: Dict[str, List[Tuple[List[int], List[int]]]],
        params: Dict[str, np.ndarray],
    ) -> None:
        super().__init__()

        all_indices: List[List[int]] = []
        all_k1s: List[np.ndarray] = []
        all_k2s: List[np.ndarray] = []
        all_v0: List[float] = []

        for dih_name, dih_list in dihedral_groups.items():
            # Find parameter key prefix
            param_prefix = self._get_param_prefix(dih_name)
            if param_prefix is None:
                logger.warning(
                    "No parameter prefix found for dihedral group '%s', skipping",
                    dih_name,
                )
                continue

            k1s_key = f"{param_prefix}_k1s"
            k2s_key = f"{param_prefix}_k2s"
            v0_key = f"{param_prefix}_v_0"

            if k1s_key not in params:
                logger.warning("Parameter '%s' not found in NPZ, skipping", k1s_key)
                continue

            k1s_arr = params[k1s_key]  # (n_fourier, D, D, D, D)
            k2s_arr = params[k2s_key]
            v0_arr = params[v0_key]    # (D, D, D, D)
            n_fourier = k1s_arr.shape[0]
            dim = k1s_arr.shape[1]  # 24 or 25

            for indices, types_ in dih_list:
                # Index into parameter arrays using bead types
                # For 24-dim arrays: types 1-23 map to indices 1-23 directly
                # For 25-dim arrays: types 1-24 map to indices 1-24
                t = types_
                if any(ti >= dim for ti in t):
                    # Type exceeds array dimension (e.g., O=24 in a 24-dim array)
                    # This dihedral has no parameters, skip
                    continue

                v0_val = float(v0_arr[t[0], t[1], t[2], t[3]])
                k1s_val = k1s_arr[:, t[0], t[1], t[2], t[3]]  # (n_fourier,)
                k2s_val = k2s_arr[:, t[0], t[1], t[2], t[3]]

                # Skip if all parameters are zero
                if (abs(v0_val) < 1e-12
                        and np.all(np.abs(k1s_val) < 1e-12)
                        and np.all(np.abs(k2s_val) < 1e-12)):
                    continue

                all_indices.append(indices)
                all_k1s.append(k1s_val)
                all_k2s.append(k2s_val)
                all_v0.append(v0_val)

        self.n_dihedrals = len(all_indices)

        if self.n_dihedrals == 0:
            self.register_buffer(
                "dih_indices", torch.zeros((0, 4), dtype=torch.long)
            )
            self.register_buffer("v0", torch.zeros(0))
            self.register_buffer("k1s", torch.zeros((0, 0)))
            self.register_buffer("k2s", torch.zeros((0, 0)))
            return

        # Pad k1s/k2s to same number of Fourier terms
        max_n = max(arr.shape[0] for arr in all_k1s)
        k1s_padded = np.zeros((self.n_dihedrals, max_n), dtype=np.float32)
        k2s_padded = np.zeros((self.n_dihedrals, max_n), dtype=np.float32)
        for i, (k1, k2) in enumerate(zip(all_k1s, all_k2s)):
            k1s_padded[i, : k1.shape[0]] = k1
            k2s_padded[i, : k2.shape[0]] = k2

        self.register_buffer(
            "dih_indices",
            torch.tensor(all_indices, dtype=torch.long),
        )
        self.register_buffer("v0", torch.tensor(all_v0, dtype=torch.float32))
        self.register_buffer("k1s", torch.tensor(k1s_padded, dtype=torch.float32))
        self.register_buffer("k2s", torch.tensor(k2s_padded, dtype=torch.float32))

        # Fourier order indices: 1, 2, ..., max_n
        self.register_buffer(
            "fourier_n",
            torch.arange(1, max_n + 1, dtype=torch.float32),
        )

        logger.info(
            "FourierDihedralPrior: %d dihedrals, max %d Fourier terms",
            self.n_dihedrals,
            max_n,
        )

    @staticmethod
    def _get_param_prefix(dih_name: str) -> Optional[str]:
        """Map dihedral group name to NPZ parameter key prefix.

        Examples:
            'ALA_phi' -> 'ALA_phi'
            'non_pro_omega' -> 'non_pro_omega'
            'gamma_1' -> 'gamma_1'
        """
        # The dih_name should already match the NPZ key prefix
        return dih_name

    @staticmethod
    def _compute_dihedral_angle(
        p0: torch.Tensor,
        p1: torch.Tensor,
        p2: torch.Tensor,
        p3: torch.Tensor,
    ) -> torch.Tensor:
        """Compute dihedral angle phi for atoms p0-p1-p2-p3.

        Uses the atan2 formula for numerical stability.

        Args:
            p0, p1, p2, p3: (N, 3) position tensors.

        Returns:
            phi: (N,) dihedral angles in radians, range [-pi, pi].
        """
        b1 = p1 - p0
        b2 = p2 - p1
        b3 = p3 - p2

        n1 = torch.cross(b1, b2, dim=-1)
        n2 = torch.cross(b2, b3, dim=-1)

        b2_norm = torch.norm(b2, dim=-1, keepdim=True).clamp(min=1e-8)
        b2_hat = b2 / b2_norm

        m = torch.cross(n1, b2_hat, dim=-1)

        x = (n1 * n2).sum(dim=-1)
        y = (m * n2).sum(dim=-1)

        return torch.atan2(y, x)

    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        """Compute total Fourier dihedral energy.

        Args:
            positions: (n_beads, 3).

        Returns:
            Scalar total dihedral energy.
        """
        if self.n_dihedrals == 0:
            return torch.zeros(1, device=positions.device, dtype=positions.dtype)

        p0 = positions[self.dih_indices[:, 0]]
        p1 = positions[self.dih_indices[:, 1]]
        p2 = positions[self.dih_indices[:, 2]]
        p3 = positions[self.dih_indices[:, 3]]

        phi = self._compute_dihedral_angle(p0, p1, p2, p3)  # (n_dih,)

        # Fourier expansion: E = v_0 + sum_n [k1s_n * cos(n*phi) + k2s_n * sin(n*phi)]
        # phi: (n_dih,), fourier_n: (max_n,)
        n_phi = phi.unsqueeze(-1) * self.fourier_n.unsqueeze(0)  # (n_dih, max_n)

        cos_terms = (self.k1s * torch.cos(n_phi)).sum(dim=-1)
        sin_terms = (self.k2s * torch.sin(n_phi)).sum(dim=-1)

        energy = self.v0 + cos_terms + sin_terms
        return energy.sum()


class ExcludedVolumePrior(nn.Module):
    """Excluded volume repulsion: E = (sigma_{ij} / r_{ij})^12.

    Applied to all non-bonded pairs (excluding 1-2 and 1-3 bonded pairs)
    within a specified cutoff distance.

    For small proteins (< ~1000 beads), all-pairs computation is used.
    For larger systems, a cutoff-based neighbor list should be employed.
    """

    def __init__(
        self,
        sigma: np.ndarray,
        exclusion_pairs: Set[Tuple[int, int]],
        n_beads: int,
        bead_types: np.ndarray,
        cutoff: float = 8.0,
    ) -> None:
        super().__init__()
        self.register_buffer(
            "sigma", torch.tensor(sigma, dtype=torch.float32)
        )
        self.cutoff = cutoff
        self.n_beads = n_beads

        # Pre-compute non-excluded pair indices and their sigma values
        types_ = np.asarray(bead_types, dtype=np.int64)
        pair_i_list: List[int] = []
        pair_j_list: List[int] = []
        sigma_list: List[float] = []

        for i in range(n_beads):
            for j in range(i + 1, n_beads):
                pair = (i, j)
                if pair in exclusion_pairs:
                    continue
                ti, tj = int(types_[i]), int(types_[j])
                s = float(sigma[ti, tj])
                if s < 1e-8:
                    continue
                pair_i_list.append(i)
                pair_j_list.append(j)
                sigma_list.append(s)

        n_pairs = len(pair_i_list)
        self.register_buffer(
            "pair_i", torch.tensor(pair_i_list, dtype=torch.long)
        )
        self.register_buffer(
            "pair_j", torch.tensor(pair_j_list, dtype=torch.long)
        )
        self.register_buffer(
            "pair_sigma", torch.tensor(sigma_list, dtype=torch.float32)
        )

        logger.info(
            "ExcludedVolumePrior: %d non-excluded pairs (cutoff=%.1f A)",
            n_pairs,
            cutoff,
        )

    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        """Compute excluded volume repulsion energy.

        Args:
            positions: (n_beads, 3).

        Returns:
            Scalar repulsion energy.
        """
        if self.pair_i.shape[0] == 0:
            return torch.zeros(1, device=positions.device, dtype=positions.dtype)

        r_ij = positions[self.pair_j] - positions[self.pair_i]  # (n_pairs, 3)
        dist = torch.norm(r_ij, dim=-1)  # (n_pairs,)

        # Apply cutoff
        mask = dist < self.cutoff
        if not mask.any():
            return torch.zeros(1, device=positions.device, dtype=positions.dtype)

        dist_masked = dist[mask]
        sigma_masked = self.pair_sigma[mask]

        # Smooth switching function near cutoff to avoid force discontinuities
        # Using a cosine switch between r_switch and r_cutoff
        r_switch = self.cutoff * 0.8
        raw_energy = (sigma_masked / dist_masked) ** 12

        # Switching function: 1 at r < r_switch, smoothly goes to 0 at r_cutoff
        switch = torch.ones_like(dist_masked)
        transition = (dist_masked > r_switch) & (dist_masked < self.cutoff)
        if transition.any():
            x = (dist_masked[transition] - r_switch) / (self.cutoff - r_switch)
            # Cosine switch: S(x) = 0.5 * (1 + cos(pi * x))
            switch[transition] = 0.5 * (1.0 + torch.cos(torch.pi * x))

        energy = (raw_energy * switch).sum()
        return energy


# =============================================================================
# Main Assembly: ProteinCGPrior
# =============================================================================

class ProteinCGPrior(nn.Module):
    """Protein CG Prior Energy Module.

    Combines harmonic bonds, harmonic angles (in cos-theta space),
    Fourier dihedrals, and excluded volume repulsion into a single
    differentiable energy model.

    Usage:
        prior = ProteinCGPrior.from_charron2023(params_path, bead_types)
        E = prior(positions_tensor)
        F = prior.compute_forces(positions_np)

    Args:
        params_path: Path to NPZ file with prior parameters.
        bead_types: 1D array of bead type IDs (1-24), shape (n_beads,).
        repulsion_cutoff: Cutoff distance for excluded volume (Angstroms).
    """

    def __init__(
        self,
        params_path: str,
        bead_types: np.ndarray,
        repulsion_cutoff: float = 8.0,
    ) -> None:
        super().__init__()

        bead_types = np.asarray(bead_types, dtype=np.int64)
        self.n_beads = len(bead_types)

        # Load parameters
        params = dict(np.load(params_path))

        # Build topology
        self.topology = ProteinTopology(bead_types)

        # --- Bond prior ---
        self.bond_prior = HarmonicBondPrior(
            k_bulk=params["bulk_bonds_k"],
            x0_bulk=params["bulk_bonds_x_0"],
            k_nterm=params["n_term_bonds_k"],
            x0_nterm=params["n_term_bonds_x_0"],
            k_cterm=params["c_term_bonds_k"],
            x0_cterm=params["c_term_bonds_x_0"],
        )

        # --- Angle prior ---
        self.angle_prior = HarmonicAnglePrior(
            k_bulk=params["bulk_angles_k"],
            x0_bulk=params["bulk_angles_x_0"],
            k_nterm=params["n_term_angles_k"],
            x0_nterm=params["n_term_angles_x_0"],
            k_cterm=params["c_term_angles_k"],
            x0_cterm=params["c_term_angles_x_0"],
        )

        # --- Dihedral prior ---
        self.dihedral_prior = FourierDihedralPrior(
            dihedral_groups=self.topology.dihedral_groups,
            params=params,
        )

        # --- Excluded volume prior ---
        self.repulsion_prior = ExcludedVolumePrior(
            sigma=params["repulsion_sigma"],
            exclusion_pairs=self.topology.exclusion_pairs,
            n_beads=self.n_beads,
            bead_types=bead_types,
            cutoff=repulsion_cutoff,
        )

        # Register topology arrays as buffers for batched operation
        self.register_buffer(
            "bond_indices",
            torch.tensor(self.topology.bond_list, dtype=torch.long),
        )
        self.register_buffer(
            "bond_types_buf",
            torch.tensor(self.topology.bond_types, dtype=torch.long),
        )
        self.register_buffer(
            "bond_contexts",
            torch.tensor(self.topology.bond_context, dtype=torch.long),
        )
        self.register_buffer(
            "angle_indices",
            torch.tensor(self.topology.angle_list, dtype=torch.long),
        )
        self.register_buffer(
            "angle_types_buf",
            torch.tensor(self.topology.angle_types, dtype=torch.long),
        )
        self.register_buffer(
            "angle_contexts",
            torch.tensor(self.topology.angle_context, dtype=torch.long),
        )

        logger.info(
            "ProteinCGPrior initialized: %d beads, %d residues",
            self.n_beads,
            len(self.topology.residues),
        )

    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        """Compute total prior energy.

        Args:
            positions: (n_beads, 3) tensor with requires_grad=True for
                force computation via autograd.

        Returns:
            Scalar total prior energy.
        """
        E_bond = self.bond_prior(
            positions, self.bond_indices, self.bond_types_buf, self.bond_contexts
        )
        E_angle = self.angle_prior(
            positions, self.angle_indices, self.angle_types_buf, self.angle_contexts
        )
        E_dihedral = self.dihedral_prior(positions)
        E_repulsion = self.repulsion_prior(positions)

        return E_bond + E_angle + E_dihedral + E_repulsion

    def compute_energy_components(
        self, positions_input,
    ) -> Dict[str, float]:
        """Compute individual energy components for analysis.

        Args:
            positions_input: (n_beads, 3) tensor or NumPy array.

        Returns:
            Dictionary with keys 'bond', 'angle', 'dihedral', 'repulsion',
            each mapping to a float energy value.
        """
        if isinstance(positions_input, np.ndarray):
            positions_input = torch.tensor(
                positions_input, dtype=torch.float32
            )
        positions = positions_input.to(self.bond_indices.device)
        return {
            "bond": self.bond_prior(
                positions, self.bond_indices, self.bond_types_buf, self.bond_contexts
            ).item(),
            "angle": self.angle_prior(
                positions, self.angle_indices, self.angle_types_buf, self.angle_contexts
            ).item(),
            "dihedral": self.dihedral_prior(positions).item(),
            "repulsion": self.repulsion_prior(positions).item(),
        }

    def compute_forces_analytical(self, positions: torch.Tensor) -> torch.Tensor:
        """Compute prior forces analytically (no autograd). TorchScript-safe.

        Args:
            positions: (n_beads, 3) tensor.

        Returns:
            (n_beads, 3) tensor of forces F = -dE/dr.
        """
        n = positions.shape[0]
        forces = torch.zeros_like(positions)

        # ---- Bond forces: E = k/2*(r-r0)^2, F = -k*(r-r0)*r_hat ----
        bi = self.bond_indices
        if bi.shape[0] > 0:
            r_ij = positions[bi[:, 1]] - positions[bi[:, 0]]  # j - i
            dist = torch.norm(r_ij, dim=-1, keepdim=True).clamp(min=1e-8)
            r_hat = r_ij / dist
            dist_s = dist.squeeze(-1)
            t0 = self.bond_types_buf[:, 0]
            t1 = self.bond_types_buf[:, 1]
            k_b = self.bond_prior.k[self.bond_contexts, t0, t1]
            x0_b = self.bond_prior.x0[self.bond_contexts, t0, t1]
            f_mag = -k_b * (dist_s - x0_b)  # scalar per bond
            f_vec = f_mag.unsqueeze(-1) * r_hat  # (n_bonds, 3)
            # F on j: +f_vec, F on i: -f_vec
            forces.scatter_add_(0, bi[:, 1:2].expand_as(f_vec), f_vec)
            forces.scatter_add_(0, bi[:, 0:1].expand_as(f_vec), -f_vec)

        # ---- Angle forces: E = k/2*(cosθ - cosθ0)^2 ----
        ai = self.angle_indices
        if ai.shape[0] > 0:
            # i-j(center)-k
            r_ji = positions[ai[:, 0]] - positions[ai[:, 1]]
            r_jk = positions[ai[:, 2]] - positions[ai[:, 1]]
            ra = torch.norm(r_ji, dim=-1, keepdim=True).clamp(min=1e-8)
            rb = torch.norm(r_jk, dim=-1, keepdim=True).clamp(min=1e-8)
            cos_theta = (r_ji * r_jk).sum(dim=-1) / (ra.squeeze(-1) * rb.squeeze(-1))
            cos_theta = cos_theta.clamp(-1.0 + 1e-7, 1.0 - 1e-7)

            at0 = self.angle_types_buf[:, 0]
            at1 = self.angle_types_buf[:, 1]
            at2 = self.angle_types_buf[:, 2]
            k_a = self.angle_prior.k[self.angle_contexts, at0, at1, at2]
            cos_eq = self.angle_prior.cos_theta0[self.angle_contexts, at0, at1, at2]

            # dE/d(cosθ) = k*(cosθ - cosθ0)
            dE_dcos = k_a * (cos_theta - cos_eq)  # (n_angles,)

            # d(cosθ)/dr_i = r_jk/(ra*rb) - cosθ * r_ji/ra^2
            dcosth_dri = r_jk / (ra * rb) - cos_theta.unsqueeze(-1) * r_ji / (ra * ra)
            # d(cosθ)/dr_k = r_ji/(ra*rb) - cosθ * r_jk/rb^2
            dcosth_drk = r_ji / (ra * rb) - cos_theta.unsqueeze(-1) * r_jk / (rb * rb)
            # d(cosθ)/dr_j = -(d/dr_i + d/dr_k)
            dcosth_drj = -(dcosth_dri + dcosth_drk)

            # F = -dE/dr = -dE/d(cosθ) * d(cosθ)/dr
            f_i = -dE_dcos.unsqueeze(-1) * dcosth_dri
            f_k = -dE_dcos.unsqueeze(-1) * dcosth_drk
            f_j = -dE_dcos.unsqueeze(-1) * dcosth_drj

            forces.scatter_add_(0, ai[:, 0:1].expand_as(f_i), f_i)
            forces.scatter_add_(0, ai[:, 1:2].expand_as(f_j), f_j)
            forces.scatter_add_(0, ai[:, 2:3].expand_as(f_k), f_k)

        # ---- Dihedral forces: Fourier series ----
        dp = self.dihedral_prior
        di = dp.dih_indices
        if di.shape[0] > 0:
            p0 = positions[di[:, 0]]
            p1 = positions[di[:, 1]]
            p2 = positions[di[:, 2]]
            p3 = positions[di[:, 3]]

            b1 = p1 - p0
            b2 = p2 - p1
            b3 = p3 - p2

            n1 = torch.cross(b1, b2, dim=-1)
            n2 = torch.cross(b2, b3, dim=-1)
            b2_len = torch.norm(b2, dim=-1, keepdim=True).clamp(min=1e-8)
            b2_hat = b2 / b2_len

            m = torch.cross(n1, b2_hat, dim=-1)
            x_val = (n1 * n2).sum(dim=-1)
            y_val = (m * n2).sum(dim=-1)
            phi = torch.atan2(y_val, x_val)

            # dE/dφ = Σ_n [-n*k1s_n*sin(nφ) + n*k2s_n*cos(nφ)]
            n_phi = phi.unsqueeze(-1) * dp.fourier_n.unsqueeze(0)
            dE_dphi = (dp.fourier_n.unsqueeze(0) * (
                -dp.k1s * torch.sin(n_phi) + dp.k2s * torch.cos(n_phi)
            )).sum(dim=-1)  # (n_dih,)

            # dφ/dr using Bekker/GROMACS convention
            n1_sq = (n1 * n1).sum(dim=-1, keepdim=True).clamp(min=1e-16)
            n2_sq = (n2 * n2).sum(dim=-1, keepdim=True).clamp(min=1e-16)
            b2_sq = (b2 * b2).sum(dim=-1, keepdim=True).clamp(min=1e-16)

            # c01 = b1·b2/|b2|^2, c32 = (-b3)·b2/|b2|^2
            c01 = (b1 * b2).sum(dim=-1, keepdim=True) / b2_sq
            c32 = (-b3 * b2).sum(dim=-1, keepdim=True) / b2_sq

            dphi_dp0 = b2_len * n1 / n1_sq
            dphi_dp3 = -b2_len * n2 / n2_sq
            dphi_dp1 = (c01 - 1.0) * dphi_dp0 - c32 * dphi_dp3
            dphi_dp2 = (c32 - 1.0) * dphi_dp3 - c01 * dphi_dp0

            f0 = -dE_dphi.unsqueeze(-1) * dphi_dp0
            f1 = -dE_dphi.unsqueeze(-1) * dphi_dp1
            f2 = -dE_dphi.unsqueeze(-1) * dphi_dp2
            f3 = -dE_dphi.unsqueeze(-1) * dphi_dp3

            forces.scatter_add_(0, di[:, 0:1].expand_as(f0), f0)
            forces.scatter_add_(0, di[:, 1:2].expand_as(f1), f1)
            forces.scatter_add_(0, di[:, 2:3].expand_as(f2), f2)
            forces.scatter_add_(0, di[:, 3:4].expand_as(f3), f3)

        # ---- Excluded volume forces: E = (σ/r)^12 * S(r) ----
        rp = self.repulsion_prior
        if rp.pair_i.shape[0] > 0:
            r_ij = positions[rp.pair_j] - positions[rp.pair_i]
            dist = torch.norm(r_ij, dim=-1).clamp(min=1e-8)
            mask = dist < rp.cutoff
            if mask.any():
                dist_m = dist[mask]
                sigma_m = rp.pair_sigma[mask]
                r_ij_m = r_ij[mask]
                r_hat_m = r_ij_m / dist_m.unsqueeze(-1)

                ratio = sigma_m / dist_m  # σ/r
                raw_e = ratio ** 12  # (σ/r)^12

                r_switch = rp.cutoff * 0.8
                switch = torch.ones_like(dist_m)
                dswitch = torch.zeros_like(dist_m)
                trans = (dist_m > r_switch) & (dist_m < rp.cutoff)
                if trans.any():
                    x_t = (dist_m[trans] - r_switch) / (rp.cutoff - r_switch)
                    switch[trans] = 0.5 * (1.0 + torch.cos(torch.pi * x_t))
                    dswitch[trans] = -0.5 * torch.pi / (rp.cutoff - r_switch) * torch.sin(torch.pi * x_t)

                # dE/dr = -12*(σ/r)^12/r * S(r) + (σ/r)^12 * dS/dr
                dE_dr = -12.0 * raw_e / dist_m * switch + raw_e * dswitch
                # F_j = -dE/dr * r_hat, F_i = dE/dr * r_hat
                f_rep = -dE_dr.unsqueeze(-1) * r_hat_m

                idx_j = rp.pair_j[mask].unsqueeze(-1).expand_as(f_rep)
                idx_i = rp.pair_i[mask].unsqueeze(-1).expand_as(f_rep)
                forces.scatter_add_(0, idx_j, f_rep)
                forces.scatter_add_(0, idx_i, -f_rep)

        return forces

    def compute_forces(
        self,
        positions_np: np.ndarray,
        types_np: Optional[np.ndarray] = None,
        cell_np: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Compute prior forces via autograd (NumPy API).

        Compatible with the existing CG training pipeline's
        ``compute_forces`` interface.

        Args:
            positions_np: (n_beads, 3) NumPy array of positions.
            types_np: Unused (topology already built). Kept for API compat.
            cell_np: Unused (no PBC). Kept for API compatibility.

        Returns:
            (n_beads, 3) NumPy array of prior forces F = -dE/dr.
        """
        device = next(self.parameters(), torch.empty(0)).device
        # Fallback: check buffers if no parameters
        if not list(self.parameters()):
            for buf in self.buffers():
                device = buf.device
                break

        pos = torch.tensor(
            positions_np, dtype=torch.float32, device=device, requires_grad=True
        )
        E = self.forward(pos)
        (grad,) = torch.autograd.grad(
            E, pos, create_graph=False, retain_graph=False
        )
        F = -grad
        return F.detach().cpu().numpy()

    def compute_energy(
        self,
        positions_np: np.ndarray,
        types_np: Optional[np.ndarray] = None,
        cell_np: Optional[np.ndarray] = None,
    ) -> float:
        """Compute prior energy (NumPy API).

        Args:
            positions_np: (n_beads, 3) NumPy array.
            types_np: Unused.
            cell_np: Unused.

        Returns:
            Scalar prior energy value.
        """
        device = next(self.parameters(), torch.empty(0)).device
        if not list(self.parameters()):
            for buf in self.buffers():
                device = buf.device
                break

        pos = torch.tensor(positions_np, dtype=torch.float32, device=device)
        with torch.no_grad():
            E = self.forward(pos)
        return float(E.item())

    def compute_forces_batch(
        self,
        positions_batch: np.ndarray,
    ) -> np.ndarray:
        """Compute prior forces for a batch of frames.

        Args:
            positions_batch: (n_frames, n_beads, 3) NumPy array.

        Returns:
            (n_frames, n_beads, 3) NumPy array of prior forces.
        """
        n_frames = positions_batch.shape[0]
        forces_batch = np.zeros_like(positions_batch)
        for i in range(n_frames):
            forces_batch[i] = self.compute_forces(positions_batch[i])
        return forces_batch

    @classmethod
    def from_charron2023(
        cls,
        params_path: str,
        bead_types: np.ndarray,
        repulsion_cutoff: float = 8.0,
    ) -> "ProteinCGPrior":
        """Factory method to create from Charron et al. 2023 parameters.

        Args:
            params_path: Path to ``prior_all_params.npz``.
            bead_types: 1D array of bead type IDs (1-24).
            repulsion_cutoff: Cutoff for excluded volume repulsion.

        Returns:
            Initialized ProteinCGPrior module.
        """
        return cls(params_path, bead_types, repulsion_cutoff)
