"""
Coarse-Grained (CG) Mapping Utilities for BAM-torch

This module provides functions to convert atomistic trajectories to
coarse-grained representations using various mapping schemes.

Supported mapping methods:
- Center of Mass (COM): CG site at the center of mass of atom group
- Center of Geometry (COG): CG site at the geometric center of atom group

Supported systems:
- Water (H2O): 3 atoms -> 1 bead
- Methane (CH4): 5 atoms -> 1 bead
- Ethanol (C2H5OH): 9 atoms -> 2 or 3 beads
- Benzene (C6H6): 12 atoms -> 1 or 6 beads
- Amino acids: Various mappings
- Custom user-defined mappings
- Auto-detection from trajectory
"""

import numpy as np
import torch
from typing import Dict, List, Optional, Tuple, Union
from ase import Atoms
from ase.io import read
from collections import Counter


# Atomic masses for common elements (amu)
ATOMIC_MASSES = {
    1: 1.008,    # H
    6: 12.011,   # C
    7: 14.007,   # N
    8: 15.999,   # O
    9: 18.998,   # F
    15: 30.974,  # P
    16: 32.065,  # S
    17: 35.453,  # Cl
    35: 79.904,  # Br
    53: 126.90,  # I
}

# Element symbols for auto-detection
ELEMENT_SYMBOLS = {
    1: 'H', 6: 'C', 7: 'N', 8: 'O', 9: 'F',
    15: 'P', 16: 'S', 17: 'Cl', 35: 'Br', 53: 'I'
}


# =============================================================================
# Predefined CG Mapping Presets
# =============================================================================

CG_PRESETS = {
    # Water: H2O -> 1 bead (W)
    'water': {
        'method': 'com',
        'atoms_per_molecule': 3,
        'formula': 'H2O',
        'description': 'Water molecule: 3 atoms -> 1 CG bead at COM',
        'beads': [
            {'name': 'W', 'type_id': 0, 'atom_indices': [0, 1, 2]}
        ]
    },

    # Methane: CH4 -> 1 bead (M)
    'methane': {
        'method': 'com',
        'atoms_per_molecule': 5,
        'formula': 'CH4',
        'description': 'Methane molecule: 5 atoms -> 1 CG bead at COM',
        'beads': [
            {'name': 'M', 'type_id': 0, 'atom_indices': [0, 1, 2, 3, 4]}
        ]
    },

    # Ethane: C2H6 -> 2 beads (one per CH3)
    'ethane': {
        'method': 'com',
        'atoms_per_molecule': 8,
        'formula': 'C2H6',
        'description': 'Ethane molecule: 8 atoms -> 2 CG beads (CH3 groups)',
        'beads': [
            {'name': 'C1', 'type_id': 0, 'atom_indices': [0, 2, 3, 4]},      # CH3
            {'name': 'C2', 'type_id': 0, 'atom_indices': [1, 5, 6, 7]}       # CH3
        ]
    },

    # Propane: C3H8 -> 3 beads
    'propane': {
        'method': 'com',
        'atoms_per_molecule': 11,
        'formula': 'C3H8',
        'description': 'Propane molecule: 11 atoms -> 3 CG beads',
        'beads': [
            {'name': 'C1', 'type_id': 0, 'atom_indices': [0, 3, 4, 5]},      # CH3
            {'name': 'C2', 'type_id': 1, 'atom_indices': [1, 6, 7]},         # CH2
            {'name': 'C3', 'type_id': 0, 'atom_indices': [2, 8, 9, 10]}      # CH3
        ]
    },

    # Butane: C4H10 -> 4 beads
    'butane': {
        'method': 'com',
        'atoms_per_molecule': 14,
        'formula': 'C4H10',
        'description': 'Butane molecule: 14 atoms -> 4 CG beads',
        'beads': [
            {'name': 'C1', 'type_id': 0, 'atom_indices': [0, 4, 5, 6]},      # CH3
            {'name': 'C2', 'type_id': 1, 'atom_indices': [1, 7, 8]},         # CH2
            {'name': 'C3', 'type_id': 1, 'atom_indices': [2, 9, 10]},        # CH2
            {'name': 'C4', 'type_id': 0, 'atom_indices': [3, 11, 12, 13]}    # CH3
        ]
    },

    # Benzene: C6H6 -> 1 bead (whole ring) or 6 beads
    'benzene': {
        'method': 'com',
        'atoms_per_molecule': 12,
        'formula': 'C6H6',
        'description': 'Benzene molecule: 12 atoms -> 1 CG bead (ring center)',
        'beads': [
            {'name': 'BZ', 'type_id': 0, 'atom_indices': list(range(12))}
        ]
    },

    # Benzene with 6 beads (one per CH group)
    'benzene_6bead': {
        'method': 'com',
        'atoms_per_molecule': 12,
        'formula': 'C6H6',
        'description': 'Benzene molecule: 12 atoms -> 6 CG beads (CH groups)',
        'beads': [
            {'name': 'B1', 'type_id': 0, 'atom_indices': [0, 6]},   # CH
            {'name': 'B2', 'type_id': 0, 'atom_indices': [1, 7]},   # CH
            {'name': 'B3', 'type_id': 0, 'atom_indices': [2, 8]},   # CH
            {'name': 'B4', 'type_id': 0, 'atom_indices': [3, 9]},   # CH
            {'name': 'B5', 'type_id': 0, 'atom_indices': [4, 10]},  # CH
            {'name': 'B6', 'type_id': 0, 'atom_indices': [5, 11]}   # CH
        ]
    },

    # Methanol: CH3OH -> 2 beads
    'methanol': {
        'method': 'com',
        'atoms_per_molecule': 6,
        'formula': 'CH4O',
        'description': 'Methanol molecule: 6 atoms -> 2 CG beads (CH3 + OH)',
        'beads': [
            {'name': 'ME', 'type_id': 0, 'atom_indices': [0, 2, 3, 4]},  # CH3
            {'name': 'OH', 'type_id': 1, 'atom_indices': [1, 5]}         # OH
        ]
    },

    # Ethanol: C2H5OH -> 3 beads
    'ethanol': {
        'method': 'com',
        'atoms_per_molecule': 9,
        'formula': 'C2H6O',
        'description': 'Ethanol molecule: 9 atoms -> 3 CG beads (CH3 + CH2 + OH)',
        'beads': [
            {'name': 'C1', 'type_id': 0, 'atom_indices': [0, 3, 4, 5]},  # CH3
            {'name': 'C2', 'type_id': 1, 'atom_indices': [1, 6, 7]},     # CH2
            {'name': 'OH', 'type_id': 2, 'atom_indices': [2, 8]}         # OH
        ]
    },

    # Carbon dioxide: CO2 -> 1 bead
    'co2': {
        'method': 'com',
        'atoms_per_molecule': 3,
        'formula': 'CO2',
        'description': 'Carbon dioxide: 3 atoms -> 1 CG bead',
        'beads': [
            {'name': 'CO2', 'type_id': 0, 'atom_indices': [0, 1, 2]}
        ]
    },

    # Ammonia: NH3 -> 1 bead
    'ammonia': {
        'method': 'com',
        'atoms_per_molecule': 4,
        'formula': 'NH3',
        'description': 'Ammonia molecule: 4 atoms -> 1 CG bead',
        'beads': [
            {'name': 'NH3', 'type_id': 0, 'atom_indices': [0, 1, 2, 3]}
        ]
    },

    # Acetone: C3H6O -> 3 beads
    'acetone': {
        'method': 'com',
        'atoms_per_molecule': 10,
        'formula': 'C3H6O',
        'description': 'Acetone molecule: 10 atoms -> 3 CG beads',
        'beads': [
            {'name': 'C1', 'type_id': 0, 'atom_indices': [0, 3, 4, 5]},  # CH3
            {'name': 'CO', 'type_id': 1, 'atom_indices': [1, 6]},        # C=O
            {'name': 'C2', 'type_id': 0, 'atom_indices': [2, 7, 8, 9]}   # CH3
        ]
    },

    # DMSO: C2H6SO -> 3 beads
    'dmso': {
        'method': 'com',
        'atoms_per_molecule': 10,
        'formula': 'C2H6OS',
        'description': 'DMSO molecule: 10 atoms -> 3 CG beads',
        'beads': [
            {'name': 'C1', 'type_id': 0, 'atom_indices': [0, 3, 4, 5]},  # CH3
            {'name': 'SO', 'type_id': 1, 'atom_indices': [1, 2]},        # S=O
            {'name': 'C2', 'type_id': 0, 'atom_indices': [6, 7, 8, 9]}   # CH3
        ]
    },

    # ==========================================================================
    # Protein CG Mappings (Cα representation)
    # ==========================================================================

    # Chignolin: 10-residue mini-protein (Cα mapping)
    # Sequence: GLY-TYR-ASP-PRO-GLU-THR-GLY-THR-TRP-GLY
    'chignolin': {
        'method': 'calpha',
        'atoms_per_molecule': None,  # Variable (full protein)
        'formula': 'Chignolin',
        'n_residues': 10,
        'description': 'Chignolin mini-protein: 10 residues -> 10 Cα beads',
        'sequence': 'GYDPETGTWG',
        'beads': [
            {'name': 'GLY1', 'type_id': 0, 'residue': 'GLY'},
            {'name': 'TYR2', 'type_id': 1, 'residue': 'TYR'},
            {'name': 'ASP3', 'type_id': 2, 'residue': 'ASP'},
            {'name': 'PRO4', 'type_id': 3, 'residue': 'PRO'},
            {'name': 'GLU5', 'type_id': 4, 'residue': 'GLU'},
            {'name': 'THR6', 'type_id': 5, 'residue': 'THR'},
            {'name': 'GLY7', 'type_id': 0, 'residue': 'GLY'},
            {'name': 'THR8', 'type_id': 5, 'residue': 'THR'},
            {'name': 'TRP9', 'type_id': 6, 'residue': 'TRP'},
            {'name': 'GLY10', 'type_id': 0, 'residue': 'GLY'},
        ],
        'note': 'For proteins, use MDTraj/MDAnalysis to extract Cα positions'
    },
}


class CGMapping:
    """
    Coarse-Grained Mapping class for converting atomistic data to CG representation.

    Supports:
    - Predefined presets (water, methane, ethanol, etc.)
    - Custom user-defined mappings
    - Auto-detection from trajectory
    - Multiple bead types within a molecule
    - COM (center of mass) or COG (center of geometry) methods

    Attributes:
        mapping_config: Dictionary containing mapping definitions
        method: Mapping method ('com' or 'cog')
        bead_definitions: List of bead definitions with atom indices
        num_cg_sites: Number of CG sites per molecule
    """

    def __init__(self, mapping_config: Union[Dict, str]):
        """
        Initialize CG mapping.

        Args:
            mapping_config: Either a preset name (str) or dictionary with keys:
                - method: 'com' (center of mass) or 'cog' (center of geometry)
                - atoms_per_molecule: Number of atoms per molecule
                - beads: List of bead definitions, each with:
                    - name: Bead name (e.g., 'W' for water)
                    - type_id: Integer type ID for the bead
                    - atom_indices: List of atom indices within molecule (0-indexed)
        """
        # Handle preset names
        if isinstance(mapping_config, str):
            if mapping_config.lower() in CG_PRESETS:
                mapping_config = CG_PRESETS[mapping_config.lower()].copy()
            else:
                raise ValueError(f"Unknown preset: {mapping_config}. "
                               f"Available: {list(CG_PRESETS.keys())}")

        self.method = mapping_config.get('method', 'com')
        self.atoms_per_molecule = mapping_config['atoms_per_molecule']
        self.beads = mapping_config['beads']
        self.num_cg_sites = len(self.beads)
        self.formula = mapping_config.get('formula', 'Unknown')
        self.description = mapping_config.get('description', '')

        # Store original config
        self.mapping_config = mapping_config

        # Build atom index to bead mapping
        self._build_mapping()

        # Validate mapping
        self._validate_mapping()

    def _build_mapping(self):
        """Build internal mapping structures."""
        self.bead_atom_indices = []
        self.bead_types = []
        self.bead_names = []

        for bead in self.beads:
            self.bead_atom_indices.append(bead['atom_indices'])
            self.bead_types.append(bead['type_id'])
            self.bead_names.append(bead['name'])

        # Count unique bead types
        self.num_bead_types = len(set(self.bead_types))

    def _validate_mapping(self):
        """Validate that the mapping is consistent."""
        # Check all atom indices are within range
        all_indices = []
        for indices in self.bead_atom_indices:
            for idx in indices:
                if idx < 0 or idx >= self.atoms_per_molecule:
                    raise ValueError(f"Atom index {idx} out of range "
                                   f"[0, {self.atoms_per_molecule})")
                all_indices.append(idx)

        # Check for duplicate assignments (optional - some mappings may want overlap)
        if len(all_indices) != len(set(all_indices)):
            print("Warning: Some atoms are assigned to multiple beads")

    def get_num_molecules(self, n_atoms: int) -> int:
        """Calculate number of molecules from total atom count."""
        if n_atoms % self.atoms_per_molecule != 0:
            raise ValueError(
                f"Total atoms {n_atoms} not divisible by atoms_per_molecule "
                f"{self.atoms_per_molecule}. Check your mapping configuration."
            )
        return n_atoms // self.atoms_per_molecule

    def get_num_cg_sites(self, n_atoms: int) -> int:
        """Calculate total number of CG sites from total atom count."""
        n_molecules = self.get_num_molecules(n_atoms)
        return n_molecules * self.num_cg_sites

    def atomistic_to_cg_positions(
        self,
        positions: np.ndarray,
        atomic_numbers: np.ndarray,
        cell: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Convert atomistic positions to CG positions.

        Args:
            positions: Atomistic positions, shape (n_atoms, 3)
            atomic_numbers: Atomic numbers, shape (n_atoms,)
            cell: Unit cell for PBC handling, shape (3, 3) or None

        Returns:
            CG positions, shape (n_cg_sites, 3)
        """
        n_atoms = len(positions)
        n_molecules = self.get_num_molecules(n_atoms)
        n_cg_sites = n_molecules * self.num_cg_sites

        cg_positions = np.zeros((n_cg_sites, 3), dtype=np.float64)

        for mol_idx in range(n_molecules):
            mol_start = mol_idx * self.atoms_per_molecule

            for bead_idx, atom_indices in enumerate(self.bead_atom_indices):
                # Global atom indices for this bead in this molecule
                global_indices = [mol_start + i for i in atom_indices]

                # Get positions of atoms in this bead
                bead_positions = positions[global_indices]

                # Handle PBC: use minimum image convention relative to first atom
                if cell is not None and len(bead_positions) > 1:
                    bead_positions = self._apply_minimum_image(
                        bead_positions, cell
                    )

                # Calculate CG position
                if self.method == 'com':
                    # Center of mass
                    masses = []
                    for i in global_indices:
                        Z = atomic_numbers[i]
                        if Z not in ATOMIC_MASSES:
                            raise KeyError(
                                f"Atomic number {Z} not found in ATOMIC_MASSES. "
                                f"Please add it to ATOMIC_MASSES dictionary in cg_mapping.py"
                            )
                        masses.append(ATOMIC_MASSES[Z])
                    masses = np.array(masses)
                    total_mass = masses.sum()
                    cg_pos = np.sum(bead_positions * masses[:, np.newaxis], axis=0) / total_mass
                else:  # cog
                    # Center of geometry
                    cg_pos = np.mean(bead_positions, axis=0)

                # Store CG position
                cg_site_idx = mol_idx * self.num_cg_sites + bead_idx
                cg_positions[cg_site_idx] = cg_pos

        return cg_positions

    def _apply_minimum_image(
        self,
        positions: np.ndarray,
        cell: np.ndarray
    ) -> np.ndarray:
        """
        Apply minimum image convention relative to first atom.

        Args:
            positions: Positions of atoms in a bead, shape (n_atoms_in_bead, 3)
            cell: Unit cell, shape (3, 3)

        Returns:
            Unwrapped positions relative to first atom
        """
        # Use first atom as reference
        ref_pos = positions[0]
        unwrapped = np.zeros_like(positions)
        unwrapped[0] = ref_pos

        # Inverse cell for fractional coordinates
        try:
            cell_inv = np.linalg.inv(cell)
        except np.linalg.LinAlgError:
            # Non-periodic or singular cell
            return positions

        for i in range(1, len(positions)):
            diff = positions[i] - ref_pos
            # Convert to fractional coordinates
            frac_diff = diff @ cell_inv
            # Apply minimum image
            frac_diff = frac_diff - np.round(frac_diff)
            # Convert back to Cartesian
            unwrapped[i] = ref_pos + frac_diff @ cell

        return unwrapped

    def atomistic_to_cg_forces(
        self,
        forces: np.ndarray,
        atomic_numbers: np.ndarray
    ) -> np.ndarray:
        """
        Convert atomistic forces to CG forces using force matching.

        For bottom-up CG, the CG force on a bead is the sum of atomistic forces
        on all atoms belonging to that bead.

        F_CG = sum(F_atom) for all atoms in the bead

        Args:
            forces: Atomistic forces, shape (n_atoms, 3)
            atomic_numbers: Atomic numbers, shape (n_atoms,)

        Returns:
            CG forces, shape (n_cg_sites, 3)
        """
        n_atoms = len(forces)
        n_molecules = self.get_num_molecules(n_atoms)
        n_cg_sites = n_molecules * self.num_cg_sites

        cg_forces = np.zeros((n_cg_sites, 3), dtype=np.float64)

        for mol_idx in range(n_molecules):
            mol_start = mol_idx * self.atoms_per_molecule

            for bead_idx, atom_indices in enumerate(self.bead_atom_indices):
                # Global atom indices for this bead in this molecule
                global_indices = [mol_start + i for i in atom_indices]

                # Sum forces on all atoms in this bead
                cg_force = np.sum(forces[global_indices], axis=0)

                # Store CG force
                cg_site_idx = mol_idx * self.num_cg_sites + bead_idx
                cg_forces[cg_site_idx] = cg_force

        return cg_forces

    def get_cg_types(self, n_atoms: int) -> np.ndarray:
        """
        Get CG bead types for all CG sites.

        Args:
            n_atoms: Total number of atomistic atoms

        Returns:
            CG types, shape (n_cg_sites,)
        """
        n_molecules = self.get_num_molecules(n_atoms)
        n_cg_sites = n_molecules * self.num_cg_sites

        cg_types = np.zeros(n_cg_sites, dtype=np.int64)

        for mol_idx in range(n_molecules):
            for bead_idx, type_id in enumerate(self.bead_types):
                cg_site_idx = mol_idx * self.num_cg_sites + bead_idx
                cg_types[cg_site_idx] = type_id

        return cg_types

    def convert_atoms_to_cg(self, atoms: Atoms) -> Dict:
        """
        Convert an ASE Atoms object to CG representation.

        Args:
            atoms: ASE Atoms object with atomistic data

        Returns:
            Dictionary with CG data:
                - positions: CG positions
                - forces: CG forces (if available)
                - energy: Total energy (unchanged)
                - types: CG bead types
                - cell: Unit cell (unchanged)
        """
        positions = atoms.get_positions()
        atomic_numbers = atoms.get_atomic_numbers()
        cell = np.array(atoms.get_cell())

        cg_data = {
            'positions': self.atomistic_to_cg_positions(positions, atomic_numbers, cell),
            'types': self.get_cg_types(len(positions)),
            'cell': cell,
            'energy': atoms.get_potential_energy() if atoms.calc else 0.0,
        }

        if atoms.calc and 'forces' in atoms.calc.results:
            cg_data['forces'] = self.atomistic_to_cg_forces(
                atoms.get_forces(), atomic_numbers
            )
        else:
            cg_data['forces'] = np.zeros((len(cg_data['positions']), 3))

        return cg_data

    def __repr__(self):
        return (f"CGMapping(formula={self.formula}, method={self.method}, "
                f"atoms_per_mol={self.atoms_per_molecule}, beads={self.num_cg_sites}, "
                f"bead_types={self.num_bead_types})")


# =============================================================================
# Preset Creation Functions (for backward compatibility)
# =============================================================================

def create_water_cg_mapping() -> Dict:
    """Create a CG mapping configuration for water molecules."""
    return CG_PRESETS['water'].copy()


def create_methane_cg_mapping() -> Dict:
    """Create a CG mapping configuration for methane molecules."""
    return CG_PRESETS['methane'].copy()


def create_ethanol_cg_mapping() -> Dict:
    """Create a CG mapping configuration for ethanol molecules."""
    return CG_PRESETS['ethanol'].copy()


def create_benzene_cg_mapping(n_beads: int = 1) -> Dict:
    """Create a CG mapping configuration for benzene molecules."""
    if n_beads == 6:
        return CG_PRESETS['benzene_6bead'].copy()
    return CG_PRESETS['benzene'].copy()


# =============================================================================
# Auto-detection Functions
# =============================================================================

def detect_molecule_from_atoms(atoms: Atoms) -> Optional[str]:
    """
    Try to auto-detect the molecule type from an ASE Atoms object.

    Args:
        atoms: ASE Atoms object

    Returns:
        Preset name if detected, None otherwise
    """
    formula = atoms.get_chemical_formula(mode='hill')
    n_atoms = len(atoms)

    # Try to match formula with presets
    formula_to_preset = {
        'H2O': 'water',
        'OH2': 'water',
        'CH4': 'methane',
        'H4C': 'methane',
        'C2H6': 'ethane',
        'H6C2': 'ethane',
        'C6H6': 'benzene',
        'H6C6': 'benzene',
        'CH4O': 'methanol',
        'CH3OH': 'methanol',
        'C2H6O': 'ethanol',
        'C2H5OH': 'ethanol',
        'CO2': 'co2',
        'NH3': 'ammonia',
        'H3N': 'ammonia',
    }

    # Check if formula matches any preset (accounting for multiples)
    for preset_formula, preset_name in formula_to_preset.items():
        preset_atoms = CG_PRESETS[preset_name]['atoms_per_molecule']
        if n_atoms % preset_atoms == 0:
            # Check element counts
            n_mol = n_atoms // preset_atoms
            # This is a simple heuristic; could be improved
            if formula == preset_formula or n_atoms == preset_atoms:
                return preset_name

    return None


def auto_detect_cg_mapping(traj: List[Atoms]) -> Dict:
    """
    Automatically detect and create CG mapping from trajectory.

    Args:
        traj: List of ASE Atoms objects

    Returns:
        CG mapping configuration dictionary
    """
    if len(traj) == 0:
        raise ValueError("Empty trajectory")

    atoms = traj[0]
    preset = detect_molecule_from_atoms(atoms)

    if preset:
        print(f"Auto-detected molecule type: {preset}")
        return CG_PRESETS[preset].copy()

    # If no preset matches, create a simple 1-molecule-1-bead mapping
    print("Could not auto-detect molecule type. Using single-bead mapping.")
    n_atoms = len(atoms)

    return {
        'method': 'com',
        'atoms_per_molecule': n_atoms,
        'formula': atoms.get_chemical_formula(mode='hill'),
        'description': f'Auto-generated: {n_atoms} atoms -> 1 CG bead',
        'beads': [
            {'name': 'X', 'type_id': 0, 'atom_indices': list(range(n_atoms))}
        ]
    }


def create_custom_cg_mapping(
    atoms_per_molecule: int,
    bead_definitions: List[Dict],
    method: str = 'com',
    formula: str = 'Custom',
    description: str = ''
) -> Dict:
    """
    Create a custom CG mapping configuration.

    Args:
        atoms_per_molecule: Number of atoms per molecule
        bead_definitions: List of bead definitions, each with:
            - name: Bead name
            - type_id: Integer type ID
            - atom_indices: List of atom indices within molecule
        method: 'com' or 'cog'
        formula: Chemical formula (for reference)
        description: Description of the mapping

    Returns:
        CG mapping configuration dictionary

    Example:
        >>> # Create custom mapping for a polymer repeat unit
        >>> mapping = create_custom_cg_mapping(
        ...     atoms_per_molecule=20,
        ...     bead_definitions=[
        ...         {'name': 'BB', 'type_id': 0, 'atom_indices': [0,1,2,3,4]},  # backbone
        ...         {'name': 'SC', 'type_id': 1, 'atom_indices': [5,6,7,8,9]},  # sidechain
        ...     ],
        ...     method='com',
        ...     formula='C10H20',
        ...     description='Polymer repeat unit'
        ... )
    """
    return {
        'method': method,
        'atoms_per_molecule': atoms_per_molecule,
        'formula': formula,
        'description': description,
        'beads': bead_definitions
    }


# =============================================================================
# Trajectory Conversion
# =============================================================================

def convert_trajectory_to_cg(
    traj: List[Atoms],
    mapping: CGMapping,
    show_progress: bool = False
) -> List[Dict]:
    """
    Convert an entire trajectory to CG representation.

    Args:
        traj: List of ASE Atoms objects
        mapping: CGMapping object
        show_progress: Whether to show progress bar

    Returns:
        List of dictionaries with CG data for each frame
    """
    from tqdm import tqdm

    iterator = tqdm(traj, desc="Converting to CG") if show_progress else traj
    cg_traj = []

    for atoms in iterator:
        cg_data = mapping.convert_atoms_to_cg(atoms)
        cg_traj.append(cg_data)

    return cg_traj


def get_available_presets() -> Dict[str, str]:
    """
    Get all available CG mapping presets with descriptions.

    Returns:
        Dictionary of preset names and descriptions
    """
    return {name: preset.get('description', '') for name, preset in CG_PRESETS.items()}


def print_available_presets():
    """Print all available CG mapping presets."""
    print("\n" + "="*60)
    print("Available CG Mapping Presets")
    print("="*60)
    for name, preset in CG_PRESETS.items():
        desc = preset.get('description', '')
        formula = preset.get('formula', '')
        n_atoms = preset.get('atoms_per_molecule', 0)
        n_beads = len(preset.get('beads', []))
        print(f"\n  {name}:")
        print(f"    Formula: {formula}")
        print(f"    Atoms per molecule: {n_atoms}")
        print(f"    CG beads: {n_beads}")
        print(f"    Description: {desc}")
    print("\n" + "="*60)


# =============================================================================
# Main (for testing)
# =============================================================================

if __name__ == "__main__":
    print_available_presets()

    # Test water mapping
    print("\n\nTesting water CG mapping...")
    mapping = CGMapping('water')
    print(mapping)

    # Test with dummy water data
    n_waters = 10
    n_atoms = n_waters * 3

    positions = np.zeros((n_atoms, 3))
    atomic_numbers = np.zeros(n_atoms, dtype=int)

    for i in range(n_waters):
        idx = i * 3
        positions[idx] = [i * 3, 0, 0]      # O
        positions[idx + 1] = [i * 3 + 0.96, 0.27, 0]  # H
        positions[idx + 2] = [i * 3 - 0.24, 0.93, 0]  # H
        atomic_numbers[idx] = 8     # O
        atomic_numbers[idx + 1] = 1  # H
        atomic_numbers[idx + 2] = 1  # H

    cg_positions = mapping.atomistic_to_cg_positions(positions, atomic_numbers)
    cg_types = mapping.get_cg_types(n_atoms)

    print(f"Atomistic atoms: {n_atoms}")
    print(f"CG sites: {len(cg_positions)}")
    print(f"CG types: {cg_types}")
    print("Water CG mapping test passed!")

    # Test ethanol mapping
    print("\n\nTesting ethanol CG mapping...")
    mapping_eth = CGMapping('ethanol')
    print(mapping_eth)
    print(f"Ethanol has {mapping_eth.num_bead_types} unique bead types")
    print("Ethanol CG mapping test passed!")
