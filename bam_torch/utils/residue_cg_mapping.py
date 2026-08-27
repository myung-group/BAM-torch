"""
Generic residue-based coarse-grained mapper using MDAnalysis.

Supports heterogeneous systems with multiple molecule types (lipids, proteins,
DNA, RNA, water). Uses MARTINI-style mappings from martini_mappings.py.

Position mapping: Center of Mass (COM)
Force mapping: Sum of atomic forces
Energy: System-level (not decomposable per bead)

Requires MDAnalysis (optional dependency, lazy-imported).
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Union


# Unit conversion constants (GROMACS -> BAM-torch)
KJ_MOL_TO_EV = 0.01036427230133       # 1 kJ/mol = 0.01036 eV
NM_TO_ANGSTROM = 10.0                  # 1 nm = 10 Å
# Force: kJ/(mol*nm) -> eV/Å: multiply by KJ_MOL_TO_EV (since 1 kJ/(mol*nm) = 0.01036 eV/Å)
FORCE_CONVERSION = KJ_MOL_TO_EV


def _lazy_import_mda():
    """Lazy import MDAnalysis with clear error message."""
    try:
        import MDAnalysis as mda
        return mda
    except ImportError:
        raise ImportError(
            "MDAnalysis is required for residue-based CG mapping.\n"
            "Install with: pip install MDAnalysis\n"
            "Or: conda install -c conda-forge mdanalysis"
        )


def compute_com(positions: np.ndarray, masses: np.ndarray,
                box: Optional[np.ndarray] = None) -> np.ndarray:
    """Compute center of mass with PBC-aware unwrapping.

    For periodic systems, atoms in the same bead may be wrapped to
    opposite sides of the box. Uses the angular coordinate method:
    map each coordinate to an angle on a circle, average in angular
    space, then convert back.

    Reference: Bai & Breen, J. Mol. Graph. Modelling 26, 1315 (2008)

    Args:
        positions: (n_atoms, 3) atomic positions in Angstrom
        masses: (n_atoms,) atomic masses
        box: (3,) box dimensions [Lx, Ly, Lz] or None for non-periodic

    Returns:
        com: (3,) center of mass position
    """
    total_mass = masses.sum()
    if total_mass == 0:
        raise ValueError("Total mass is zero.")

    if box is None or np.all(box < 1e-6):
        # Non-periodic: standard COM
        return (positions * masses[:, np.newaxis]).sum(axis=0) / total_mass

    # PBC-aware COM using angular coordinate method
    com = np.zeros(3)
    for dim in range(3):
        L = box[dim]
        # Map positions to angles: theta = 2*pi*x/L
        theta = 2.0 * np.pi * positions[:, dim] / L
        # Mass-weighted average of sin and cos
        xi = np.sum(masses * np.cos(theta)) / total_mass
        zeta = np.sum(masses * np.sin(theta)) / total_mass
        # Convert back to position
        theta_avg = np.arctan2(-zeta, -xi) + np.pi
        com[dim] = L * theta_avg / (2.0 * np.pi)

    return com


def compute_cg_force(forces: np.ndarray) -> np.ndarray:
    """Compute CG force as sum of atomic forces."""
    return forces.sum(axis=0)


class ResidueBasedCGMapper:
    """Generic MARTINI-style CG mapper for any combination of molecule types.

    Works with MDAnalysis Universe objects. Automatically detects residue types,
    looks up mappings from the MARTINI registry, and performs COM-based mapping.

    Supports heterogeneous systems (e.g., lipid bilayer + water + protein).

    Args:
        universe: MDAnalysis Universe loaded from topology + trajectory
        residue_names: List of residue names to map (e.g., ['DLIPC', 'SOL']).
            If None, auto-detects all residues with known MARTINI mappings.
        custom_mappings: Dict of custom mappings to override registry entries.
            Format: {'RESNAME': {'bead_order': [...], 'atom_names': {...}}}
        include_water: Whether to include water molecules as W beads.
        waters_per_bead: Number of water molecules per W bead (default: 4).
        ignore_residues: List of residue names to skip (e.g., ['NA', 'CL']).

    Example:
        >>> u = mda.Universe('step7_1.gro', 'rerun.trr')
        >>> mapper = ResidueBasedCGMapper(u, residue_names=['DLIPC'],
        ...                               include_water=True)
        >>> pos, frc, types = mapper.map_frame(u.trajectory[0].positions,
        ...                                     u.trajectory[0].forces)
    """

    # Common water residue names
    WATER_RESNAMES = {'SOL', 'WAT', 'HOH', 'TIP3', 'TIP4', 'SPC', 'SPCE', 'TIP3P'}

    def __init__(
        self,
        universe,
        residue_names: Optional[List[str]] = None,
        custom_mappings: Optional[Dict] = None,
        include_water: bool = False,
        waters_per_bead: int = 4,
        ignore_residues: Optional[List[str]] = None,
    ):
        from .martini_mappings import MARTINI_MAPPINGS, get_mass_from_name

        self.u = universe
        self.include_water = include_water
        self.waters_per_bead = waters_per_bead
        self.ignore_residues = set(ignore_residues or [])
        self._get_mass_from_name = get_mass_from_name

        # Merge registry with custom mappings
        self.mappings = dict(MARTINI_MAPPINGS)
        if custom_mappings:
            self.mappings.update(custom_mappings)

        # Detect or validate residue names
        if residue_names is None:
            self._residue_config = self._detect_residues()
        else:
            self._residue_config = self._validate_residues(residue_names)

        # Separate molecule residues from water residues
        self._molecule_resnames = []
        self._water_resname = None

        for resname, count in self._residue_config.items():
            mapping = self.mappings.get(resname, {})
            if mapping.get('type') == 'water_cluster':
                if include_water:
                    self._water_resname = resname
            elif mapping.get('type') == 'ion':
                continue
            else:
                self._molecule_resnames.append(resname)

        # Build global bead type map
        self._global_type_map, self._bead_type_names = self._build_global_type_map()

        # Build per-residue mapping
        self._build_residue_mapping()

        # Build water mapper if needed
        self._water_mapper = None
        if include_water and self._water_resname is not None:
            self._build_water_mapping()

        # Build bond topology from AA bonds + CG mapping
        self._bond_topology = self._build_bond_topology()

        # Calculate total beads
        self._n_molecule_beads = sum(
            len(info['beads']) for info in self._residue_bead_info
        )
        self._n_water_beads = 0
        if self._water_mapper is not None:
            self._n_water_beads = self._water_mapper['n_water_beads']

        self._n_total_beads = self._n_molecule_beads + self._n_water_beads

        # Cache types array (same for all frames)
        self._types_array = self._compute_types_array()

        # Print summary
        self._print_summary()

    def _detect_residues(self) -> Dict[str, int]:
        """Auto-detect residue types present in the system."""
        resname_counts = {}
        for res in self.u.residues:
            rn = res.resname.strip()
            if rn in self.ignore_residues:
                continue
            if rn in self.mappings:
                resname_counts[rn] = resname_counts.get(rn, 0) + 1

        if not resname_counts:
            all_resnames = set(res.resname.strip() for res in self.u.residues)
            raise ValueError(
                f"No known MARTINI mappings found for residues: {all_resnames}\n"
                f"Available mappings: {list(self.mappings.keys())}\n"
                f"Use custom_mappings parameter to define mappings for your system."
            )

        return resname_counts

    def _validate_residues(self, residue_names: List[str]) -> Dict[str, int]:
        """Validate requested residue names exist in the system."""
        resname_counts = {}
        for resname in residue_names:
            selection = self.u.select_atoms(f'resname {resname}')
            if len(selection) == 0:
                print(f"  Warning: No atoms found for residue '{resname}'")
                continue
            n_res = len(selection.residues)
            resname_counts[resname] = n_res

        # Auto-detect water residues if include_water is True
        if self.include_water:
            for res in self.u.residues:
                rn = res.resname.strip()
                if rn in self.WATER_RESNAMES and rn not in resname_counts:
                    selection = self.u.select_atoms(f'resname {rn}')
                    resname_counts[rn] = len(selection.residues)
                    break  # Only need one water type

        return resname_counts

    def _build_global_type_map(self) -> Tuple[Dict, Dict]:
        """Assign globally unique bead type IDs across all residue types.

        Returns:
            global_type_map: {(resname, bead_name): global_type_id}
            bead_type_names: {global_type_id: bead_name}
        """
        global_type_map = {}
        bead_type_names = {}
        current_id = 0

        # Molecule beads first
        for resname in self._molecule_resnames:
            mapping = self.mappings[resname]
            for bead_name in mapping['bead_order']:
                key = (resname, bead_name)
                if key not in global_type_map:
                    global_type_map[key] = current_id
                    bead_type_names[current_id] = bead_name
                    current_id += 1

        # Water bead last
        if self.include_water and self._water_resname is not None:
            global_type_map[(self._water_resname, 'W')] = current_id
            bead_type_names[current_id] = 'W'
            current_id += 1

        return global_type_map, bead_type_names

    def _build_residue_mapping(self):
        """Build atom-index-to-bead mapping for each molecule residue."""
        self._residue_bead_info = []

        for resname in self._molecule_resnames:
            mapping = self.mappings[resname]
            residues = self.u.select_atoms(f'resname {resname}').residues

            for res in residues:
                beads = []
                for bead_name in mapping['bead_order']:
                    atom_names = mapping['atom_names'][bead_name]

                    indices = []
                    masses = []
                    for atom in res.atoms:
                        if atom.name in atom_names:
                            indices.append(atom.index)
                            masses.append(self._get_mass_from_name(atom.name))

                    if len(indices) == 0:
                        print(f"  Warning: No atoms matched for bead '{bead_name}' "
                              f"in residue {res.resname} {res.resid}")

                    global_type = self._global_type_map[(resname, bead_name)]
                    beads.append({
                        'bead_name': bead_name,
                        'global_type': global_type,
                        'atom_indices': np.array(indices, dtype=np.int64),
                        'masses': np.array(masses, dtype=np.float64),
                    })

                self._residue_bead_info.append({
                    'resname': resname,
                    'resid': res.resid,
                    'beads': beads,
                })

    def _build_water_mapping(self):
        """Build water molecule info for spatial clustering."""
        resname = self._water_resname

        # Try multiple water residue names
        water_atoms = None
        actual_resname = resname
        if resname in self.WATER_RESNAMES:
            for wname in self.WATER_RESNAMES:
                sel = self.u.select_atoms(f'resname {wname}')
                if len(sel) > 0:
                    water_atoms = sel
                    actual_resname = wname
                    break
        else:
            water_atoms = self.u.select_atoms(f'resname {resname}')

        if water_atoms is None or len(water_atoms) == 0:
            print("  No water molecules found.")
            return

        water_residues = water_atoms.residues
        n_waters = len(water_residues)
        n_water_beads = n_waters // self.waters_per_bead
        n_waters_mapped = n_water_beads * self.waters_per_bead

        # Build per-residue info
        residue_atom_indices = []
        residue_masses = []
        oxygen_indices = []

        for res in water_residues:
            indices = []
            masses = []
            oxygen_idx = None

            for atom in res.atoms:
                indices.append(atom.index)
                masses.append(self._get_mass_from_name(atom.name))
                if atom.name.startswith('O'):
                    oxygen_idx = atom.index

            residue_atom_indices.append(indices)
            residue_masses.append(np.array(masses))
            if oxygen_idx is None:
                oxygen_idx = indices[0]
            oxygen_indices.append(oxygen_idx)

        w_type = self._global_type_map.get((self._water_resname, 'W'))
        self._water_mapper = {
            'resname': actual_resname,
            'n_waters': n_waters,
            'n_water_beads': n_water_beads,
            'n_waters_mapped': n_waters_mapped,
            'n_waters_leftover': n_waters - n_waters_mapped,
            'residue_atom_indices': residue_atom_indices,
            'residue_masses': residue_masses,
            'oxygen_indices': np.array(oxygen_indices),
            'global_type': w_type,
        }

    def _cluster_waters_spatial(self, positions: np.ndarray) -> List[List[int]]:
        """Cluster water molecules spatially using greedy nearest-neighbor."""
        from scipy.spatial import cKDTree

        wm = self._water_mapper
        n_mapped = wm['n_waters_mapped']
        oxygen_positions = positions[wm['oxygen_indices'][:n_mapped]]

        tree = cKDTree(oxygen_positions)
        assigned = np.zeros(n_mapped, dtype=bool)
        clusters = []

        for _ in range(wm['n_water_beads']):
            unassigned = np.where(~assigned)[0]
            if len(unassigned) == 0:
                break

            seed_idx = unassigned[0]
            seed_pos = oxygen_positions[seed_idx]

            k = min(self.waters_per_bead * 3, len(unassigned))
            _, neighbor_indices = tree.query(seed_pos, k=k)

            cluster = []
            for idx in neighbor_indices:
                if not assigned[idx]:
                    cluster.append(idx)
                    assigned[idx] = True
                    if len(cluster) == self.waters_per_bead:
                        break

            clusters.append(cluster)

        return clusters

    def _compute_types_array(self) -> np.ndarray:
        """Compute bead types array (same for all frames)."""
        types = []

        # Molecule beads
        for res_info in self._residue_bead_info:
            for bead in res_info['beads']:
                types.append(bead['global_type'])

        # Water beads
        if self._water_mapper is not None:
            w_type = self._water_mapper['global_type']
            types.extend([w_type] * self._water_mapper['n_water_beads'])

        return np.array(types, dtype=np.int32)

    def map_frame(
        self,
        positions: np.ndarray,
        forces: np.ndarray,
        box: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Map a single frame from AA to CG.

        Args:
            positions: All-atom positions (n_atoms, 3) in Angstrom
            forces: All-atom forces (n_atoms, 3) in eV/Angstrom (already converted)
            box: Box dimensions (3,) [Lx, Ly, Lz] in Angstrom for PBC-aware COM.
                 If None, standard (non-periodic) COM is used.

        Returns:
            cg_positions: (n_total_beads, 3)
            cg_forces: (n_total_beads, 3)
            cg_types: (n_total_beads,) - cached, same every frame
        """
        cg_positions = np.zeros((self._n_total_beads, 3), dtype=np.float32)
        cg_forces = np.zeros((self._n_total_beads, 3), dtype=np.float32)

        # Map molecule beads
        bead_idx = 0
        for res_info in self._residue_bead_info:
            for bead in res_info['beads']:
                ai = bead['atom_indices']
                m = bead['masses']
                if len(ai) > 0:
                    cg_positions[bead_idx] = compute_com(positions[ai], m, box)
                    cg_forces[bead_idx] = compute_cg_force(forces[ai])
                bead_idx += 1

        # Map water beads (spatial clustering)
        if self._water_mapper is not None and self._n_water_beads > 0:
            clusters = self._cluster_waters_spatial(positions)
            wm = self._water_mapper

            for cluster in clusters:
                all_indices = []
                all_masses = []
                for res_idx in cluster:
                    all_indices.extend(wm['residue_atom_indices'][res_idx])
                    all_masses.extend(wm['residue_masses'][res_idx])

                all_indices = np.array(all_indices)
                all_masses = np.array(all_masses)

                cg_positions[bead_idx] = compute_com(positions[all_indices], all_masses, box)
                cg_forces[bead_idx] = compute_cg_force(forces[all_indices])
                bead_idx += 1

        return cg_positions, cg_forces, self._types_array

    def _build_bond_topology(self) -> Dict:
        """Derive CG bond topology from AA bonds + CG bead mapping.

        For each molecule type, identifies which CG beads are bonded by
        checking if any AA atoms across bead boundaries share an AA bond.

        Returns:
            Dict with per-resname bond topology:
            {
                'OCT': {'n_beads_per_mol': 2, 'bonds': [[0, 1]]},
                'OCN': {'n_beads_per_mol': 3, 'bonds': [[0, 1], [1, 2]]},
                ...
            }
            Also stores a 'global' key with full system topology.
        """
        bond_topologies = {}

        for resname in self._molecule_resnames:
            mapping = self.mappings[resname]
            n_beads_per_mol = len(mapping['bead_order'])

            # Use the first residue of this type to determine bonds
            first_res_info = None
            for ri in self._residue_bead_info:
                if ri['resname'] == resname:
                    first_res_info = ri
                    break

            if first_res_info is None:
                continue

            # Build atom_index → local_bead_index map
            atom_to_bead = {}
            for bead_idx, bead in enumerate(first_res_info['beads']):
                for ai in bead['atom_indices']:
                    atom_to_bead[int(ai)] = bead_idx

            # Check AA bonds: if atom_i ∈ bead_a and atom_j ∈ bead_b (a≠b),
            # then bead_a and bead_b are bonded
            bonds_local = set()
            try:
                for atom_idx in atom_to_bead:
                    atom = self.u.atoms[atom_idx]
                    for bond in atom.bonds:
                        other = bond.partner(atom)
                        other_idx = int(other.index)
                        if other_idx in atom_to_bead:
                            bead_a = atom_to_bead[atom_idx]
                            bead_b = atom_to_bead[other_idx]
                            if bead_a != bead_b:
                                pair = (min(bead_a, bead_b), max(bead_a, bead_b))
                                bonds_local.add(pair)
            except Exception:
                # If AA bonds not available in topology, skip
                pass

            bonds_list = sorted(bonds_local)

            bond_topologies[resname] = {
                'n_beads_per_mol': n_beads_per_mol,
                'bonds': [list(b) for b in bonds_list],
            }

        # Build global topology (for single-resname systems)
        if len(bond_topologies) == 1:
            resname = list(bond_topologies.keys())[0]
            bond_topologies['global'] = bond_topologies[resname]
        elif len(bond_topologies) > 1:
            # Multi-resname: combine (more complex, store per-resname)
            bond_topologies['global'] = None

        return bond_topologies

    def _print_summary(self):
        """Print mapping summary."""
        print(f"\n{'='*60}")
        print("ResidueBasedCGMapper Summary")
        print(f"{'='*60}")

        for resname in self._molecule_resnames:
            n_res = self._residue_config.get(resname, 0)
            mapping = self.mappings[resname]
            n_beads = len(mapping['bead_order'])
            print(f"  {resname}: {n_res} residues x {n_beads} beads = {n_res * n_beads} beads")

        if self._water_mapper is not None:
            wm = self._water_mapper
            print(f"  Water ({wm['resname']}): {wm['n_waters']} molecules -> "
                  f"{wm['n_water_beads']} W beads ({self.waters_per_bead}:1)")
            if wm['n_waters_leftover'] > 0:
                print(f"    Leftover: {wm['n_waters_leftover']} waters (not mapped)")

        print(f"\n  Total beads: {self._n_total_beads}")
        print(f"  Bead types: {self.n_bead_types}")
        print(f"  Type map: {self._bead_type_names}")
        print(f"{'='*60}")

    # -------------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------------

    @property
    def n_beads(self) -> int:
        """Total number of CG beads per frame."""
        return self._n_total_beads

    @property
    def n_bead_types(self) -> int:
        """Number of unique bead types."""
        return len(self._bead_type_names)

    @property
    def bead_type_names(self) -> Dict[int, str]:
        """Mapping of global type ID to bead name."""
        return dict(self._bead_type_names)

    @property
    def types(self) -> np.ndarray:
        """Bead types array (n_beads,)."""
        return self._types_array.copy()

    @property
    def bead_masses(self) -> np.ndarray:
        """Mass of each bead (n_beads,) in g/mol."""
        masses = []
        for res_info in self._residue_bead_info:
            for bead in res_info['beads']:
                masses.append(bead['masses'].sum())
        if self._water_mapper is not None:
            # Water bead mass = waters_per_bead * (O + 2H)
            water_mass = self.waters_per_bead * 18.015
            masses.extend([water_mass] * self._water_mapper['n_water_beads'])
        return np.array(masses, dtype=np.float64)

    @property
    def bond_topology(self) -> Dict:
        """CG bond topology derived from AA bonds.

        Returns dict with per-resname topology and 'global' key.
        Example: {'OCT': {'n_beads_per_mol': 2, 'bonds': [[0, 1]]}, 'global': ...}
        """
        return self._bond_topology

    @property
    def molecule_resnames(self) -> List[str]:
        """List of molecule residue names being mapped."""
        return list(self._molecule_resnames)

    @property
    def residue_config(self) -> Dict[str, int]:
        """Residue name -> count mapping."""
        return dict(self._residue_config)
