"""
CG Dataset Preprocessing and Loading Utilities

This module provides functions to:
1. Preprocess atomistic trajectories into CG representation and save to NPZ
2. Load preprocessed CG datasets for training

Usage:
    # Preprocessing
    python -m bam_torch.utils.cg_dataset preprocess \
        --input water.traj --output water_cg.npz --mapping water

    # Or use in code
    from bam_torch.utils.cg_dataset import preprocess_to_cg, load_cg_dataset
"""

import os
import numpy as np
from typing import Dict, List, Optional, Union, Tuple
from pathlib import Path
import argparse


class CGDataset:
    """
    Dataset class for preprocessed CG data stored in NPZ format.

    NPZ file structure:
        - positions: (n_frames, n_cg_sites, 3) - CG positions
        - forces: (n_frames, n_cg_sites, 3) - CG forces
        - energies: (n_frames,) - Total energies
        - types: (n_cg_sites,) - CG bead types
        - cells: (n_frames, 3, 3) - Unit cells
        - metadata: dict with mapping info, cutoff, etc.
    """

    def __init__(self, npz_path: str):
        """
        Load preprocessed CG dataset from NPZ file.

        Args:
            npz_path: Path to NPZ file
        """
        self.npz_path = npz_path
        self._load_data()

    def _load_data(self):
        """Load data from NPZ file.

        Supports two formats:
        1. Fixed-size: positions (n_frames, n_sites, 3), types (n_sites,)
        2. Multi-system: positions (total_beads, 3) with frame_offsets/frame_sizes
           for variable bead counts per frame.
        """
        data = np.load(self.npz_path, allow_pickle=True)

        self.energies = data['energies']    # (n_frames,)
        self.cells = data['cells']          # (n_frames, 3, 3)
        self.metadata = data['metadata'].item() if 'metadata' in data else {}
        self.n_frames = len(self.energies)

        # Detect format
        if 'frame_offsets' in data:
            # Multi-system flat format
            self.multi_system = True
            self.positions = data['positions']    # (total_beads, 3)
            self.forces = data['forces']          # (total_beads, 3)
            self.types = data['types']            # (total_beads,)
            self.frame_offsets = data['frame_offsets']  # (n_frames,)
            self.frame_sizes = data['frame_sizes']      # (n_frames,)
            self.n_cg_sites = int(self.frame_sizes.max())

            # Optional stress field for stress matching (target stress tensor per frame)
            if 'stress' in data.files:
                self.stress = data['stress']  # (n_frames, 3, 3) in eV/Å³ (pressure units)
                self.has_stress = True
            else:
                self.stress = None
                self.has_stress = False

            print(f"Loaded multi-system CG dataset: {self.npz_path}")
            print(f"  - Frames: {self.n_frames}")
            print(f"  - Beads per frame: {int(self.frame_sizes.min())}-{int(self.frame_sizes.max())}")
            print(f"  - CG bead types: {len(np.unique(self.types))}")
            if self.has_stress:
                print(f"  - Stress field present: shape {self.stress.shape}")
        else:
            # Legacy fixed-size format
            self.multi_system = False
            self.positions = data['positions']  # (n_frames, n_sites, 3)
            self.forces = data['forces']        # (n_frames, n_sites, 3)
            self.types = data['types']          # (n_sites,)
            self.n_cg_sites = len(self.types)

            # Optional stress field
            if 'stress' in data.files:
                self.stress = data['stress']  # (n_frames, 3, 3) in eV/Å³
                self.has_stress = True
            else:
                self.stress = None
                self.has_stress = False

            print(f"Loaded CG dataset: {self.npz_path}")
            print(f"  - Frames: {self.n_frames}")
            print(f"  - CG sites per frame: {self.n_cg_sites}")
            print(f"  - CG bead types: {len(np.unique(self.types))}")
            if self.has_stress:
                print(f"  - Stress field present: shape {self.stress.shape}")

        if self.metadata:
            print(f"  - Mapping: {self.metadata.get('mapping_name', self.metadata.get('mapping', 'unknown'))}")
            print(f"  - Formula: {self.metadata.get('formula', 'unknown')}")

    def __len__(self):
        return self.n_frames

    def __getitem__(self, idx):
        """Get a single frame as dictionary."""
        if self.multi_system:
            start = int(self.frame_offsets[idx])
            size = int(self.frame_sizes[idx])
            result = {
                'positions': self.positions[start:start + size],
                'forces': self.forces[start:start + size],
                'energy': self.energies[idx],
                'types': self.types[start:start + size],
                'cell': self.cells[idx]
            }
            if self.has_stress:
                result['stress'] = self.stress[idx]
            return result
        else:
            result = {
                'positions': self.positions[idx],
                'forces': self.forces[idx],
                'energy': self.energies[idx],
                'types': self.types,
                'cell': self.cells[idx]
            }
            if self.has_stress:
                result['stress'] = self.stress[idx]
            return result

    def get_frame_dict(self, idx: int) -> Dict:
        """Get frame as dictionary (same as __getitem__)."""
        return self[idx]

    def get_all_frames(self) -> List[Dict]:
        """Get all frames as list of dictionaries."""
        return [self[i] for i in range(len(self))]

    def get_subset(self, indices: List[int]) -> List[Dict]:
        """Get subset of frames by indices."""
        return [self[i] for i in indices]


def preprocess_to_cg(
    input_path: str,
    output_path: str,
    mapping_config: Union[str, Dict],
    n_frames: Optional[int] = None,
    start_frame: int = 0,
    stride: int = 1,
    show_progress: bool = True,
    delta_learning: bool = False,
    prior_config: Optional[Union[str, Dict]] = None
) -> str:
    """
    Preprocess atomistic trajectory to CG representation and save to NPZ.

    Args:
        input_path: Path to atomistic trajectory (ASE-readable format)
        output_path: Output NPZ file path
        mapping_config: CG mapping preset name (str) or custom config (dict)
        n_frames: Number of frames to process (None = all)
        start_frame: Starting frame index
        stride: Frame stride
        show_progress: Show progress bar
        delta_learning: If True, compute delta forces (F_total - F_prior)
        prior_config: Prior FF preset name or config dict (required if delta_learning=True)

    Returns:
        Path to saved NPZ file
    """
    from .cg_mapping import CGMapping, CG_PRESETS
    from .prior_ff import PriorForceField

    from ase.io import read

    print("="*60)
    print("CG Dataset Preprocessing")
    if delta_learning:
        print("(Delta Learning Mode)")
    print("="*60)

    # Setup prior FF if delta learning is enabled
    prior_ff = None
    if delta_learning:
        if prior_config is None:
            # Auto-select prior based on mapping
            if isinstance(mapping_config, str):
                prior_config = mapping_config  # Use same preset name
            else:
                raise ValueError("prior_config is required for delta learning with custom mapping")

        if isinstance(prior_config, str):
            prior_ff = PriorForceField.from_preset(prior_config)
        else:
            prior_ff = PriorForceField.from_config(prior_config)

        print(f"\nPrior Force Field: {prior_config}")
        print(f"  - Type: {prior_ff.prior_type}")

    # Load trajectory
    print(f"\nLoading trajectory: {input_path}")
    traj = read(input_path, index=slice(start_frame, None, stride))

    if n_frames is not None:
        traj = traj[:n_frames]

    print(f"  - Total frames: {len(traj)}")
    print(f"  - Atoms per frame: {len(traj[0])}")

    # Create CG mapping
    if isinstance(mapping_config, str):
        mapping_name = mapping_config
        if mapping_config.lower() in CG_PRESETS:
            mapping_config = CG_PRESETS[mapping_config.lower()].copy()
        else:
            raise ValueError(f"Unknown preset: {mapping_config}")
    else:
        mapping_name = mapping_config.get('formula', 'custom')

    mapping = CGMapping(mapping_config)

    print(f"\nCG Mapping:")
    print(f"  - Name: {mapping_name}")
    print(f"  - Formula: {mapping.formula}")
    print(f"  - Atoms per molecule: {mapping.atoms_per_molecule}")
    print(f"  - CG beads per molecule: {mapping.num_cg_sites}")
    print(f"  - Method: {mapping.method}")

    # Convert trajectory
    print(f"\nConverting to CG representation...")

    n_atoms = len(traj[0])
    n_cg_sites = mapping.get_num_cg_sites(n_atoms)
    n_frames_total = len(traj)

    # Pre-allocate arrays
    positions = np.zeros((n_frames_total, n_cg_sites, 3), dtype=np.float32)
    forces = np.zeros((n_frames_total, n_cg_sites, 3), dtype=np.float32)
    energies = np.zeros(n_frames_total, dtype=np.float64)
    cells = np.zeros((n_frames_total, 3, 3), dtype=np.float32)

    # Get CG types (same for all frames)
    types = mapping.get_cg_types(n_atoms).astype(np.int32)

    # Process frames
    if show_progress:
        from tqdm import tqdm
        iterator = tqdm(enumerate(traj), total=n_frames_total, desc="Processing")
    else:
        iterator = enumerate(traj)

    for i, atoms in iterator:
        cg_data = mapping.convert_atoms_to_cg(atoms)
        positions[i] = cg_data['positions']
        forces[i] = cg_data['forces']  # This is F_total
        energies[i] = cg_data['energy']
        cells[i] = cg_data['cell']

    # Compute delta forces if enabled
    if delta_learning and prior_ff is not None:
        print(f"\nComputing delta forces (F_delta = F_total - F_prior)...")
        delta_forces = np.zeros_like(forces)

        if show_progress:
            from tqdm import tqdm
            iterator = tqdm(range(n_frames_total), desc="Computing delta")
        else:
            iterator = range(n_frames_total)

        for i in iterator:
            prior_forces = prior_ff.compute_forces(
                positions[i], types, cells[i] if cells[i].any() else None
            )
            delta_forces[i] = forces[i] - prior_forces

        # Store both total and delta forces
        total_forces = forces.copy()
        forces = delta_forces  # Use delta forces as the main training target

        print(f"  - Total force range: [{total_forces.min():.4f}, {total_forces.max():.4f}] eV/Å")
        print(f"  - Prior force range: [{(total_forces - delta_forces).min():.4f}, {(total_forces - delta_forces).max():.4f}] eV/Å")
        print(f"  - Delta force range: [{delta_forces.min():.4f}, {delta_forces.max():.4f}] eV/Å")

    # Create metadata
    metadata = {
        'mapping_name': mapping_name,
        'mapping_config': mapping.mapping_config,
        'formula': mapping.formula,
        'atoms_per_molecule': mapping.atoms_per_molecule,
        'beads_per_molecule': mapping.num_cg_sites,
        'method': mapping.method,
        'n_molecules': n_cg_sites // mapping.num_cg_sites,
        'source_file': os.path.basename(input_path),
        'n_frames': n_frames_total,
        'n_cg_sites': n_cg_sites,
        'bead_names': mapping.bead_names,
        'bead_types': mapping.bead_types,
        'delta_learning': delta_learning,
        'prior_config': str(prior_config) if delta_learning else None,
    }

    # Save to NPZ
    print(f"\nSaving to: {output_path}")

    save_dict = {
        'positions': positions,
        'forces': forces,  # Delta forces if delta_learning, else total forces
        'energies': energies,
        'types': types,
        'cells': cells,
        'metadata': metadata
    }

    # Also save total forces if delta learning is enabled
    if delta_learning:
        save_dict['total_forces'] = total_forces

    np.savez_compressed(output_path, **save_dict)

    # Print summary
    file_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
    print(f"\nSaved CG dataset:")
    print(f"  - File size: {file_size:.2f} MB")
    print(f"  - Frames: {n_frames_total}")
    print(f"  - CG sites: {n_cg_sites}")
    print(f"  - Energy range: [{energies.min():.4f}, {energies.max():.4f}] eV")
    print(f"  - Force range: [{forces.min():.4f}, {forces.max():.4f}] eV/Å")

    print("="*60)

    return output_path


def load_cg_dataset(npz_path: str) -> CGDataset:
    """
    Load preprocessed CG dataset from NPZ file.

    Args:
        npz_path: Path to NPZ file

    Returns:
        CGDataset object
    """
    return CGDataset(npz_path)



def get_dataloader_from_npz(
    npz_path: str,
    ntrain: int,
    nvalid: int,
    nbatch: int,
    cutoff: float,
    random_seed: int = 42,
    regress_forces: bool = True,
    max_neigh: Optional[int] = None,
    rank: int = 0,
    world_size: int = 1,
    bond_topology: Optional[dict] = None,
    graph_cache: bool = False,
) -> Tuple:
    """
    Create DataLoaders from preprocessed CG NPZ file.

    Args:
        npz_path: Path to preprocessed CG NPZ file
        ntrain: Number of training samples
        nvalid: Number of validation samples
        nbatch: Batch size
        cutoff: Cutoff distance for neighbor list
        random_seed: Random seed for data splitting
        regress_forces: Whether to include forces
        max_neigh: Maximum number of neighbors
        rank: Process rank for distributed training
        world_size: Number of processes

    Returns:
        train_loader, valid_loader, uniq_type, enr_avg_per_type
    """
    import torch
    from torch_geometric.loader import DataLoader
    from .utils import get_graphset_cg, get_cg_enr_avg_per_type

    # Load dataset
    dataset = CGDataset(npz_path)

    # Check if we have enough frames
    total_needed = ntrain + nvalid
    if total_needed > len(dataset):
        raise ValueError(
            f"Not enough frames in dataset: need {total_needed}, have {len(dataset)}"
        )

    # Split into train/valid
    torch.manual_seed(random_seed)
    indices = torch.randperm(len(dataset))[:total_needed].tolist()

    train_indices = indices[:ntrain]
    valid_indices = indices[ntrain:ntrain + nvalid]

    train_data = dataset.get_subset(train_indices)
    valid_data = dataset.get_subset(valid_indices)

    if rank == 0:
        print(f"\nDataset split:")
        print(f"  - Training: {len(train_data)}")
        print(f"  - Validation: {len(valid_data)}")

    # Get number of CG types: use max type value + 1 (types are direct indices)
    num_cg_types = int(np.max(dataset.types)) + 1

    # Calculate energy averages
    all_data = train_data + valid_data
    enr_avg_per_type, uniq_type, enr_var = get_cg_enr_avg_per_type(
        all_data, num_cg_types
    )

    if rank == 0:
        print(f"\nMean energy per CG type:")
        for t, e in enr_avg_per_type.items():
            print(f"  Type {t}: {e:.4f} eV")

    # Create graph datasets (optional mp-style disk cache of built graphset)
    import os as _os, pickle as _pickle, hashlib as _hashlib, time as _time
    loaders = []
    for data, name in [(train_data, 'train'), (valid_data, 'valid')]:
        graphset = None
        _cfile = None
        if graph_cache:
            _key = f"{_os.path.basename(npz_path)}|{name}|cut{cutoff}|n{len(data)}|seed{random_seed}|mn{max_neigh}|rf{regress_forces}|{str(bond_topology)}"
            _h = _hashlib.md5(_key.encode()).hexdigest()[:10]
            _cdir = _os.path.join(_os.path.dirname(_os.path.abspath(npz_path)), '_graphcache')
            _os.makedirs(_cdir, exist_ok=True)
            _stem = _os.path.splitext(_os.path.basename(npz_path))[0]
            _cfile = _os.path.join(_cdir, f"{_stem}_{name}_{_h}.pkl")
            if _os.path.exists(_cfile):
                _t0 = _time.time()
                with open(_cfile, 'rb') as _f:
                    graphset = _pickle.load(_f)
                if rank == 0:
                    print(f"  [graph_cache] loaded {name} graphs from {_cfile} ({_time.time()-_t0:.1f}s, {len(graphset)} graphs)")
        if graphset is None:
            _t0 = _time.time()
            graphset = get_graphset_cg(
                data, cutoff, uniq_type, enr_avg_per_type, enr_var,
                regress_forces, max_neigh,
                show_progress=(rank == 0), desc=f"Building {name} graphs",
                bond_topology=bond_topology,
            )
            if graph_cache and _cfile is not None:
                with open(_cfile, 'wb') as _f:
                    _pickle.dump(graphset, _f)
                if rank == 0:
                    print(f"  [graph_cache] built+saved {name} graphs -> {_cfile} ({_time.time()-_t0:.1f}s)")

        # Padding
        pad_nodes_to = max(g.num_nodes for g in graphset)
        pad_edges_to = max(g.num_edges for g in graphset)

        for g in graphset:
            g.pad_nodes_to = pad_nodes_to
            g.pad_edges_to = pad_edges_to

        # Create loader
        if world_size > 1:
            from torch.utils.data.distributed import DistributedSampler
            sampler = DistributedSampler(
                graphset, num_replicas=world_size, rank=rank, shuffle=(name == 'train')
            )
            loader = DataLoader(
                graphset, batch_size=nbatch, sampler=sampler, drop_last=True
            )
        else:
            loader = DataLoader(
                graphset, batch_size=nbatch, shuffle=(name == 'train'), drop_last=True
            )

        loaders.append(loader)

        if rank == 0:
            print(f"\n{name.capitalize()} loader:")
            print(f"  - Graphs: {len(graphset)}")
            print(f"  - Batches: {len(loader)}")
            print(f"  - Max nodes: {pad_nodes_to}")
            print(f"  - Max edges: {pad_edges_to}")

    return loaders[0], loaders[1], uniq_type, enr_avg_per_type


def get_dataloader_from_split_npz(
    train_npz: str,
    valid_npz: str,
    nbatch: int,
    cutoff: float,
    regress_forces: bool = True,
    max_neigh: Optional[int] = None,
    ntrain: Optional[int] = None,
    nvalid: Optional[int] = None,
    random_seed: int = 42,
    rank: int = 0,
    world_size: int = 1
) -> Tuple:
    """
    Create DataLoaders from pre-split train/valid NPZ files.

    Unlike get_dataloader_from_npz, this does not perform random splitting.
    Optionally limits the number of frames used from each file.

    Args:
        train_npz: Path to training NPZ file
        valid_npz: Path to validation NPZ file
        nbatch: Batch size
        cutoff: Cutoff distance for neighbor list
        regress_forces: Whether to include forces
        max_neigh: Maximum number of neighbors
        ntrain: Max number of training frames (None = use all)
        nvalid: Max number of validation frames (None = use all)
        random_seed: Random seed for frame selection
        rank: Process rank for distributed training
        world_size: Number of processes

    Returns:
        train_loader, valid_loader, uniq_type, enr_avg_per_type
    """
    import torch
    from torch_geometric.loader import DataLoader
    from .utils import get_graphset_cg, get_cg_enr_avg_per_type

    train_dataset = CGDataset(train_npz)
    valid_dataset = CGDataset(valid_npz)

    # Select subset of frames if ntrain/nvalid specified
    if ntrain is not None and ntrain < len(train_dataset):
        torch.manual_seed(random_seed)
        indices = torch.randperm(len(train_dataset))[:ntrain].tolist()
        train_data = train_dataset.get_subset(indices)
    else:
        train_data = train_dataset.get_all_frames()

    if nvalid is not None and nvalid < len(valid_dataset):
        torch.manual_seed(random_seed + 1)
        indices = torch.randperm(len(valid_dataset))[:nvalid].tolist()
        valid_data = valid_dataset.get_subset(indices)
    else:
        valid_data = valid_dataset.get_all_frames()

    if rank == 0:
        print(f"\nPre-split dataset:")
        print(f"  - Training: {len(train_data)} frames")
        print(f"  - Validation: {len(valid_data)} frames")

    # Get number of CG types: use max type value + 1 (types are direct indices)
    all_types = np.concatenate([train_dataset.types, valid_dataset.types])
    num_cg_types = int(np.max(all_types)) + 1

    # Calculate energy averages
    all_data = train_data + valid_data
    enr_avg_per_type, uniq_type, enr_var = get_cg_enr_avg_per_type(
        all_data, num_cg_types
    )

    if rank == 0:
        print(f"\nMean energy per CG type:")
        for t, e in enr_avg_per_type.items():
            print(f"  Type {t}: {e:.4f} eV")

    # Create graph datasets
    loaders = []
    for data, name in [(train_data, 'train'), (valid_data, 'valid')]:
        graphset = get_graphset_cg(
            data, cutoff, uniq_type, enr_avg_per_type, enr_var,
            regress_forces, max_neigh,
            show_progress=(rank == 0), desc=f"Building {name} graphs"
        )

        pad_nodes_to = max(g.num_nodes for g in graphset)
        pad_edges_to = max(g.num_edges for g in graphset)

        for g in graphset:
            g.pad_nodes_to = pad_nodes_to
            g.pad_edges_to = pad_edges_to

        if world_size > 1:
            from torch.utils.data.distributed import DistributedSampler
            sampler = DistributedSampler(
                graphset, num_replicas=world_size, rank=rank, shuffle=(name == 'train')
            )
            loader = DataLoader(
                graphset, batch_size=nbatch, sampler=sampler, drop_last=True
            )
        else:
            loader = DataLoader(
                graphset, batch_size=nbatch, shuffle=(name == 'train'), drop_last=True
            )

        loaders.append(loader)

        if rank == 0:
            print(f"\n{name.capitalize()} loader:")
            print(f"  - Graphs: {len(graphset)}")
            print(f"  - Batches: {len(loader)}")
            print(f"  - Max nodes: {pad_nodes_to}")
            print(f"  - Max edges: {pad_edges_to}")

    return loaders[0], loaders[1], uniq_type, enr_avg_per_type


def merge_cg_npz(
    npz_paths: List[str],
    output_path: str,
    show_progress: bool = True,
) -> str:
    """
    Merge multiple CG NPZ files into a single multi-system flat NPZ file.

    Automatically builds a unified global type map from all input files.
    Each input file can have different bead counts and type mappings.

    Args:
        npz_paths: List of paths to CG NPZ files
        output_path: Output path for merged NPZ file
        show_progress: Print progress info

    Returns:
        Path to saved merged NPZ file
    """
    # Phase 1: Scan all files to build unified global type map
    file_infos = []
    all_type_names = set()

    for path in npz_paths:
        data = np.load(path, allow_pickle=True)
        meta = data['metadata'].item()

        # Extract local type name mapping (id -> name)
        local_id_to_name = {}
        if 'bead_type_names' in meta:
            # gromacs_to_cg format: {id: name}
            for k, v in meta['bead_type_names'].items():
                local_id_to_name[int(k)] = v
        elif 'bead_types' in meta:
            # convert_aa_to_cg format: {name: id}
            for name, idx in meta['bead_types'].items():
                local_id_to_name[int(idx)] = name
        elif 'bead_names' in meta:
            # preprocess_to_cg format: list of names
            for idx, name in enumerate(meta['bead_names']):
                local_id_to_name[idx] = name
        else:
            raise ValueError(f"Cannot determine bead type names from {path}")

        n_frames = len(data['energies'])
        n_beads = data['positions'].shape[1] if data['positions'].ndim == 3 else None

        file_infos.append({
            'path': path,
            'n_frames': n_frames,
            'n_beads': n_beads,
            'local_id_to_name': local_id_to_name,
        })
        all_type_names.update(local_id_to_name.values())
        del data

    # Build global type map (sorted for consistency)
    global_type_map = {name: idx for idx, name in enumerate(sorted(all_type_names))}
    global_id_to_name = {v: k for k, v in global_type_map.items()}

    if show_progress:
        print(f"{'='*60}")
        print(f"Merging {len(npz_paths)} CG NPZ files")
        print(f"{'='*60}")
        print(f"\nGlobal type map ({len(global_type_map)} types):")
        for name, idx in global_type_map.items():
            print(f"  {idx:3d}: {name}")

    # Phase 2: Pre-compute total sizes for pre-allocation
    total_frames = sum(info['n_frames'] for info in file_infos)
    total_beads = sum(info['n_frames'] * info['n_beads'] for info in file_infos)

    if show_progress:
        print(f"\nTotal: {total_frames} frames, {total_beads} beads")
        print(f"Pre-allocating arrays...")

    # Pre-allocate output arrays
    positions = np.zeros((total_beads, 3), dtype=np.float32)
    forces = np.zeros((total_beads, 3), dtype=np.float32)
    types = np.zeros(total_beads, dtype=np.int32)
    energies = np.zeros(total_frames, dtype=np.float64)
    cells = np.zeros((total_frames, 3, 3), dtype=np.float32)
    frame_offsets = np.zeros(total_frames, dtype=np.int64)
    frame_sizes = np.zeros(total_frames, dtype=np.int32)

    # Phase 3: Fill arrays one system at a time
    bead_offset = 0
    frame_idx = 0
    system_info = {}

    for info in file_infos:
        data = np.load(info['path'], allow_pickle=True)
        local_id_to_name = info['local_id_to_name']

        # Build remap array
        n_local = max(local_id_to_name.keys()) + 1
        remap = np.zeros(n_local, dtype=np.int32)
        for local_id, name in local_id_to_name.items():
            remap[local_id] = global_type_map[name]

        # Remap types once
        global_types = remap[data['types']]  # (n_beads,)

        n_frames = info['n_frames']
        n_beads = info['n_beads']
        sysname = os.path.splitext(os.path.basename(info['path']))[0]

        if show_progress:
            print(f"\n{sysname}: {n_frames} frames, {n_beads} beads/frame")

        # Vectorized copy: reshape (n_frames, n_beads, 3) -> (n_frames*n_beads, 3)
        total_sys_beads = n_frames * n_beads
        bead_end = bead_offset + total_sys_beads
        positions[bead_offset:bead_end] = data['positions'].reshape(-1, 3)
        forces[bead_offset:bead_end] = data['forces'].reshape(-1, 3)

        # Tile global_types for all frames
        types[bead_offset:bead_end] = np.tile(global_types, n_frames)

        # Fill frame_offsets and frame_sizes
        offsets = bead_offset + np.arange(n_frames, dtype=np.int64) * n_beads
        frame_offsets[frame_idx:frame_idx + n_frames] = offsets
        frame_sizes[frame_idx:frame_idx + n_frames] = n_beads

        # Copy energies and cells as blocks
        energies[frame_idx:frame_idx + n_frames] = data['energies']
        cells[frame_idx:frame_idx + n_frames] = data['cells']

        bead_offset = bead_end
        frame_idx += n_frames

        system_info[sysname] = {'frames': n_frames, 'beads_per_frame': n_beads}
        del data

    total_frames = len(energies)

    metadata = {
        'format': 'multi_system_flat',
        'n_frames': total_frames,
        'n_bead_types': len(global_type_map),
        'global_type_map': global_type_map,
        'global_id_to_name': global_id_to_name,
        'systems': system_info,
        'source_files': [os.path.basename(p) for p in npz_paths],
        'unit_position': 'Angstrom',
        'unit_force': 'eV/Angstrom',
        'unit_energy': 'eV',
    }

    if show_progress:
        print(f"\nSaving to: {output_path}")

    np.savez_compressed(output_path,
        positions=positions,
        forces=forces,
        types=types,
        energies=energies,
        cells=cells,
        frame_offsets=frame_offsets,
        frame_sizes=frame_sizes,
        metadata=metadata,
    )

    file_size = os.path.getsize(output_path) / (1024 * 1024)

    if show_progress:
        print(f"\n{'='*60}")
        print(f"Merged CG dataset saved: {output_path}")
        print(f"  - File size: {file_size:.1f} MB")
        print(f"  - Total frames: {total_frames}")
        print(f"  - Total beads: {len(positions)}")
        print(f"  - Bead types: {len(global_type_map)}")
        print(f"  - Unique types used: {len(np.unique(types))}")
        for sname, sinfo in system_info.items():
            print(f"  - {sname}: {sinfo['frames']} frames, {sinfo['beads_per_frame']} beads")
        print(f"{'='*60}")

    return output_path


def split_cg_npz(
    npz_path: str,
    output_dir: str,
    ntrain: int,
    nvalid: int,
    ntest: int = 0,
    stratify: bool = True,
    random_seed: int = 42,
    show_progress: bool = True,
) -> Dict[str, str]:
    """
    Split a multi-system CG NPZ file into train/valid/test sets.

    Frames are sampled equally from each system (stratified split) to ensure
    balanced representation. Output files use the same multi-system flat format.

    Args:
        npz_path: Path to merged multi-system NPZ file
        output_dir: Directory for output files (train_data.npz, valid_data.npz, test_data.npz)
        ntrain: Number of training frames
        nvalid: Number of validation frames
        ntest: Number of test frames (0 to skip)
        stratify: If True, sample equally from each system
        random_seed: Random seed for reproducibility
        show_progress: Print progress info

    Returns:
        Dict of split name -> output file path
    """
    rng = np.random.RandomState(random_seed)

    data = np.load(npz_path, allow_pickle=True)
    metadata = data['metadata'].item()

    if 'frame_offsets' not in data:
        raise ValueError("split_cg_npz requires multi-system flat format (with frame_offsets)")

    positions = data['positions']
    forces = data['forces']
    types = data['types']
    energies = data['energies']
    cells = data['cells']
    frame_offsets = data['frame_offsets']
    frame_sizes = data['frame_sizes']
    n_frames = len(energies)

    systems = metadata.get('systems', {})
    global_type_map = metadata.get('global_type_map', {})
    global_id_to_name = metadata.get('global_id_to_name', {})

    if show_progress:
        print(f"{'='*60}")
        print(f"Splitting CG dataset: {npz_path}")
        print(f"{'='*60}")
        print(f"Total frames: {n_frames}")
        print(f"Systems: {list(systems.keys())}")

    # Identify per-system frame ranges
    system_ranges = []
    frame_cursor = 0
    for sname, sinfo in systems.items():
        nf = sinfo['frames']
        system_ranges.append((sname, frame_cursor, frame_cursor + nf, sinfo['beads_per_frame']))
        frame_cursor += nf

    n_systems = len(system_ranges)
    total_needed = ntrain + nvalid + ntest

    if stratify and n_systems > 0:
        # Equal allocation per system
        per_sys_train = ntrain // n_systems
        per_sys_valid = nvalid // n_systems
        per_sys_test = ntest // n_systems

        # Distribute remainders to first systems
        extra_train = ntrain % n_systems
        extra_valid = nvalid % n_systems
        extra_test = ntest % n_systems

        split_indices = {'train': [], 'valid': [], 'test': []}

        for i, (sname, start, end, _) in enumerate(system_ranges):
            sys_n = end - start
            nt = per_sys_train + (1 if i < extra_train else 0)
            nv = per_sys_valid + (1 if i < extra_valid else 0)
            ns = per_sys_test + (1 if i < extra_test else 0)

            needed = nt + nv + ns
            if needed > sys_n:
                raise ValueError(
                    f"System {sname} has {sys_n} frames but needs {needed} "
                    f"(train={nt}, valid={nv}, test={ns})"
                )

            # Random permutation within this system
            perm = rng.permutation(sys_n) + start
            split_indices['train'].extend(perm[:nt])
            split_indices['valid'].extend(perm[nt:nt + nv])
            if ns > 0:
                split_indices['test'].extend(perm[nt + nv:nt + nv + ns])

            if show_progress:
                print(f"  {sname}: train={nt}, valid={nv}, test={ns} (from {sys_n} frames)")
    else:
        # Non-stratified random split
        perm = rng.permutation(n_frames)
        split_indices = {
            'train': perm[:ntrain].tolist(),
            'valid': perm[ntrain:ntrain + nvalid].tolist(),
            'test': perm[ntrain + nvalid:ntrain + nvalid + ntest].tolist() if ntest > 0 else [],
        }

    # Save each split
    os.makedirs(output_dir, exist_ok=True)
    output_paths = {}

    splits_to_save = [('train', split_indices['train']),
                      ('valid', split_indices['valid'])]
    if ntest > 0:
        splits_to_save.append(('test', split_indices['test']))

    for split_name, indices in splits_to_save:
        indices = sorted(indices)
        n_split = len(indices)

        # Compute total beads for this split
        split_total_beads = sum(int(frame_sizes[i]) for i in indices)

        # Pre-allocate
        sp = np.zeros((split_total_beads, 3), dtype=np.float32)
        sf = np.zeros((split_total_beads, 3), dtype=np.float32)
        st = np.zeros(split_total_beads, dtype=np.int32)
        se = np.zeros(n_split, dtype=np.float64)
        sc = np.zeros((n_split, 3, 3), dtype=np.float32)
        so = np.zeros(n_split, dtype=np.int64)
        ss = np.zeros(n_split, dtype=np.int32)

        bead_cursor = 0
        for j, idx in enumerate(indices):
            src_start = int(frame_offsets[idx])
            src_size = int(frame_sizes[idx])
            src_end = src_start + src_size

            sp[bead_cursor:bead_cursor + src_size] = positions[src_start:src_end]
            sf[bead_cursor:bead_cursor + src_size] = forces[src_start:src_end]
            st[bead_cursor:bead_cursor + src_size] = types[src_start:src_end]
            so[j] = bead_cursor
            ss[j] = src_size
            se[j] = energies[idx]
            sc[j] = cells[idx]
            bead_cursor += src_size

        split_meta = {
            'format': 'multi_system_flat',
            'n_frames': n_split,
            'n_bead_types': metadata.get('n_bead_types', len(global_type_map)),
            'global_type_map': global_type_map,
            'global_id_to_name': global_id_to_name,
            'systems': systems,
            'split': split_name,
            'random_seed': random_seed,
            'unit_position': 'Angstrom',
            'unit_force': 'eV/Angstrom',
            'unit_energy': 'eV',
        }

        out_path = os.path.join(output_dir, f'{split_name}_data.npz')
        np.savez_compressed(out_path,
            positions=sp, forces=sf, types=st,
            energies=se, cells=sc,
            frame_offsets=so, frame_sizes=ss,
            metadata=split_meta,
        )
        fsize = os.path.getsize(out_path) / (1024 * 1024)
        output_paths[split_name] = out_path

        if show_progress:
            print(f"\n{split_name}_data.npz: {n_split} frames, {split_total_beads} beads, {fsize:.1f} MB")

    if show_progress:
        print(f"\n{'='*60}")

    del data
    return output_paths


def print_npz_info(npz_path: str):
    """Print information about a CG NPZ file."""
    print("="*60)
    print(f"CG Dataset Info: {npz_path}")
    print("="*60)

    data = np.load(npz_path, allow_pickle=True)

    print(f"\nArrays:")
    for key in data.files:
        arr = data[key]
        if key == 'metadata':
            print(f"  - {key}: dict")
        else:
            print(f"  - {key}: shape={arr.shape}, dtype={arr.dtype}")

    if 'metadata' in data:
        metadata = data['metadata'].item()
        print(f"\nMetadata:")
        for key, value in metadata.items():
            if isinstance(value, dict):
                print(f"  - {key}: <dict>")
            elif isinstance(value, (list, np.ndarray)) and len(str(value)) > 50:
                print(f"  - {key}: <{type(value).__name__}>")
            else:
                print(f"  - {key}: {value}")

    # Statistics
    print(f"\nStatistics:")
    print(f"  - Energy: min={data['energies'].min():.4f}, max={data['energies'].max():.4f}, "
          f"mean={data['energies'].mean():.4f} eV")
    print(f"  - Force magnitude: max={np.abs(data['forces']).max():.4f} eV/Å")

    print("="*60)


# =============================================================================
# CLI Interface
# =============================================================================

def main():
    """Command-line interface for CG dataset preprocessing."""
    parser = argparse.ArgumentParser(
        description='CG Dataset Preprocessing Tool',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Preprocess water trajectory
  python -m bam_torch.utils.cg_dataset preprocess \\
      --input water.traj --output water_cg.npz --mapping water

  # Preprocess with custom mapping
  python -m bam_torch.utils.cg_dataset preprocess \\
      --input traj.xyz --output cg.npz --mapping ethanol

  # Show NPZ file info
  python -m bam_torch.utils.cg_dataset info --input water_cg.npz

  # List available presets
  python -m bam_torch.utils.cg_dataset list-presets
"""
    )

    subparsers = parser.add_subparsers(dest='command', help='Commands')

    # Preprocess command
    prep_parser = subparsers.add_parser('preprocess', help='Preprocess trajectory to CG')
    prep_parser.add_argument('--input', '-i', required=True, help='Input trajectory file')
    prep_parser.add_argument('--output', '-o', required=True, help='Output NPZ file')
    prep_parser.add_argument('--mapping', '-m', default='water',
                            help='CG mapping preset name (default: water)')
    prep_parser.add_argument('--n-frames', '-n', type=int, default=None,
                            help='Number of frames to process (default: all)')
    prep_parser.add_argument('--start', type=int, default=0,
                            help='Starting frame index (default: 0)')
    prep_parser.add_argument('--stride', type=int, default=1,
                            help='Frame stride (default: 1)')

    # Info command
    info_parser = subparsers.add_parser('info', help='Show NPZ file info')
    info_parser.add_argument('--input', '-i', required=True, help='Input NPZ file')

    # List presets command
    list_parser = subparsers.add_parser('list-presets', help='List available CG presets')

    args = parser.parse_args()

    if args.command == 'preprocess':
        preprocess_to_cg(
            input_path=args.input,
            output_path=args.output,
            mapping_config=args.mapping,
            n_frames=args.n_frames,
            start_frame=args.start,
            stride=args.stride
        )

    elif args.command == 'info':
        print_npz_info(args.input)

    elif args.command == 'list-presets':
        from .cg_mapping import print_available_presets
        print_available_presets()

    else:
        parser.print_help()


if __name__ == '__main__':
    main()
