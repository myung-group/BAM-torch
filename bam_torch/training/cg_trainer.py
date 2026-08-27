"""
Coarse-Grained (CG) Trainer for BAM-torch

This trainer handles CG model training using bottom-up approach:
- Converts atomistic trajectories to CG representation
- Uses the standard RACE model architecture with CG inputs
- Trains on CG energies and forces derived from atomistic data

Supports two data modes:
1. On-the-fly conversion: fname_traj (atomistic trajectory) + mapping config
2. Preprocessed NPZ: fname_cg_npz (preprocessed CG dataset)

Supports two learning modes:
1. Direct learning: Learn F_total directly
2. Delta learning: Learn F_delta = F_total - F_prior
"""

import torch
import numpy as np
from e3nn import o3

from .base_trainer import BaseTrainer
from bam_torch.utils.utils import get_dataloader_cg
from bam_torch.utils.cg_mapping import CGMapping


class CGTrainer(BaseTrainer):
    """
    Trainer for Coarse-Grained (CG) models.

    Inherits from BaseTrainer and overrides:
    - configure_dataloader: Uses CG-specific data loading
    - configure_checkpoint: Saves CG-specific information

    The model architecture (RACE) remains unchanged; only the input data
    is transformed to CG representation.
    """

    def __init__(self, json_data, rank=0, world_size=1):
        """
        Initialize CG Trainer.

        Args:
            json_data: Configuration dictionary containing:
                - Standard BAM training parameters
                - cg_config: CG-specific configuration
                    - mapping: CG mapping configuration or "water" for preset
                    - cutoff: CG cutoff distance (typically larger than atomistic)
                    - fname_npz: (optional) Path to preprocessed CG NPZ file
                    - delta_learning: (optional) Enable delta learning mode
                    - prior: (optional) Prior FF preset name or config for delta learning
            rank: Process rank for distributed training
            world_size: Number of processes

        Data modes:
            1. On-the-fly: Use 'fname_traj' + 'mapping' to convert atomistic data
            2. Preprocessed: Use 'fname_npz' to load pre-converted CG data

        Learning modes:
            1. Direct: Learn F_total directly (default)
            2. Delta: Learn F_delta = F_total - F_prior (set delta_learning=True)
        """
        # Store CG config before calling parent __init__
        self.cg_config = json_data.get('cg_config', {})

        # Determine data mode
        self.use_preprocessed = ('fname_npz' in self.cg_config or
                                 'fname_train_npz' in self.cg_config)

        # Determine learning mode
        self.delta_learning = self.cg_config.get('delta_learning', False)
        self.prior_config = self.cg_config.get('prior', None)

        # Initialize prior FF for delta learning
        self.prior_ff = None
        if self.delta_learning:
            from bam_torch.utils.prior_ff import PriorForceField
            if self.prior_config is None:
                # Auto-select prior based on mapping
                mapping = self.cg_config.get('mapping', 'water')
                if isinstance(mapping, str):
                    self.prior_config = mapping
                else:
                    raise ValueError("prior config is required for delta learning with custom mapping")

            if isinstance(self.prior_config, str):
                self.prior_ff = PriorForceField.from_config({'type': self.prior_config})
            else:
                # For priors with "auto" parameters,
                # load actual values from preprocessed NPZ metadata
                prior_type = self.prior_config.get('type', '')
                if prior_type in ('harmonic_repulsive', 'repulsive_lj', 'harmonic_bond'):
                    npz_path = self.cg_config.get('fname_npz', None)
                    self.prior_config = self._load_prior_from_npz(self.prior_config, npz_path)
                self.prior_ff = PriorForceField.from_config(self.prior_config)

        super().__init__(json_data, rank, world_size)

    @staticmethod
    def _load_prior_from_npz(prior_config: dict, npz_path: str) -> dict:
        """Load prior parameters from NPZ metadata when config has 'auto' values."""
        import numpy as np

        if npz_path is None:
            return prior_config

        # Check if any values are "auto" (top-level or nested)
        has_auto = False
        for v in prior_config.values():
            if v == 'auto':
                has_auto = True
                break
            if isinstance(v, dict):
                if any(sv == 'auto' for sv in v.values() if isinstance(sv, str)):
                    has_auto = True
                    break

        if not has_auto:
            return prior_config

        # Load prior config from NPZ metadata
        try:
            data = np.load(npz_path, allow_pickle=True)
            if 'metadata' in data.files:
                metadata = data['metadata'].item() if data['metadata'].ndim == 0 else {}
                saved_config = metadata.get('prior_config', None)
                if saved_config is not None:
                    print("  Loaded prior parameters from NPZ metadata")
                    return saved_config
        except Exception as e:
            print(f"  Warning: Could not load prior from NPZ: {e}")

        # Try companion JSON file
        import os
        json_path = npz_path.replace('.npz', '_prior_config.json')
        if os.path.exists(json_path):
            import json
            with open(json_path) as f:
                saved_config = json.load(f)
            print(f"  Loaded prior parameters from {json_path}")
            return saved_config

        return prior_config

    def configure_dataloader(self):
        """
        Configure CG dataloaders.

        Supports two modes:
        1. On-the-fly: Converts atomistic trajectory to CG representation
        2. Preprocessed: Loads pre-converted CG data from NPZ file

        Returns:
            train_loader, valid_loader, uniq_type, enr_avg_per_type
        """
        if self.rank == 0:
            print("\n" + "="*60)
            print("Coarse-Grained (CG) Training Mode")
            if self.delta_learning:
                print("(Delta Learning: F_delta = F_total - F_prior)")
            print("="*60)

            if self.delta_learning and self.prior_ff is not None:
                print(f"\nPrior Force Field: {self.prior_config}")
                print(f"  - Type: {self.prior_ff.prior_type}")

        # Get CG-specific cutoff (default: use model cutoff)
        cg_cutoff = self.cg_config.get('cutoff', self.json_data['cutoff'])

        # Number of CG types
        num_cg_types = self.json_data.get('num_cg_types', 1)

        # Mode 3: mp-style sharded streaming (large datasets)
        if self.cg_config.get('shard_dir'):
            return self._configure_dataloader_from_shards(cg_cutoff, num_cg_types)

        if self.use_preprocessed:
            # Mode 2: Load from preprocessed NPZ file
            return self._configure_dataloader_from_npz(cg_cutoff, num_cg_types)
        else:
            # Mode 1: On-the-fly conversion from atomistic trajectory
            return self._configure_dataloader_on_the_fly(cg_cutoff, num_cg_types)

    def _configure_dataloader_from_shards(self, cg_cutoff, num_cg_types):
        """mp-style sharded streaming: load shard manifest, stream shards per epoch."""
        import os, json
        shard_dir = self.cg_config['shard_dir']
        with open(os.path.join(shard_dir, 'manifest.json')) as f:
            man = json.load(f)
        # Guard: shard graphs carry baked-in force labels; the prior is NOT
        # subtracted on the fly in shard mode. Delta learning therefore
        # requires shards built with make_delta_shards.py (pre-subtracted).
        man_delta = man.get('delta_prior_applied', False)
        if self.delta_learning and not man_delta:
            raise RuntimeError(
                'delta_learning=True but shard manifest lacks delta_prior_applied: '
                'these shards store raw F_total, so training would silently fit '
                'F_total instead of F_delta. Pre-subtract with make_delta_shards.py '
                'and point shard_dir at the delta shards.')
        if (not self.delta_learning) and man_delta:
            raise RuntimeError(
                'shards are delta-subtracted (delta_prior_applied) but '
                'delta_learning is False — label/model mismatch.')
        self.shard_mode = True
        self._train_shards = [os.path.join(shard_dir, x) for x in man['train_shards']]
        self._valid_shards = [os.path.join(shard_dir, x) for x in man['valid_shards']]
        self._shard_bt = man.get('bond_topology')
        uniq_type = {int(k): v for k, v in man['uniq_type'].items()}
        enr_avg = {int(k): v for k, v in man['enr_avg_per_type'].items()}
        if self.rank == 0:
            print(f"\nData mode: mp-style sharded streaming")
            print(f"  Shard dir: {shard_dir}")
            print(f"  Train shards: {len(self._train_shards)} ({man.get('n_train')} frames)")
            print(f"  Valid shards: {len(self._valid_shards)} ({man.get('n_valid')} frames)")
            print(f"  Shard size: {man.get('shard_size')} frames/shard")
        return None, None, uniq_type, enr_avg

    def _load_shard_loader(self, shard_path, shuffle):
        """Load one pkl shard -> padded PyG DataLoader."""
        import pickle
        from torch_geometric.loader import DataLoader
        with open(shard_path, 'rb') as f:
            graphset = pickle.load(f)
        pad_nodes = max(g.num_nodes for g in graphset)
        pad_edges = max(g.num_edges for g in graphset)
        for g in graphset:
            g.pad_nodes_to = pad_nodes
            g.pad_edges_to = pad_edges
        loader = DataLoader(graphset, batch_size=self.json_data['nbatch'],
                            shuffle=shuffle, drop_last=True)
        return graphset, loader

    def train_one_epoch(self, mode='train'):
        """Stream shards for one epoch when in shard mode; else default behaviour."""
        if not getattr(self, 'shard_mode', False):
            return super().train_one_epoch(mode)
        import gc
        import torch as _torch
        shards = self._train_shards if mode == 'train' else self._valid_shards
        agg = {}
        wsum = 0
        for sp in shards:
            graphset, loader = self._load_shard_loader(sp, shuffle=(mode == 'train'))
            ld = super().train_one_epoch(mode, data_loader=loader)
            w = max(len(loader), 1)
            for k, v in ld.items():
                fv = float(v)
                if fv == fv:  # not NaN
                    agg[k] = agg.get(k, 0.0) + fv * w
            wsum += w
            del graphset, loader
            gc.collect()
            if _torch.cuda.is_available():
                _torch.cuda.empty_cache()
        return {k: _torch.tensor(v / max(wsum, 1)) for k, v in agg.items()}

    def _configure_dataloader_on_the_fly(self, cg_cutoff, num_cg_types):
        """
        Configure dataloaders with on-the-fly CG conversion.

        Args:
            cg_cutoff: Cutoff distance for CG neighbor list
            num_cg_types: Number of CG bead types

        Returns:
            train_loader, valid_loader, uniq_type, enr_avg_per_type
        """
        if self.rank == 0:
            print("\nData mode: On-the-fly CG conversion")

        # Get CG mapping configuration
        cg_mapping_config = self._get_cg_mapping_config()

        train_loader, valid_loader, uniq_type, enr_avg_per_type = \
            get_dataloader_cg(
                fname=self.json_data['fname_traj'],
                cg_mapping_config=cg_mapping_config,
                ntrain=self.json_data['ntrain'],
                nvalid=self.json_data['nvalid'],
                nbatch=self.json_data['nbatch'],
                cutoff=cg_cutoff,
                random_seed=self.json_data['NN']['data_seed'],
                num_cg_types=num_cg_types,
                regress_forces=self.json_data['regress_forces'],
                max_neigh=self.json_data.get('max_neigh'),
                rank=self.rank,
                world_size=self.world_size,
                num_workers=self.json_data.get('num_workers', 4),
            )

        # Store CG mapping for checkpoint
        self.cg_mapping_config = cg_mapping_config

        return train_loader, valid_loader, uniq_type, enr_avg_per_type

    def _configure_dataloader_from_npz(self, cg_cutoff, num_cg_types):
        """
        Configure dataloaders from preprocessed NPZ file(s).

        Supports two modes:
        1. Single NPZ with internal split: cg_config.fname_npz + ntrain/nvalid
        2. Pre-split NPZ files: cg_config.fname_train_npz + cg_config.fname_valid_npz

        Args:
            cg_cutoff: Cutoff distance for CG neighbor list
            num_cg_types: Number of CG bead types

        Returns:
            train_loader, valid_loader, uniq_type, enr_avg_per_type
        """
        from bam_torch.utils.cg_dataset import CGDataset

        if 'fname_train_npz' in self.cg_config:
            # Pre-split mode: separate train/valid NPZ files
            from bam_torch.utils.cg_dataset import get_dataloader_from_split_npz

            train_npz = self.cg_config['fname_train_npz']
            valid_npz = self.cg_config['fname_valid_npz']

            if self.rank == 0:
                print(f"\nData mode: Pre-split NPZ")
                print(f"  Train: {train_npz}")
                print(f"  Valid: {valid_npz}")

            train_loader, valid_loader, uniq_type, enr_avg_per_type = \
                get_dataloader_from_split_npz(
                    train_npz=train_npz,
                    valid_npz=valid_npz,
                    nbatch=self.json_data['nbatch'],
                    cutoff=cg_cutoff,
                    regress_forces=self.json_data['regress_forces'],
                    max_neigh=self.json_data.get('max_neigh'),
                    ntrain=self.json_data.get('ntrain'),
                    nvalid=self.json_data.get('nvalid'),
                    random_seed=self.json_data['NN'].get('data_seed', 42),
                    rank=self.rank,
                    world_size=self.world_size
                )

            dataset = CGDataset(train_npz)
        else:
            # Single NPZ with internal split
            from bam_torch.utils.cg_dataset import get_dataloader_from_npz

            npz_path = self.cg_config['fname_npz']

            if self.rank == 0:
                print(f"\nData mode: Preprocessed NPZ")
                print(f"  NPZ file: {npz_path}")

            train_loader, valid_loader, uniq_type, enr_avg_per_type = \
                get_dataloader_from_npz(
                    npz_path=npz_path,
                    ntrain=self.json_data['ntrain'],
                    nvalid=self.json_data['nvalid'],
                    nbatch=self.json_data['nbatch'],
                    cutoff=cg_cutoff,
                    random_seed=self.json_data['NN']['data_seed'],
                    regress_forces=self.json_data['regress_forces'],
                    max_neigh=self.json_data.get('max_neigh'),
                    rank=self.rank,
                    world_size=self.world_size,
                    bond_topology=self._get_bond_topology(npz_path),
                    graph_cache=self.json_data.get('graph_cache', False),
                )

            dataset = CGDataset(npz_path)

        # Store mapping config from NPZ metadata for checkpoint
        self.cg_mapping_config = dataset.metadata.get('mapping_config', {})

        return train_loader, valid_loader, uniq_type, enr_avg_per_type

    def _get_bond_topology(self, npz_path: str = None):
        """Get bond topology: config > NPZ metadata > None.

        Priority:
            1. cg_config.bond_topology (user-specified in input.json)
            2. NPZ metadata.bond_topology.global (auto-generated during mapping)
            3. None (no bond flag)
        """
        # 1. User-specified in config
        bt = self.cg_config.get('bond_topology')
        if bt is not None:
            return bt

        # 2. Auto-read from NPZ metadata
        if npz_path is not None:
            try:
                data = np.load(npz_path, allow_pickle=True)
                if 'metadata' in data:
                    meta = data['metadata'].item()
                    if isinstance(meta, dict) and 'bond_topology' in meta:
                        bt_meta = meta['bond_topology']
                        # Use 'global' key for single-resname systems
                        if isinstance(bt_meta, dict):
                            bt = bt_meta.get('global', None)
                            if bt is not None and self.rank == 0:
                                print(f"  Bond topology auto-loaded from NPZ metadata: "
                                      f"{bt['n_beads_per_mol']} beads/mol, "
                                      f"{len(bt['bonds'])} bonds/mol")
                            return bt
            except Exception:
                pass

        return None

    def _get_cg_mapping_config(self):
        """
        Get CG mapping configuration.

        Supports:
        - Preset names: 'water', 'methane', 'ethane', 'propane', 'butane',
                       'benzene', 'benzene_6bead', 'methanol', 'ethanol',
                       'co2', 'ammonia', 'acetone', 'dmso'
        - Custom mapping dictionary
        - 'auto': Auto-detect from trajectory

        Returns:
            CG mapping configuration dictionary
        """
        from bam_torch.utils.cg_mapping import CG_PRESETS

        mapping_config = self.cg_config.get('mapping', 'water')

        # Handle preset names (string)
        if isinstance(mapping_config, str):
            preset_name = mapping_config.lower()

            if preset_name == 'auto':
                # Auto-detection will be handled in get_dataloader_cg
                return {'auto': True}

            if preset_name in CG_PRESETS:
                if self.rank == 0:
                    preset = CG_PRESETS[preset_name]
                    print(f"\nUsing CG preset: {preset_name}")
                    print(f"  Formula: {preset.get('formula', 'N/A')}")
                    print(f"  Atoms per molecule: {preset.get('atoms_per_molecule', 'N/A')}")
                    print(f"  CG beads per molecule: {len(preset.get('beads', []))}")
                    print(f"  Method: {preset.get('method', 'com')}")
                return CG_PRESETS[preset_name].copy()
            else:
                available = list(CG_PRESETS.keys())
                raise ValueError(
                    f"Unknown CG mapping preset: '{mapping_config}'\n"
                    f"Available presets: {available}\n"
                    f"Or provide a custom mapping dictionary."
                )

        # Handle custom dictionary
        elif isinstance(mapping_config, dict):
            if self.rank == 0:
                print(f"\nUsing custom CG mapping:")
                print(f"  Formula: {mapping_config.get('formula', 'Custom')}")
                print(f"  Atoms per molecule: {mapping_config.get('atoms_per_molecule', 'N/A')}")
                print(f"  CG beads: {len(mapping_config.get('beads', []))}")
            return mapping_config

        else:
            raise ValueError(
                f"CG mapping must be a preset name (str) or dictionary, "
                f"got {type(mapping_config)}"
            )

    def configure_checkpoint(self):
        """
        Configure checkpoint with CG-specific information.

        Extends base checkpoint to include:
        - CG mapping configuration
        - CG type information

        Returns:
            loss_dict, ckpt
        """
        loss_dict, ckpt = super().configure_checkpoint()

        # Add CG-specific information
        ckpt['cg_config'] = self.cg_config
        ckpt['cg_mapping_config'] = getattr(self, 'cg_mapping_config', None)
        ckpt['is_cg_model'] = True
        ckpt['delta_learning'] = self.delta_learning
        ckpt['prior_config'] = self.prior_config

        # Initialize scale_shift as dictionaries (required by base_trainer.train())
        # Keys are element/type indices from enr_avg_per_element
        ckpt['train_scale_shift'] = {k: [] for k in self.enr_avg_per_element.keys()}
        ckpt['valid_scale_shift'] = {k: [] for k in self.enr_avg_per_element.keys()}
        ckpt['valid_scale_shift_origin'] = []

        return loss_dict, ckpt

    def set_model(self):
        """
        Set up the model for CG training.

        Uses the standard RACE model but with:
        - num_species = num_cg_types (number of CG bead types)
        - Potentially larger cutoff for CG interactions

        Returns:
            Configured model
        """
        model_config = self.json_data

        # CG-specific: use num_cg_types instead of num_species
        num_cg_types = model_config.get('num_cg_types', 1)

        cutoff = model_config.get('cutoff', 10.0)  # CG typically needs larger cutoff
        avg_num_neighbors = model_config.get('avg_num_neighbors', 50)  # More neighbors for CG

        hidden_irreps = o3.Irreps(
            model_config.get('hidden_channels', "64x0e+32x1o+16x2e")
        )
        features_dim = model_config.get('features_dim', 64)
        num_basis_func = model_config.get('num_radial_basis', 8)
        nlayers = model_config.get('nlayers', 3)
        max_ell = model_config.get('max_ell', 2)

        output_irreps = model_config.get('output_channels', "1x0e")
        active_fn = model_config.get('active_fn', "identity")
        regress_forces = model_config.get('regress_forces', "direct")
        if regress_forces == True:
            regress_forces = "direct"
        elif regress_forces == False:
            regress_forces = "false"

        # cuEquivariance config
        cueq_config = model_config.get('cueq_config')
        if cueq_config == None or cueq_config:
            try:
                import cuequivariance as cue
                import cuequivariance_torch as cuet
                CUET_AVAILABLE = True
            except ImportError:
                CUET_AVAILABLE = False
            if CUET_AVAILABLE:
                from bam_torch.model.wrapper_ops import CuEquivarianceConfig
                cueq_config = CuEquivarianceConfig(
                    enabled=True,
                    layout="ir_mul",
                    group="O3_e3nn",
                    optimize_all=True,
                )
                self.msg += f'\nequiv. lib.:\n\033[33m -- CuEquivariance\033[0m\n'
            else:
                cueq_config = None
                self.msg += f'\nequiv. lib.:\n\033[33m -- e3nn\033[0m\n'
        else:
            cueq_config = None
            self.msg += f'\nequiv. lib.:\n\033[33m -- e3nn\033[0m\n'

        # Use standard RACE model from registry
        from bam_torch.model import MODEL_REGISTRY
        model_name = model_config.get("model", "race").lower()
        model_cls = MODEL_REGISTRY.get(model_name)
        if model_cls is None:
            raise ValueError(f"Unknown model type: {model_name}")

        # Bond flag for CG systems (optional)
        use_bond_flag = model_config.get('use_bond_flag', False)

        # RACE interaction block: "slow"(기본, unweighted FullTensorProduct)
        # 또는 "fast"(weighted channel-wise TP -> openequivariance 가속 가능).
        # 둘은 가중치 호환이 안 되므로 바꾸면 재훈련이 필요하다.
        interaction_block = model_config.get('interaction_block', 'slow')

        # Create model with CG parameters
        model = model_cls(
            cutoff=cutoff,
            avg_num_neighbors=avg_num_neighbors,
            num_species=num_cg_types,  # Use CG types instead of atomic species
            max_ell=max_ell,
            num_basis_func=num_basis_func,
            hidden_irreps=hidden_irreps,
            nlayers=nlayers,
            features_dim=features_dim,
            output_irreps=output_irreps,
            active_fn=active_fn,
            regress_forces=regress_forces,
            cueq_config=cueq_config,
            use_bond_flag=use_bond_flag,
            interaction_block=interaction_block,
            compute_stress=model_config.get('compute_stress', True),
        )

        if self.rank == 0:
            print(f"\nCG Model Configuration:")
            print(f"  - Model type: {model_name.upper()}")
            print(f"  - Number of CG types: {num_cg_types}")
            print(f"  - Cutoff: {cutoff} Å")
            print(f"  - Hidden channels: {hidden_irreps}")
            print(f"  - Number of layers: {nlayers}")
            print(f"  - Interaction block: {interaction_block}")

        return model


    def predict_with_prior(self, positions, types, cell=None):
        """
        Predict forces using delta learning: F_total = F_prior + F_delta.

        This method should be used at inference time when delta learning was used.

        Args:
            positions: CG positions, shape (n_atoms, 3)
            types: CG bead types, shape (n_atoms,)
            cell: Unit cell, shape (3, 3) or None

        Returns:
            Total predicted forces, shape (n_atoms, 3)
        """
        if not self.delta_learning or self.prior_ff is None:
            raise RuntimeError("predict_with_prior requires delta_learning mode")

        # Convert to numpy if needed
        if isinstance(positions, torch.Tensor):
            positions_np = positions.detach().cpu().numpy()
        else:
            positions_np = positions

        if isinstance(types, torch.Tensor):
            types_np = types.detach().cpu().numpy()
        else:
            types_np = types

        if cell is not None and isinstance(cell, torch.Tensor):
            cell_np = cell.detach().cpu().numpy()
        else:
            cell_np = cell

        # Compute prior forces
        prior_forces = self.prior_ff.compute_forces(positions_np, types_np, cell_np)

        # Get model prediction (F_delta)
        # Note: This assumes the model is already loaded and in eval mode
        # The actual implementation would depend on how you call the model
        # This is a placeholder showing the concept

        return prior_forces  # + model_delta_forces

    @staticmethod
    def load_with_prior(checkpoint_path):
        """
        Load a trained CG model with its prior FF configuration.

        Args:
            checkpoint_path: Path to the checkpoint file

        Returns:
            model, prior_ff, cg_config
        """
        from bam_torch.utils.prior_ff import PriorForceField

        ckpt = torch.load(checkpoint_path, map_location='cpu')

        # Extract configurations
        cg_config = ckpt.get('cg_config', {})
        delta_learning = ckpt.get('delta_learning', False)
        prior_config = ckpt.get('prior_config', None)

        # Recreate prior FF if delta learning was used
        prior_ff = None
        if delta_learning and prior_config is not None:
            if isinstance(prior_config, str):
                prior_ff = PriorForceField.from_config({'type': prior_config})
            else:
                prior_ff = PriorForceField.from_config(prior_config)

        return ckpt, prior_ff, cg_config

    def scale_shift(self, preds, data, mode):
        """Skip scale-shift when enr_lambda=0 (force-only sub-graph training)."""
        e_lambda = self.json_data["NN"].get('enr_lambda', 1)
        if e_lambda == 0:
            return preds
        return super().scale_shift(preds, data, mode)


# For backward compatibility
CoarseGrainedTrainer = CGTrainer
