"""
Multihead Trainer for BAM-torch

N-Head Multihead Finetuning Trainer:
- Head 0: Target dataset (main finetuning)
- Head 1: Replay dataset (catastrophic forgetting prevention)
- Head 2+: Additional datasets (optional)

Uses RACEMultihead model from bam_torch.model.models
"""

import os
import gc
import signal
import atexit
import torch
import numpy as np
from typing import Dict, Any, List
from pathlib import Path
from datetime import datetime
from contextlib import nullcontext

from torch.nn.parallel import DistributedDataParallel as DDP
from e3nn import o3

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

from .base_trainer import BaseTrainer
from .loss import RMSELoss, HuberLoss, l2_regularization
from bam_torch.utils.utils import date, get_dataloader_multihead

try:
    from torch_ema import ExponentialMovingAverage
except ImportError:
    ExponentialMovingAverage = None


class MultiheadTrainer(BaseTrainer):
    """
    N-Head Multihead Finetuning Trainer

    Inherits from BaseTrainer and overrides:
    - Model creation (uses RACEMultihead)
    - Data loading (multiple datasets with head indices)
    - Loss computation (per-head weighted loss)
    - Checkpoint handling (foundation model loading)

    Compatible with GitHub latest BaseTrainer (setup() pattern)
    """

    def __init__(self, json_data: Dict[str, Any], rank: int = 0, world_size: int = 1):
        # Multihead config pre-processing (needed before super().__init__)
        self.multihead_config = json_data.get("multihead", {})
        if not self.multihead_config.get("enabled", False):
            raise ValueError("Multihead finetuning is not enabled in config")

        # datasets_config initiating
        self.datasets_config = self.multihead_config.get("datasets", [])
        if not self.datasets_config:
            raise ValueError("multihead.datasets must be specified and non-empty")

        # Head names and weights
        self.heads = [ds.get("name", f"head_{i}") for i, ds in enumerate(self.datasets_config)]
        self.num_heads = len(self.heads)
        self.loss_weights = [ds.get("loss_weight", 1.0) for ds in self.datasets_config]

        # Smoke test config
        self.smoke_config = json_data.get("smoke_test", {})
        self.smoke_enabled = self.smoke_config.get("enabled", False)
        self.smoke_max_batches = self.smoke_config.get("max_batches", 5)
        if self.smoke_enabled and rank == 0:
            print(f"\n⚠️ SMOKE TEST MODE: max_batches={self.smoke_max_batches}")

        # Extract the foundation model's setting
        self.foundation_config = self._extract_foundation_config(json_data, rank)

        # Set Batch logs directory
        self._setup_batch_logs(json_data, rank)

        super().__init__(json_data, rank, world_size)

    def _extract_foundation_config(self, json_data: Dict[str, Any], rank: int) -> Dict[str, Any]:
        """Return the input.json of the foundation model
        """
        foundation_path = json_data.get('NN', {}).get('foundation_model')
        if not foundation_path:
            raise ValueError("foundation_model must be specified in NN config for multihead finetuning")
        if not os.path.exists(foundation_path):
            raise FileNotFoundError(f"Foundation model not found: {foundation_path}")

        foundation_ckpt = torch.load(foundation_path, map_location='cpu', weights_only=False)
        foundation_json = foundation_ckpt.get('input.json', {})

        if rank == 0 and foundation_json:
            print(f"✓ Loaded foundation config from {foundation_path}")

        return foundation_json

    def setup(self):
        """Configure all core training components - Multihead version.

        Overrides BaseTrainer.setup() to:
        1. Load foundation model after model creation
        2. Use multihead-specific dataloader
        3. Configure EMA with multihead settings
        """
        self.set_random_seed()  # Reproducibility
        self.device = self.configure_device()
        self.model, self.n_params, _, self.start_epoch = self.configure_model()

        # Load the foundation model (if not restart)
        if not self.json_data['NN'].get('restart', False):
            self._load_foundation_model()

        self.optimizer = self.configure_optimizer()
        self.train_loader, self.valid_loader, self.uniq_element, self.enr_avg_per_element \
                       = self.configure_dataloader()
        self.scheduler = self.configure_scheduler()
        self.loss_fn, self.loss_config = self.configure_loss()
        self.log_config, self.log_interval, self.logger = self.configure_logger()
        self.loss_dict, self.ckpt = self.configure_checkpoint()
        self.ema = self.configure_exponential_moving_average()

    def _setup_batch_logs(self, json_data: Dict[str, Any], rank: int):
        """Set Batch logs directory and log-file
        """
        # Get Node ID and Local GPU rank
        if 'SLURM_NODEID' in os.environ:
            node_id = int(os.environ['SLURM_NODEID'])
        elif 'NODE_RANK' in os.environ:
            node_id = int(os.environ['NODE_RANK'])
        else:
            node_id = 0

        if 'SLURM_LOCALID' in os.environ:
            local_rank = int(os.environ['SLURM_LOCALID'])
        elif 'LOCAL_RANK' in os.environ:
            local_rank = int(os.environ['LOCAL_RANK'])
        else:
            local_rank = rank

        # Set batch size and log-directory/file per rank
        batch_log_root = Path(json_data.get('batch_size_log_root', 'batch_logs'))
        self.batch_log_dir = batch_log_root / f"rank_{rank}"
        self.batch_log_dir.mkdir(parents=True, exist_ok=True)

        # Set logfile per GPU (generated in batch_log_dir)
        log_filename = self.batch_log_dir / f'node{node_id}_gpu{local_rank}_global{rank}.log'
        self.gpu_log = open(log_filename, 'w')
        atexit.register(lambda: self.gpu_log.close() if hasattr(self, 'gpu_log') and not self.gpu_log.closed else None)

        print(f"[Rank {rank}] Initialized - Node: {node_id}, Local GPU: {local_rank}",
              file=self.gpu_log, flush=True)

        batch_log_filename = self.batch_log_dir / "batch_sizes.log"
        self.batch_size_log = open(batch_log_filename, 'w')
        self.batch_size_log.write(
            f"# Batch size log for rank {rank} (node {node_id}, local GPU {local_rank})\n"
        )
        self.batch_size_log.write("# timestamp,epoch,mode,file,batch_idx,graphs,total_nodes,total_edges,head_composition\n")
        self.batch_size_log.flush()
        atexit.register(lambda: self.batch_size_log.close() if hasattr(self, 'batch_size_log') and not self.batch_size_log.closed else None)

        if rank == 0:
            print(f"✓ Batch logs directory created: {batch_log_root}")

    def _log_batch_size(self, mode: str, batch_idx: int, data, epoch: int = 0):
        """Report the batch size information to log-file
        """
        if not hasattr(self, 'batch_size_log') or self.batch_size_log.closed:
            return

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # The number of graph in the batch
        num_graphs = data.ptr.numel() - 1 if hasattr(data, 'ptr') else 1

        # The total number of nodes and edges
        total_nodes = data.num_nodes if hasattr(data, 'num_nodes') else len(data.positions)
        total_edges = data.edge_index.shape[1] if hasattr(data, 'edge_index') else 0

        # Head configuration
        if hasattr(data, 'head'):
            head_counts = {}
            heads = data.head.flatten().tolist()
            for h in heads:
                head_counts[h] = head_counts.get(h, 0) + 1
            head_str = str(head_counts)
        else:
            head_str = "N/A"

        log_line = f"{timestamp},{epoch},{mode},batch,{batch_idx},{num_graphs},{total_nodes},{total_edges},{head_str}\n"
        self.batch_size_log.write(log_line)
        self.batch_size_log.flush()

    def set_model(self):
        """
        Override: Generate RACEMultihead model
        Prioritize the foundation model configuration to ensure shape compatibility
        """
        model_config = self.json_data

        # Use the foundation configuration first if available
        if self.foundation_config:
            fc = self.foundation_config
            if self.rank == 0:
                print(f"\n\033[36mUsing Foundation model config:\033[0m")
                print(f"  - hidden_channels: {fc.get('hidden_channels', 'N/A')}")
                print(f"  - features_dim: {fc.get('features_dim', 'N/A')}")
                print(f"  - nlayers: {fc.get('nlayers', 'N/A')}")
                print(f"  - cutoff: {fc.get('cutoff', 'N/A')}")

            # Load from the foundation config
            hidden_irreps_str = fc.get('hidden_channels', model_config['hidden_channels'])
            hidden_irreps = o3.Irreps(hidden_irreps_str)
            features_dim = fc.get('features_dim', model_config['features_dim'])
            nlayers = fc.get('nlayers', model_config['nlayers'])
            cutoff = fc.get('cutoff', model_config['cutoff'])
            num_basis_func = fc.get('num_radial_basis', model_config['num_radial_basis'])
            max_ell = fc.get('max_ell', model_config['max_ell'])
        else:
            # If not exist the foundation config, load from input.json
            hidden_irreps = o3.Irreps(model_config['hidden_channels'])
            features_dim = model_config['features_dim']
            nlayers = model_config['nlayers']
            cutoff = model_config['cutoff']
            num_basis_func = model_config['num_radial_basis']
            max_ell = model_config['max_ell']

        # Load the remaining (dataset-related) settings from input.json
        avg_num_neighbors = model_config['avg_num_neighbors']
        num_species = model_config['num_species']

        output_irreps = model_config.get('output_channels', "1x0e")
        active_fn = model_config.get('active_fn', "identity")

        regress_forces = model_config.get('regress_forces')
        if regress_forces == True:
            regress_forces = "autograd"
        elif regress_forces == False:
            regress_forces = "false"

        mlp_irreps = o3.Irreps(f"{features_dim}x0e")

        # Generate the RACEUnified model (Single-head + Multihead)
        from bam_torch.model.models import RACEUnified

        model = RACEUnified(
            cutoff=cutoff,
            avg_num_neighbors=avg_num_neighbors,
            num_species=num_species,
            max_ell=max_ell,
            num_basis_func=num_basis_func,
            hidden_irreps=hidden_irreps,
            nlayers=nlayers,
            features_dim=features_dim,
            output_irreps=output_irreps,
            active_fn=active_fn,
            radial_MLP=[64, 64],
            MLP_irreps=mlp_irreps,
            regress_forces=regress_forces,
            compute_stress=True,
            heads=self.heads,
            cueq_config=None,
            l_separated_layer_norm=model_config.get('l_separated_layer_norm', False),
        )

        self.msg += f'\n\033[33m -- Multihead model with {self.num_heads} heads: {self.heads}\033[0m\n'

        return model

    def _load_foundation_enr_avg(self):
        """
        Load enr_avg_per_element from the foundation model

        Returns:
            enr_avg_per_element: {species_index: energy}
            uniq_element: {atomic_number: species_index}
        """
        foundation_path = self.json_data.get('NN', {}).get('foundation_model')
        if not foundation_path or not os.path.exists(foundation_path):
            if self.rank == 0:
                print("⚠️ No foundation model specified, cannot load foundation enr_avg_per_element")
            return None, None

        try:
            foundation_ckpt = torch.load(foundation_path, map_location='cpu', weights_only=False)

            enr_avg_per_element = foundation_ckpt.get('enr_avg_per_element')
            uniq_element = foundation_ckpt.get('uniq_element')

            if enr_avg_per_element is None or uniq_element is None:
                if self.rank == 0:
                    print("⚠️ Foundation model does not contain enr_avg_per_element or uniq_element")
                return None, None

            if self.rank == 0:
                print(f"✓ Loaded foundation enr_avg_per_element ({len(enr_avg_per_element)} species)")

            return enr_avg_per_element, uniq_element

        except Exception as e:
            if self.rank == 0:
                print(f"⚠️ Failed to load foundation enr_avg_per_element: {e}")
            return None, None

    def _load_foundation_model(self):
        """Load foundation model weights and extend the readout layers
        """
        foundation_path = self.json_data['NN']['foundation_model']

        if self.rank == 0:
            print(f"\n{'='*80}")
            print(f"Loading Foundation model: {foundation_path}")
            print(f"{'='*80}\n")

        # BAM-torch pkl format: {'params': state_dict, ...}
        foundation_ckpt = torch.load(foundation_path, map_location='cpu', weights_only=False)
        foundation_state = foundation_ckpt['params']

        # Refer present model
        model = self.model.module if self.ddp else self.model

        self._load_from_state_dict(model, foundation_state, self.num_heads)


    def _load_from_state_dict(self, model, foundation_state, num_heads):
        """
        Fallback: load parameters from state_dict

        Use when only the state_dict is available (no model object)
        Reference: https://github.com/ACEsuit/mace/blob/main/mace/tools/finetuning_utils.py

        Parameters structure of the foundation model's readout block:
        - readouts.X.linear_1.weight: [hidden_dim_0e * MLP_dim] = [128 * 64] = [8192]
        - readouts.X.linear_1.output_mask: [MLP_dim] = [64]
        - readouts.X.linear_2.weight: [MLP_dim * output_dim] = [64 * 1] = [64]
        - readouts.X.linear_2.output_mask: [output_dim] = [1]

        Target model (num_heads=2):
        - readouts.X.linear_1.weight: [hidden_dim_0e * MLP_dim * num_heads] = [128 * 64 * 2] = [16384]
        - readouts.X.linear_1.output_mask: [MLP_dim * num_heads] = [128]
        - readouts.X.linear_2.weight: [MLP_dim * num_heads * output_dim * num_heads] = [64 * 2 * 1 * 2] = [256]
        - readouts.X.linear_2.output_mask: [output_dim * num_heads] = [2]
        """
        if self.rank == 0:
            print("Loading from state_dict (fallback method)...")

        current_state = model.state_dict()
        loaded_count = 0
        readout_expanded = 0
        skipped_readout = []
        # Extract MLP_dim and hidden_dim_0e from the foundation NonLinearReadoutBlock
        mlp_dim = next(p.numel() for n, p in foundation_state.items()
                       if 'linear_2.weight' in n and p.numel() > 0)
        linear_1_numel = next(p.numel() for n, p in foundation_state.items()
                              if 'linear_1.weight' in n and p.numel() > 0)
        hidden_dim_0e = linear_1_numel // mlp_dim

        target_output_dim = num_heads

        if self.rank == 0:
            print(f"  - hidden_dim_0e: {hidden_dim_0e} (from linear_1.weight / MLP_dim)")
            print(f"  - MLP_dim: {mlp_dim} (from linear_2.weight)")
            print(f"  - num_heads: {num_heads}")

        for name, param in foundation_state.items():
            clean_name = name[7:] if name.startswith('module.') else name

            if clean_name not in current_state:
                continue

            target_param = current_state[clean_name]

            # Readout parameters require extension
            if 'readouts' in clean_name:
                if param.numel() == 0:
                    # Empty parameter (no bias)
                    continue

                expanded = None
                scale_factor = 1.0

                if 'linear_1.weight' in clean_name:
                    # Foundation: [hidden_0e * MLP_dim] = [8192]
                    # Target: [hidden_0e * MLP_dim * num_heads] = [16384]
                    # (MLP_dim, hidden_0e) -> (MLP_dim * num_heads, hidden_0e)
                    if param.numel() == hidden_dim_0e * mlp_dim:
                        expanded = param.view(mlp_dim, hidden_dim_0e).repeat(num_heads, 1).flatten()
                    else:
                        # Fallback
                        expanded = param.repeat(num_heads)

                elif 'linear_1.output_mask' in clean_name:
                    # Foundation: [MLP_dim] = [64]
                    # Target: [MLP_dim * num_heads] = [128]
                    expanded = param.repeat(num_heads)

                elif 'linear_1.bias' in clean_name:
                    # Foundation: [] or [MLP_dim]
                    # Target: [MLP_dim * num_heads]
                    if param.numel() > 0:
                        expanded = param.repeat(num_heads)

                elif 'linear_2.weight' in clean_name:
                    # Foundation: [MLP_dim * 1] = [64]
                    # Target: [MLP_dim * num_heads * num_heads] = [256]
                    # Scailing: / sqrt(MLP_dim / target_output_dim)
                    if param.numel() == mlp_dim:
                        # view(1, MLP_dim).repeat(num_heads, num_heads) -> [num_heads, MLP_dim * num_heads]
                        expanded = param.view(1, mlp_dim).repeat(num_heads, num_heads).flatten()
                        # Scailing
                        scale_factor = (mlp_dim / target_output_dim) ** 0.5
                        expanded = expanded / scale_factor
                    else:
                        expanded = param.repeat(num_heads * num_heads)

                elif 'linear_2.output_mask' in clean_name:
                    # Foundation: [1]
                    # Target: [num_heads] = [2]
                    expanded = param.repeat(num_heads)

                elif 'linear_2.bias' in clean_name:
                    # Foundation: [] or [1]
                    # Target: [num_heads]
                    if param.numel() > 0:
                        expanded = param.repeat(num_heads)

                elif 'linear.weight' in clean_name:
                    # LinearReadoutBlock: [hidden_dim_0e * output_dim]
                    if param.numel() % hidden_dim_0e == 0:
                        output_dim_linear = param.numel() // hidden_dim_0e
                        expanded = param.view(hidden_dim_0e, output_dim_linear).repeat(1, num_heads).flatten()
                        # Scailing
                        if output_dim_linear == 1:
                            scale_factor = (hidden_dim_0e / target_output_dim) ** 0.5
                            expanded = expanded / scale_factor
                    else:
                        expanded = param.repeat(num_heads)

                elif 'linear.output_mask' in clean_name:
                    expanded = param.repeat(num_heads)

                if expanded is not None:
                    if expanded.shape == target_param.shape:
                        current_state[clean_name].copy_(expanded)
                        loaded_count += 1
                        readout_expanded += 1
                        if self.rank == 0 and scale_factor != 1.0:
                            print(f"    {clean_name}: scaled by 1/{scale_factor:.2f}")
                        elif self.rank == 0:
                            print(f"    {clean_name}: expanded {param.shape} -> {expanded.shape}")
                    elif target_param.shape == param.shape:
                        current_state[clean_name].copy_(param)
                        loaded_count += 1
                    else:
                        skipped_readout.append(
                            f"{clean_name}: {param.shape} -> {expanded.shape} (target: {target_param.shape})"
                        )
            else:
                # For general parameters, directly copy when the shapes match
                if target_param.shape == param.shape:
                    current_state[clean_name].copy_(param)
                    loaded_count += 1

        model.load_state_dict(current_state, strict=False)

        if self.rank == 0:
            print(f"✓ Foundation model loaded")
            print(f"  - Loaded {loaded_count} parameters")
            print(f"  - Readout expanded: {readout_expanded} parameters")
            if skipped_readout:
                print(f"  - Readout skipped (shape mismatch): {len(skipped_readout)}")
                for s in skipped_readout[:5]:
                    print(f"    {s}")

    def configure_dataloader(self):
        """
        Override: Configure multihead data loaders (per-head + mixed).

        Returns mixed loaders for backward compatibility.
        Per-head loaders stored as self.per_head_train_loaders / self.per_head_valid_loaders.
        """
        # Load enr_avg_per_element from the foundation model (for replay)
        foundation_enr_avg, foundation_uniq_element = self._load_foundation_enr_avg()

        # Smoke test config
        smoke_config = self.json_data.get('smoke_test', {})

        (train_loader, valid_loader,
         per_head_train_loaders, per_head_valid_loaders,
         uniq_element, enr_avg_per_element, per_head_enr_avg) = \
            get_dataloader_multihead(
                self.datasets_config,
                self.json_data['cutoff'],
                self.json_data['nbatch'],
                self.json_data.get('regress_forces', True),
                self.json_data.get('max_neigh'),
                foundation_enr_avg,
                foundation_uniq_element,
                self.rank,
                self.world_size,
                smoke_config=smoke_config
            )

        # Store per-head loaders (JAX-style separate streams)
        self.per_head_train_loaders = per_head_train_loaders
        self.per_head_valid_loaders = per_head_valid_loaders

        # Store per-head E0s for checkpoint
        self.per_head_enr_avg = per_head_enr_avg

        return train_loader, valid_loader, uniq_element, enr_avg_per_element

    def configure_loss(self, reduction='mean'):
        nn_config = self.json_data.get("NN", {})
        loss_config = nn_config.get("loss_config", {})

        if not loss_config:
            if self.json_data.get("regress_forces"):
                loss_config = {'energy_loss': 'huber', 'force_loss': 'huber', 'stress_loss': 'huber'}
            else:
                loss_config = {'energy_loss': 'huber'}

        s_lambda = nn_config.get("str_lambda", 0)
        if loss_config.get('stress_loss') is None and s_lambda:
            loss_config['stress_loss'] = 'mse'

        huber_delta = loss_config.get('huber_delta', 0.01)

        loss_fn = {}
        for loss_key in ['energy_loss', 'force_loss', 'stress_loss']:
            loss_name = loss_config.get(loss_key)
            if loss_name in ['l1', 'L1', 'mae', 'MAE']:
                loss_fn[loss_key] = torch.nn.L1Loss(reduction=reduction)
            elif loss_name in ['mse', 'MSE']:
                loss_fn[loss_key] = torch.nn.MSELoss(reduction=reduction)
            elif loss_name in ['rmse', 'RMSE']:
                loss_fn[loss_key] = RMSELoss(reduction=reduction)
            elif loss_name in ['huber', 'Huber', 'HUBER']:
                loss_fn[loss_key] = HuberLoss(huber_delta=huber_delta)
            else:
                loss_fn[loss_key] = None

        return loss_fn, loss_config

    def compute_loss(self, preds, data):
        """
        Override: compute weighted loss per head
        """
        lambda_config = self.json_data["NN"]
        e_lambda = lambda_config.get('enr_lambda', 1.0)
        f_lambda = lambda_config.get('frc_lambda', 1.0)
        s_lambda = lambda_config.get('str_lambda', 1.0)
        lambd = lambda_config.get('l2_lambda', 0)

        loss = {"loss": []}

        # Load heads and weights from the config
        if 'config_head' in data:
            config_heads = data['config_head'].flatten()
        elif 'head' in data:
            # If the head is graph-level, expand it to batch level
            if hasattr(data, 'batch'):
                # Convert per-graph heads to per-configuration heads
                ptr = data.get('ptr')
                if ptr is not None and ptr.numel() > 1:
                    batch_size = ptr.numel() - 1
                    config_heads = torch.zeros(batch_size, dtype=torch.long, device=data['head'].device)
                    # Use the head of the first atom in each configuration
                    for i in range(batch_size):
                        start_idx = ptr[i].item()
                        config_heads[i] = data['head'][data['batch'] == i][0] if (data['batch'] == i).any() else 0
                else:
                    config_heads = data['head'].flatten()
            else:
                config_heads = data['head'].flatten()
        else:
            config_heads = torch.zeros(preds['energy'].shape[0], dtype=torch.long, device=preds['energy'].device)

        # Load weight
        if 'weight' in data:
            config_weights = data['weight'].flatten()
        else:
            config_weights = torch.ones_like(config_heads, dtype=torch.float)

        # Energy loss - HuberLoss with num_atoms normalization (like mp_trainer_phg)
        energy_target = data["energy"].flatten()
        energy_pred = preds["energy"].flatten()

        # Per-sample loss using HuberLoss (matching mp_trainer_phg)
        if self.loss_fn.get("energy_loss") is not None:
            loss["loss_e"] = self.loss_fn["energy_loss"](
                energy_pred,
                energy_target,
                tag="energy",
                num_atoms=data["num_nodes"]
            )
            loss["loss"].append(e_lambda * loss["loss_e"])

        # Force loss - HuberLoss (matching mp_trainer_phg)
        if "forces" in preds and self.loss_fn.get('force_loss') is not None:
            force_target = data["forces"].flatten()
            force_pred = preds["forces"].flatten()
            loss["loss_f"] = self.loss_fn["force_loss"](
                force_pred,
                force_target,
                tag="forces"
            )
            loss["loss"].append(f_lambda * loss["loss_f"])

        if "stress" in preds and preds["stress"] is not None and self.loss_fn.get('stress_loss') is not None:
            stress_target = data.get("stress")
            if stress_target is not None:
                stress_pred = preds["stress"].flatten()
                stress_target = stress_target.flatten()
                loss["loss_s"] = self.loss_fn["stress_loss"](
                    stress_pred,
                    stress_target,
                    tag="stress"
                )
                loss["loss"].append(s_lambda * loss["loss_s"])

        # L2 regularization
        if lambd:
            params = self.model.parameters()
            loss["loss_l2"] = l2_regularization(params)
            loss["loss"].append(lambd * loss["loss_l2"])

        # Total loss
        loss["loss"] = sum(loss["loss"])

        # Per-head loss tracking (for logging only - using simple MSE for monitoring)
        loss["head_losses"] = {}
        unique_heads = torch.unique(config_heads)
        for head_val in unique_heads:
            head_idx = int(head_val.item())
            head_mask = (config_heads == head_idx)
            if head_mask.any():
                # Per-atom normalized energy loss for monitoring
                head_energy_diff = energy_pred[head_mask] - energy_target[head_mask]
                # Get num_atoms for this head's samples
                ptr = data.get('ptr')
                if ptr is not None:
                    num_atoms_per_config = ptr[1:] - ptr[:-1]
                    head_num_atoms = num_atoms_per_config[head_mask]
                    head_energy_loss = ((head_energy_diff / head_num_atoms) ** 2).mean()
                else:
                    head_energy_loss = (head_energy_diff ** 2).mean()

                head_loss_dict = {
                    "loss_e": head_energy_loss.detach()
                }

                # Compute force loss for each head
                head_force_loss = None
                if "forces" in preds and self.loss_fn.get('force_loss') is not None:
                    if 'batch' in data:
                        batch_indices = data['batch']
                        # Identify atoms belonging to the corresponding head
                        atom_head_mask = head_mask[batch_indices]
                        if atom_head_mask.any():
                            force_diff = preds["forces"][atom_head_mask] - data["forces"][atom_head_mask]
                            head_force_loss = (force_diff ** 2).sum(dim=-1).mean()
                            head_loss_dict["loss_f"] = head_force_loss.detach()

                # Compute total loss per head (energy + force)
                head_total_loss = e_lambda * head_energy_loss
                if head_force_loss is not None:
                    head_total_loss = head_total_loss + f_lambda * head_force_loss
                head_loss_dict["loss"] = head_total_loss.detach()

                loss["head_losses"][head_idx] = head_loss_dict

        return loss

    def scale_shift(self, preds, data, mode):
        """Override: apply scale_shift per head (GitHub BaseTrainer compatible).
        """
        energy_target = data["energy"].flatten()
        energy_predict = preds["energy"].flatten()

        # Get the corresponding head per configuration
        if 'config_head' in data:
            config_heads = data['config_head'].flatten()
        else:
            ptr = data.get('ptr')
            if ptr is not None and ptr.numel() > 1:
                batch_size = ptr.numel() - 1
                config_heads = torch.zeros(batch_size, dtype=torch.long, device=energy_target.device)
            else:
                config_heads = torch.zeros_like(energy_target, dtype=torch.long)

        unique_heads = torch.unique(config_heads)

        for head_val in unique_heads:
            head_idx = int(head_val.item())
            head_mask = (config_heads == head_idx)

            if not head_mask.any():
                continue

            # Compute shift per head
            head_target = energy_target[head_mask]
            head_predict = energy_predict[head_mask]

            shift_enr = head_target.mean() - head_predict.mean()
            preds["energy"][head_mask] = head_predict + shift_enr

            # Record scale_shift per-head
            if mode == 'train':
                self.ckpt['train_scale_shift'].append(shift_enr.detach().cpu())
                # Per-head tracking
                if 'per_head_scale_shift' not in self.ckpt:
                    self.ckpt['per_head_scale_shift'] = {}
                if head_idx not in self.ckpt['per_head_scale_shift']:
                    self.ckpt['per_head_scale_shift'][head_idx] = []
                self.ckpt['per_head_scale_shift'][head_idx].append(shift_enr.detach().cpu().item())
            elif mode == 'valid':
                self.ckpt['valid_scale_shift'].append(shift_enr.detach().cpu())

        return preds

    def train(self):
        """Main training loop for BAM models - Multihead version.

        Follows BAM-JAX architecture:
        - Per-head gradient computation with weighted accumulation
        - Per-head independent validation
        - Best / latest checkpoint saving with epoch numbers
        - Emergency checkpointing on SIGTERM / SIGINT
        """
        # Initial test
        self.initial_test()

        # Create checkpoints directory (following BAM-JAX pattern)
        self._ckpt_dir = 'checkpoints'
        self._best_ckpt_path = None
        self._latest_ckpt_path = None
        if self.rank == 0:
            os.makedirs(self._ckpt_dir, exist_ok=True)

        # Register emergency checkpoint handler (JAX EmergencyCheckpointer)
        self._emergency_requested = False
        self._register_emergency_handler()

        # Print logger head
        if self.rank == 0:
            self.logger.print_logger_head()

        # Main training loop
        nepoch = self.json_data['NN']['nepoch']

        for epoch in range(nepoch):
            # Check emergency stop request
            if self._emergency_requested:
                if self.rank == 0:
                    print("\n[Emergency] Signal received — saving emergency checkpoint and exiting.")
                    ckpt = self._build_current_checkpoint(
                        max(epoch - 1, 0), {}, {'loss': self.loss_test_min})
                    emergency_path = os.path.join(self._ckpt_dir, 'ckpt_emergency.pkl')
                    torch.save(ckpt, emergency_path)
                    print(f"  Emergency checkpoint saved: {emergency_path}")
                break

            # Update criterion
            try:
                base_model = self.model.module if self.ddp else self.model
                base_model.update_criterion_value(epoch + self.start_epoch + 1)
            except:
                pass

            # Train (JAX-style per-head gradient accumulation)
            epoch_loss_train = self.train_one_epoch(mode='train', epoch=epoch+self.start_epoch)

            if self.ddp:
                torch.distributed.barrier()

            if (epoch+1) % self.log_interval == 0:
                # Evaluate with EMA-averaged parameters (JAX-style per-head validation)
                param_context = (
                    self.ema.average_parameters() if self.ema is not None else nullcontext()
                )
                with param_context:
                    epoch_loss_valid = self.validate_per_head(epoch=epoch+self.start_epoch)

                    if self.ddp:
                        torch.distributed.barrier()

                # Snapshot/log/save with live (non-EMA) parameters
                if self.rank == 0:
                    actual_epoch = epoch + 1 + self.start_epoch

                    # Save best checkpoint if validation loss improved
                    if epoch_loss_valid['loss'] < self.loss_test_min:
                        self.update_check_point(epoch, epoch_loss_train, epoch_loss_valid)
                        self.loss_test_min = epoch_loss_valid['loss']
                        self._save_named_checkpoint(self.ckpt, 'best', actual_epoch)

                    # Always save latest checkpoint
                    latest_ckpt = self._build_current_checkpoint(
                        epoch, epoch_loss_train, epoch_loss_valid
                    )
                    self._save_named_checkpoint(latest_ckpt, 'latest', actual_epoch)

                    # Print epoch loss
                    self.print_logger(epoch, epoch_loss_train, epoch_loss_valid)

                    # Free GPU memory
                    torch.cuda.empty_cache()
                    gc.collect()

                # Update scheduler (learning rate)
                metrics = None
                scheduler_cfg = self.json_data.get("scheduler", {})
                if isinstance(scheduler_cfg, dict) \
                        and scheduler_cfg.get("scheduler") == "ReduceLROnPlateau":
                    metrics = epoch_loss_valid['loss']
                self.scheduler.step(metrics, epoch)

    # ------------------------------------------------------------------
    # Emergency checkpoint handler (JAX EmergencyCheckpointer)
    # ------------------------------------------------------------------
    def _register_emergency_handler(self):
        """Register SIGTERM/SIGINT handlers for emergency checkpoint saving."""
        def _handler(signum, frame):
            sig_name = signal.Signals(signum).name
            if self.rank == 0:
                print(f"\n[Emergency] Received {sig_name} — will save checkpoint at next epoch boundary.")
            self._emergency_requested = True

        signal.signal(signal.SIGTERM, _handler)
        signal.signal(signal.SIGINT, _handler)

    def _build_current_checkpoint(self, epoch, epoch_loss_train, epoch_loss_valid):
        """Build a checkpoint dict with the current model state (for latest saving)."""
        state_dict = self.model.state_dict()
        if self.ddp:
            state_dict = self.model.module.state_dict()
        ema_state = None
        if self.ema is not None:
            ema_state = self.ema.state_dict()

        ckpt = {
            'loss': {
                'epoch': epoch + 1 + self.start_epoch,
                'train': epoch_loss_train.get('loss', float('nan')),
                'valid': epoch_loss_valid.get('loss', float('nan')),
            },
            'params': state_dict,
            'opt_state': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'uniq_element': self.ckpt.get('uniq_element'),
            'enr_avg_per_element': self.ckpt.get('enr_avg_per_element'),
            'input.json': self.json_data,
            'train_scale_shift': self.ckpt.get('train_scale_shift', []),
            'valid_scale_shift': self.ckpt.get('valid_scale_shift', []),
            'valid_scale_shift_origin': self.ckpt.get('valid_scale_shift_origin', []),
            'ema_state': ema_state,
        }
        if hasattr(self, 'per_head_enr_avg'):
            ckpt['per_head_enr_avg'] = self.per_head_enr_avg
        if 'per_head_scale_shift' in self.ckpt:
            ckpt['per_head_scale_shift'] = self.ckpt['per_head_scale_shift']
        return ckpt

    def _save_named_checkpoint(self, ckpt_data, tag, epoch):
        """Save checkpoint as ckpt_{tag}_epoch_{NN}.pkl, removing the previous file with same tag.

        Args:
            ckpt_data: checkpoint dictionary to save
            tag: 'best' or 'latest'
            epoch: epoch number (appended as _epoch_NN)
        """
        attr = f'_{tag}_ckpt_path'
        old_path = getattr(self, attr, None)
        if old_path and os.path.exists(old_path):
            os.remove(old_path)

        fname = os.path.join(self._ckpt_dir, f'ckpt_{tag}_epoch_{epoch:02d}.pkl')
        torch.save(ckpt_data, fname)
        setattr(self, attr, fname)

        if tag == 'best':
            val_loss = ckpt_data['loss']['valid']
            if isinstance(val_loss, torch.Tensor):
                val_loss = val_loss.item()
            print(f'  *** New best checkpoint saved: {fname} (val_loss: {val_loss:.6f}) ***')
        else:
            print(f'  Latest checkpoint saved: {fname}')

    # Initial_test: use BaseTrainer as-is (pass the epoch argument to train_one_epoch)
    def initial_test(self):
        """Run a preliminary test epoch and record the initial reference loss."""
        param_context = (
            self.ema.average_parameters() if self.ema is not None else nullcontext()
        )
        with param_context:
            epoch_loss_test = self.train_one_epoch(mode='test', epoch=0)
            if self.ddp:
                torch.distributed.barrier()
            self.loss_test_min = epoch_loss_test['loss']

    # Update_check_point: additionally save the EMA state
    def update_check_point(self, epoch, epoch_loss_train, epoch_loss_valid):
        """Update checkpoint with current training state (+ EMA state + per-head E0s).
        """
        super().update_check_point(epoch, epoch_loss_train, epoch_loss_valid)
        # Store per-head E0s (used during evaluation)
        if hasattr(self, 'per_head_enr_avg'):
            self.ckpt['per_head_enr_avg'] = self.per_head_enr_avg

    def train_one_epoch(self, mode='train', data_loader=None, epoch=0):
        """Train/validate one epoch — JAX-style per-head processing.

        Training mode:
            For each step, loads one batch per head, computes per-head gradients,
            accumulates with head loss_weights, then applies one optimizer step.
            (Mirrors JAX make_per_head_grad_step + make_accumulate_and_update_step)

        Validation / test mode:
            Falls back to mixed-loader evaluation for initial_test compatibility.
            Per-head validation is handled separately by validate_per_head().
        """
        if mode == 'train':
            return self._train_one_epoch_perhead(epoch)
        else:
            return self._eval_one_epoch_mixed(mode, data_loader, epoch)

    # ------------------------------------------------------------------
    # Training: JAX-style per-head gradient accumulation
    # ------------------------------------------------------------------
    def _train_one_epoch_perhead(self, epoch):
        """Per-head gradient computation and weighted accumulation (JAX pattern)."""
        self.model.train()
        self.ckpt['train_scale_shift'] = []
        loss_log_config = self.log_config['train']
        grad_clip = self.json_data.get('NN', {}).get('grad_clip', 1.0)

        # Per-head iterators (JAX HeadBatchStream equivalent)
        head_iters = {h: iter(loader) for h, loader in self.per_head_train_loaders.items()}
        steps_per_epoch = max(len(loader) for loader in self.per_head_train_loaders.values())

        # Accumulators
        epoch_loss_dict = {key: [] for key in loss_log_config if not key.startswith('head_')}
        head_loss_accum = {h: {'loss': [], 'loss_e': [], 'loss_f': []} for h in range(self.num_heads)}

        if self.rank == 0 and tqdm is not None:
            step_iter = tqdm(range(steps_per_epoch), desc=f"Epoch {epoch} [train]", leave=True)
        else:
            step_iter = range(steps_per_epoch)

        for step in step_iter:
            if self.smoke_enabled and step >= self.smoke_max_batches:
                if self.rank == 0 and step == self.smoke_max_batches:
                    print(f"  [SMOKE] Stopping after {self.smoke_max_batches} steps")
                break

            self.optimizer.zero_grad()
            total_weight = 0.0
            step_loss_sum = 0.0

            # --- Per-head forward + backward (gradient accumulation) ---
            per_head_loss_dicts = {}
            for head_idx in sorted(head_iters.keys()):
                # Get next batch (cycle if exhausted — JAX HeadBatchStream pattern)
                try:
                    data = next(head_iters[head_idx])
                except StopIteration:
                    head_iters[head_idx] = iter(self.per_head_train_loaders[head_idx])
                    data = next(head_iters[head_idx])

                data = self.move_to_device(data, self.device)
                self._log_batch_size('train', step, data, epoch)

                # Forward
                preds = self.model(data, True)
                preds = self.scale_shift(preds, data, 'train')
                loss_dict = self.compute_loss(preds, data)

                # Weighted backward: scale loss by head's grad_weight before backward
                head_weight = self.loss_weights[head_idx]
                scaled_loss = loss_dict['loss'] * head_weight
                scaled_loss.backward()  # gradients accumulate across heads

                total_weight += head_weight
                step_loss_sum += loss_dict['loss'].detach().item() * head_weight
                per_head_loss_dicts[head_idx] = loss_dict

            # --- Normalize accumulated gradients by total weight (JAX pattern) ---
            if total_weight > 0:
                for param in self.model.parameters():
                    if param.grad is not None:
                        param.grad.div_(total_weight)

            # Clip by global norm (JAX: optax.clip_by_global_norm)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=grad_clip)

            self.optimizer.step()

            if self.ema is not None:
                self.ema.update()

            # --- Logging ---
            weighted_loss = step_loss_sum / total_weight if total_weight > 0 else 0.0

            if self.rank == 0 and tqdm is not None and hasattr(step_iter, 'set_postfix'):
                step_iter.set_postfix({'loss': f"{weighted_loss:.4f}"})

            # Aggregate step-level losses
            for l in loss_log_config:
                if l.startswith('head_'):
                    continue
                # Combine from all heads (weighted mean)
                vals = []
                for hd, ld in per_head_loss_dicts.items():
                    v = ld.get(l)
                    if v is not None:
                        v = v.detach().cpu() if isinstance(v, torch.Tensor) else v
                        vals.append(v * self.loss_weights[hd])
                if vals:
                    epoch_loss_dict[l].append(sum(vals) / total_weight)

            # Per-head loss tracking
            for head_idx, ld in per_head_loss_dicts.items():
                if 'head_losses' in ld and head_idx in ld['head_losses']:
                    hl = ld['head_losses'][head_idx]
                    for key in ('loss', 'loss_e', 'loss_f'):
                        if key in hl:
                            head_loss_accum[head_idx][key].append(hl[key].cpu())

            if step % 50 == 0:
                torch.cuda.empty_cache()

        torch.cuda.synchronize()
        return self._aggregate_epoch_losses(epoch_loss_dict, head_loss_accum)

    # ------------------------------------------------------------------
    # Validation: per-head separate evaluation (JAX evaluate_per_head)
    # ------------------------------------------------------------------
    def validate_per_head(self, epoch=0):
        """Evaluate each head on its own validation set independently.

        Returns:
            epoch_loss_dict: combined + per-head metrics dict
            total_val_loss: weighted total validation loss (scalar)
        """
        self.model.eval()
        self.ckpt['valid_scale_shift'] = []
        loss_log_config = self.log_config['valid']

        epoch_loss_dict = {key: [] for key in loss_log_config if not key.startswith('head_')}
        head_loss_accum = {h: {'loss': [], 'loss_e': [], 'loss_f': []} for h in range(self.num_heads)}

        total_val_loss = 0.0
        total_weight = 0.0

        for head_idx, valid_loader in self.per_head_valid_loaders.items():
            head_weight = self.loss_weights[head_idx]
            head_name = self.heads[head_idx]
            head_total_loss = 0.0
            n_batches = 0

            for batch_idx, data in enumerate(valid_loader):
                if self.smoke_enabled and batch_idx >= self.smoke_max_batches:
                    break

                data = self.move_to_device(data, self.device)
                self._log_batch_size('valid', batch_idx, data, epoch)

                # no_grad() disabled: autograd.grad() needs a live graph for force computation
                preds = self.model(data, False)
                preds = self.scale_shift(preds, data, 'valid')
                loss_dict = self.compute_loss(preds, data)

                head_total_loss += loss_dict['loss'].detach().item()
                n_batches += 1

                # Aggregate non-head losses
                for l in loss_log_config:
                    if l.startswith('head_'):
                        continue
                    v = loss_dict.get(l)
                    if v is not None:
                        v = v.detach().cpu() if isinstance(v, torch.Tensor) else v
                        epoch_loss_dict[l].append(v * head_weight)

                # Per-head loss tracking
                if 'head_losses' in loss_dict and head_idx in loss_dict['head_losses']:
                    hl = loss_dict['head_losses'][head_idx]
                    for key in ('loss', 'loss_e', 'loss_f'):
                        if key in hl:
                            head_loss_accum[head_idx][key].append(hl[key].cpu())

            # Per-head summary
            if n_batches > 0:
                avg_head_loss = head_total_loss / n_batches
                total_val_loss += avg_head_loss * head_weight
                total_weight += head_weight
                if self.rank == 0:
                    print(f"  [Valid] Head {head_idx} ({head_name}): "
                          f"loss={avg_head_loss:.6f}, weight={head_weight}")

        # Normalize total validation loss
        if total_weight > 0:
            total_val_loss /= total_weight

        torch.cuda.synchronize()
        result = self._aggregate_epoch_losses(epoch_loss_dict, head_loss_accum, total_weight)
        result['loss'] = torch.tensor(total_val_loss)
        return result

    # ------------------------------------------------------------------
    # Mixed eval (for initial_test / backward compatibility)
    # ------------------------------------------------------------------
    def _eval_one_epoch_mixed(self, mode, data_loader, epoch):
        """Evaluate on mixed data loader (used by initial_test)."""
        self.model.eval()
        loss_log_config = self.log_config['valid']
        if data_loader is None:
            data_loader = self.valid_loader
        if mode == 'valid':
            self.ckpt['valid_scale_shift'] = []

        epoch_loss_dict = {key: [] for key in loss_log_config if not key.startswith('head_')}
        head_loss_accum = {h: {'loss': [], 'loss_e': [], 'loss_f': []} for h in range(self.num_heads)}

        for batch_idx, data in enumerate(data_loader):
            if self.smoke_enabled and batch_idx >= self.smoke_max_batches:
                break
            data = self.move_to_device(data, self.device)
            preds = self.model(data, False)
            preds = self.scale_shift(preds, data, mode)
            loss_dict = self.compute_loss(preds, data)

            for l in loss_log_config:
                if l.startswith('head_'):
                    continue
                val = loss_dict.get(l, torch.nan)
                epoch_loss_dict[l].append(val.detach().cpu() if isinstance(val, torch.Tensor) else val)
            if 'head_losses' in loss_dict:
                for head_idx, hl in loss_dict['head_losses'].items():
                    if head_idx < self.num_heads:
                        for key in ('loss', 'loss_e', 'loss_f'):
                            if key in hl:
                                head_loss_accum[head_idx][key].append(hl[key].cpu())

        torch.cuda.synchronize()
        return self._aggregate_epoch_losses(epoch_loss_dict, head_loss_accum)

    # ------------------------------------------------------------------
    # Helper: aggregate epoch losses
    # ------------------------------------------------------------------
    def _aggregate_epoch_losses(self, epoch_loss_dict, head_loss_accum, total_weight=None):
        """Average accumulated per-step losses into epoch-level metrics."""
        if total_weight and total_weight > 0:
            result = {
                key: torch.mean(torch.tensor(value)) / total_weight
                if value else torch.tensor(float('nan'))
                for key, value in epoch_loss_dict.items()
            }
        else:
            result = {
                key: torch.mean(torch.tensor(value)) if value else torch.tensor(float('nan'))
                for key, value in epoch_loss_dict.items()
            }

        for head_idx in range(self.num_heads):
            for key in ('loss', 'loss_e', 'loss_f'):
                vals = head_loss_accum[head_idx][key]
                if vals:
                    result[f'head_{head_idx}_{key}'] = torch.mean(torch.tensor(vals))

        return result