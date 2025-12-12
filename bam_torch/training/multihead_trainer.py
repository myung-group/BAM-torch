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
        # Multihead config 먼저 처리 (super().__init__ 전에 필요)
        self.multihead_config = json_data.get("multihead", {})
        if not self.multihead_config.get("enabled", False):
            raise ValueError("Multihead finetuning is not enabled in config")

        # datasets_config 초기화
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

        # Foundation model 설정 추출 
        self.foundation_config = self._extract_foundation_config(json_data, rank)

        # Batch logs 디렉토리 설정
        self._setup_batch_logs(json_data, rank)

        # Parent __init__ 호출 - setup()이 자동으로 호출됨
        super().__init__(json_data, rank, world_size)
    
    def _extract_foundation_config(self, json_data: Dict[str, Any], rank: int) -> Dict[str, Any]:
        """Foundation model의 input.json을 그대로 반환"""
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
        
        # Foundation model 로드 (restart가 아닌 경우)
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
        """Batch logs 디렉토리 및 로그 파일 설정"""
        # Node ID와 Local GPU rank 추출
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

        # Rank별 배치 사이즈 로그 디렉터리/파일
        batch_log_root = Path(json_data.get('batch_size_log_root', 'batch_logs'))
        self.batch_log_dir = batch_log_root / f"rank_{rank}"
        self.batch_log_dir.mkdir(parents=True, exist_ok=True)

        # GPU별 로그 파일 (batch_log_dir 내부에 생성)
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
        """배치 사이즈 정보를 로그 파일에 기록"""
        if not hasattr(self, 'batch_size_log') or self.batch_size_log.closed:
            return
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # 배치 내 그래프 수
        num_graphs = data.ptr.numel() - 1 if hasattr(data, 'ptr') else 1
        
        # 총 노드/엣지 수
        total_nodes = data.num_nodes if hasattr(data, 'num_nodes') else len(data.positions)
        total_edges = data.edge_index.shape[1] if hasattr(data, 'edge_index') else 0
        
        # Head 구성
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

    # configure_exponential_moving_average: BaseTrainer 그대로 사용

    def set_model(self):
        """
        Override: RACEMultihead 모델 생성
        Foundation config를 우선 사용하여 shape 호환성 보장
        """
        model_config = self.json_data
        
        # Foundation config가 있으면 우선 사용
        if self.foundation_config:
            fc = self.foundation_config
            if self.rank == 0:
                print(f"\n\033[36mUsing Foundation model config:\033[0m")
                print(f"  - hidden_channels: {fc.get('hidden_channels', 'N/A')}")
                print(f"  - features_dim: {fc.get('features_dim', 'N/A')}")
                print(f"  - nlayers: {fc.get('nlayers', 'N/A')}")
                print(f"  - cutoff: {fc.get('cutoff', 'N/A')}")
            
            # Foundation config에서 가져오기
            hidden_irreps_str = fc.get('hidden_channels', model_config['hidden_channels'])
            hidden_irreps = o3.Irreps(hidden_irreps_str)
            features_dim = fc.get('features_dim', model_config['features_dim'])
            nlayers = fc.get('nlayers', model_config['nlayers'])
            cutoff = fc.get('cutoff', model_config['cutoff'])
            num_basis_func = fc.get('num_radial_basis', model_config['num_radial_basis'])
            max_ell = fc.get('max_ell', model_config['max_ell'])
        else:
            # Foundation config가 없으면 input.json 사용
            hidden_irreps = o3.Irreps(model_config['hidden_channels'])
            features_dim = model_config['features_dim']
            nlayers = model_config['nlayers']
            cutoff = model_config['cutoff']
            num_basis_func = model_config['num_radial_basis']
            max_ell = model_config['max_ell']
        
        # 나머지는 input.json에서 (데이터셋 관련)
        avg_num_neighbors = model_config['avg_num_neighbors']
        num_species = model_config['num_species']
        
        output_irreps = model_config.get('output_channels', "1x0e")
        active_fn = model_config.get('active_fn', "identity")
        
        regress_forces = model_config.get('regress_forces')
        if regress_forces == True:
            regress_forces = "autograd"
        elif regress_forces == False:
            regress_forces = "false"
        
        # MLP_irreps: features_dim에서 직접 변환
        mlp_irreps = o3.Irreps(f"{features_dim}x0e")
        
        # # RACEMultihead 모델 생성 (기존 코드 - 주석 처리)
        # from bam_torch.model.models import RACEMultihead
        # 
        # model = RACEMultihead(
        #     cutoff=cutoff,
        #     avg_num_neighbors=avg_num_neighbors,
        #     num_species=num_species,
        #     max_ell=max_ell,
        #     num_basis_func=num_basis_func,
        #     hidden_irreps=hidden_irreps,
        #     nlayers=nlayers,
        #     features_dim=features_dim,
        #     output_irreps=output_irreps,
        #     active_fn=active_fn,
        #     radial_MLP=[64, 64],
        #     MLP_irreps=mlp_irreps,
        #     regress_forces=regress_forces,
        #     compute_stress=True,
        #     heads=self.heads,
        #     cueq_config=None,
        # )
        
        # RACEUnified 모델 생성 (Single-head/Multihead 통합 버전)
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
        )
        
        self.msg += f'\n\033[33m -- Multihead model with {self.num_heads} heads: {self.heads}\033[0m\n'
        
        return model
    
    def _load_foundation_enr_avg(self):
        """
        Foundation model의 enr_avg_per_element 로드
        
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
        """Foundation model weights 로드 및 readout 확장"""
        foundation_path = self.json_data['NN']['foundation_model']
        
        if self.rank == 0:
            print(f"\n{'='*80}")
            print(f"Loading Foundation model: {foundation_path}")
            print(f"{'='*80}\n")
        
        # BAM-torch pkl 형식: {'params': state_dict, ...}
        foundation_ckpt = torch.load(foundation_path, map_location='cpu', weights_only=False)
        foundation_state = foundation_ckpt['params']
        
        # 현재 모델 참조
        model = self.model.module if self.ddp else self.model
        
        self._load_from_state_dict(model, foundation_state, self.num_heads)
    
    
    def _load_from_state_dict(self, model, foundation_state, num_heads):
        """
        Fallback: state_dict에서 파라미터 로드 
        
        모델 객체가 아닌 state_dict만 있는 경우 사용
        Reference: https://github.com/ACEsuit/mace/blob/main/mace/tools/finetuning_utils.py
        
        Foundation model readout 파라미터 구조:
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
        # MLP_dim과 hidden_dim_0e를 Foundation NonLinearReadoutBlock에서 추출
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
            
            # Readout 파라미터는 확장 필요 
            if 'readouts' in clean_name:
                if param.numel() == 0:
                    # 빈 파라미터 (bias 없음)
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
                    # 스케일링 적용: / sqrt(MLP_dim / target_output_dim)
                    if param.numel() == mlp_dim:
                        # view(1, MLP_dim).repeat(num_heads, num_heads) -> [num_heads, MLP_dim * num_heads]
                        expanded = param.view(1, mlp_dim).repeat(num_heads, num_heads).flatten()
                        # 스케일링
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
                        # 스케일링
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
                # 일반 파라미터: shape이 같으면 직접 복사
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
        Override: 멀티헤드용 데이터로더 구성 (BaseTrainer 스타일)
        """
        # Foundation model의 enr_avg_per_element 로드 (replay용)
        foundation_enr_avg, foundation_uniq_element = self._load_foundation_enr_avg()
        
        # Smoke test config
        smoke_config = self.json_data.get('smoke_test', {})
        
        train_loader, valid_loader, uniq_element, enr_avg_per_element, per_head_enr_avg = \
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
        
        # Store per-head E0s for checkpoint
        self.per_head_enr_avg = per_head_enr_avg
        
        return train_loader, valid_loader, uniq_element, enr_avg_per_element
    
    def load_loss(self, reduction='mean'):
        """
        Override: Huber loss 지원 추가
        로컬 BaseTrainer는 load_loss를 호출하므로 이 이름 사용
        (GitHub 최신 버전은 configure_loss 사용)
        """
        nn_config = self.json_data.get("NN", {})
        loss_config = nn_config.get("loss_config", {})
        
        if not loss_config:
            if self.json_data.get("regress_forces"):
                loss_config = {'energy_loss': 'huber', 'force_loss': 'huber', 'stress_loss': 'huber'}
            else:
                loss_config = {'energy_loss': 'huber'}
        
        # Stress loss 기본값
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
    
    # Alias for GitHub latest compatibility
    configure_loss = load_loss
    
    def compute_loss(self, preds, data):
        """
        Override: Head별 weighted loss 계산
        """
        lambda_config = self.json_data["NN"]
        e_lambda = lambda_config.get('enr_lambda', 1.0)
        f_lambda = lambda_config.get('frc_lambda', 1.0)
        s_lambda = lambda_config.get('str_lambda', 1.0)
        lambd = lambda_config.get('l2_lambda', 0)
        
        loss = {"loss": []}
        
        # Config별 head와 weight 가져오기
        if 'config_head' in data:
            config_heads = data['config_head'].flatten()
        elif 'head' in data:
            # head가 graph-level이면 batch로 확장
            if hasattr(data, 'batch'):
                # Per-graph head를 per-config로 변환
                ptr = data.get('ptr')
                if ptr is not None and ptr.numel() > 1:
                    batch_size = ptr.numel() - 1
                    config_heads = torch.zeros(batch_size, dtype=torch.long, device=data['head'].device)
                    # 각 config의 첫 번째 atom의 head 사용
                    for i in range(batch_size):
                        start_idx = ptr[i].item()
                        config_heads[i] = data['head'][data['batch'] == i][0] if (data['batch'] == i).any() else 0
                else:
                    config_heads = data['head'].flatten()
            else:
                config_heads = data['head'].flatten()
        else:
            config_heads = torch.zeros(preds['energy'].shape[0], dtype=torch.long, device=preds['energy'].device)
        
        # Weight 가져오기
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
        
        # Stress loss (preds["stress"]가 None일 수 있음)
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
                
                # Head별 force loss 계산
                head_force_loss = None
                if "forces" in preds and self.loss_fn.get('force_loss') is not None:
                    if 'batch' in data:
                        batch_indices = data['batch']
                        # 해당 head에 속하는 atom들 찾기
                        atom_head_mask = head_mask[batch_indices]
                        if atom_head_mask.any():
                            force_diff = preds["forces"][atom_head_mask] - data["forces"][atom_head_mask]
                            head_force_loss = (force_diff ** 2).sum(dim=-1).mean()
                            head_loss_dict["loss_f"] = head_force_loss.detach()
                
                # Head별 총 loss (energy + force)
                head_total_loss = e_lambda * head_energy_loss
                if head_force_loss is not None:
                    head_total_loss = head_total_loss + f_lambda * head_force_loss
                head_loss_dict["loss"] = head_total_loss.detach()
                
                loss["head_losses"][head_idx] = head_loss_dict
        
        return loss
    
    def scale_shift(self, preds, data, mode):
        """Override: Head별 scale_shift 적용 (GitHub BaseTrainer compatible)."""
        energy_target = data["energy"].flatten()
        energy_predict = preds["energy"].flatten()
        
        # Config별 head 가져오기
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
            
            # Head별 shift 계산
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
        
        Follows GitHub latest BaseTrainer pattern with:
        - initial_test() for loss_test_min initialization
        - EMA context manager for validation
        - update_check_point() and print_logger() methods
        """
        # Initial test
        self.initial_test()
        
        # Print logger head
        if self.rank == 0:
            self.logger.print_logger_head()
        
        # Main training loop
        nepoch = self.json_data['NN']['nepoch']
        
        for epoch in range(nepoch):
            # Criterion 업데이트
            try:
                base_model = self.model.module if self.ddp else self.model
                base_model.update_criterion_value(epoch + self.start_epoch + 1)
            except:
                pass
            
            # Train
            epoch_loss_train = self.train_one_epoch(mode='train', epoch=epoch+self.start_epoch)
            
            if self.ddp:
                torch.distributed.barrier()
            
            # Validate and record with EMA context
            param_context = (
                self.ema.average_parameters() if self.ema is not None else nullcontext()
            )
            with param_context:
                if (epoch+1) % self.log_interval == 0:
                    epoch_loss_valid = self.train_one_epoch(mode='valid', epoch=epoch+self.start_epoch)
                    
                    if self.ddp:
                        torch.distributed.barrier()
                    
                    if self.rank == 0:
                        # Update check point 
                        if epoch_loss_valid['loss'] < self.loss_test_min:
                            self.update_check_point(epoch, epoch_loss_train, epoch_loss_valid)
                            self.loss_test_min = epoch_loss_valid['loss']
                            self.l_ckpt_saved = False
                        
                        # Print epoch loss
                        self.print_logger(epoch, epoch_loss_train, epoch_loss_valid)
                        
                        # Free GPU memory
                        torch.cuda.empty_cache()
                        gc.collect()
                    
                    # Update scheduler (learning rate)
                    metrics = None
                    if self.json_data["scheduler"]["scheduler"] == "ReduceLROnPlateau":
                        metrics = epoch_loss_valid['loss']
                    self.scheduler.step(metrics, epoch)
                
                # Save check point
                if (epoch+1) % self.json_data['NN']['nsave'] == 0 and not self.l_ckpt_saved:
                    torch.save(self.ckpt, self.json_data['NN']['fname_pkl'])
                    # Note: model.pt 저장 제거 (ScriptFunction pickle 에러 방지)
                    # 필요시 model.state_dict()만 저장
                    self.l_ckpt_saved = True
    
    # initial_test: BaseTrainer 그대로 사용 (단, train_one_epoch에 epoch 인자 전달 필요)
    def initial_test(self):
        """Run a preliminary test epoch and record the initial reference loss."""
        epoch_loss_test = self.train_one_epoch(mode='test', epoch=0)
        if self.ddp:
            torch.distributed.barrier()
        self.loss_test_min = epoch_loss_test['loss']
    
    # update_check_point: EMA state 저장 추가
    def update_check_point(self, epoch, epoch_loss_train, epoch_loss_valid):
        """Update checkpoint with current training state (+ EMA state + per-head E0s)."""
        # BaseTrainer 로직 호출
        super().update_check_point(epoch, epoch_loss_train, epoch_loss_valid)
        # EMA state 추가 저장
        if self.ema is not None:
            self.ckpt['ema_state'] = self.ema.state_dict()
        # Per-head E0s 저장 (evaluation에서 사용)
        if hasattr(self, 'per_head_enr_avg'):
            self.ckpt['per_head_enr_avg'] = self.per_head_enr_avg
    
    # print_logger: BaseTrainer 그대로 사용
    
    def train_one_epoch(self, mode='train', data_loader=None, epoch=0):
        """Train/validate one epoch - Multihead version with per-head loss tracking."""
        if mode == 'train':
            self.model.train()
            backprop = True
            loss_log_config = self.log_config['train']
            if data_loader is None:
                data_loader = self.train_loader
            self.ckpt['train_scale_shift'] = []
        else:  # test or valid
            self.model.eval()
            backprop = False
            loss_log_config = self.log_config['valid']
            if data_loader is None:
                data_loader = self.valid_loader
            if mode == 'valid':
                self.ckpt['valid_scale_shift'] = []
        
        # head_X_* 키는 별도로 계산되므로 초기화에서 제외
        epoch_loss_dict = {key: [] for key in loss_log_config if not key.startswith('head_')}
        
        # Head별 loss tracking
        head_loss_accum = {h: {'loss': [], 'loss_e': [], 'loss_f': []} for h in range(self.num_heads)}
        
        # tqdm progress bar for training
        if self.rank == 0 and tqdm is not None:
            data_iter = tqdm(data_loader, desc=f"Epoch {epoch} [{mode}]", leave=True)
        else:
            data_iter = data_loader
        
        for batch_idx, data in enumerate(data_iter):
            # Smoke test: limit batches
            if self.smoke_enabled and batch_idx >= self.smoke_max_batches:
                if self.rank == 0 and batch_idx == self.smoke_max_batches:
                    print(f"  [SMOKE] Stopping after {self.smoke_max_batches} batches")
                break
            
            data = self.move_to_device(data, self.device)
            
            # 배치 사이즈 로깅
            self._log_batch_size(mode, batch_idx, data, epoch)
            
            # Predict
            preds = self.model(data, backprop)
            preds = self.scale_shift(preds, data, mode)
            
            # Compute loss
            loss_dict = self.compute_loss(preds, data)
            
            # tqdm 진행 표시 업데이트
            if self.rank == 0 and tqdm is not None and hasattr(data_iter, 'set_postfix'):
                data_iter.set_postfix({
                    'loss': f"{loss_dict['loss'].item():.4f}",
                    'loss_e': f"{loss_dict.get('loss_e', 0):.4f}" if 'loss_e' in loss_dict else 'N/A'
                })
            
            loss = loss_dict['loss']
            if backprop:
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_value_(self.model.parameters(), clip_value=0.5)
                self.optimizer.step()
                
                # EMA update during training (GitHub BaseTrainer pattern)
                if self.ema is not None:
                    self.ema.update()
            
            # Log losses (head_X_* 키는 epoch 평균에서 계산되므로 제외)
            for l in loss_log_config:
                if l.startswith('head_'):
                    continue  # head별 loss는 나중에 계산
                val = loss_dict.get(l, torch.nan)
                epoch_loss_dict[l].append(val.detach().cpu() if isinstance(val, torch.Tensor) else val)
            
            # Head별 loss 누적
            if 'head_losses' in loss_dict:
                for head_idx, head_loss in loss_dict['head_losses'].items():
                    if head_idx < self.num_heads:
                        if 'loss' in head_loss:
                            head_loss_accum[head_idx]['loss'].append(head_loss['loss'].cpu())
                        if 'loss_e' in head_loss:
                            head_loss_accum[head_idx]['loss_e'].append(head_loss['loss_e'].cpu())
                        if 'loss_f' in head_loss:
                            head_loss_accum[head_idx]['loss_f'].append(head_loss['loss_f'].cpu())
            
            # Memory cleanup every N batches
            if batch_idx % 50 == 0:
                torch.cuda.empty_cache()
        
        torch.cuda.synchronize()
        
        # 평균 계산
        epoch_loss_dict = {
            key: torch.mean(torch.tensor(value)) 
            for key, value in epoch_loss_dict.items()
        }
        
        # Head별 평균 loss 추가
        for head_idx in range(self.num_heads):
            if head_loss_accum[head_idx]['loss']:
                epoch_loss_dict[f'head_{head_idx}_loss'] = torch.mean(
                    torch.tensor(head_loss_accum[head_idx]['loss'])
                )
            if head_loss_accum[head_idx]['loss_e']:
                epoch_loss_dict[f'head_{head_idx}_loss_e'] = torch.mean(
                    torch.tensor(head_loss_accum[head_idx]['loss_e'])
                )
            if head_loss_accum[head_idx]['loss_f']:
                epoch_loss_dict[f'head_{head_idx}_loss_f'] = torch.mean(
                    torch.tensor(head_loss_accum[head_idx]['loss_f'])
                )
        
        return epoch_loss_dict