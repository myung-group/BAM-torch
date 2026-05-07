"""
SimpleGNNTrainer
----------------
Trainer specialized for `SimpleScalableGNN` on large-scale universal-potential
datasets such as MPtraj. Training uses **probabilistic symmetrization** with
rotation only (`ga_method="prob_rot"`); permutation is unnecessary because
the GNN backbone is already permutation-equivariant.

Inherits from `GAMPTrainer` to reuse:
- pickle-based dataset loading and per-rank caching of preprocessed shards
- multi-node / DDP setup and distributed loss reduction
- prob-symmetrization forward pass (`pa_model_forward`)

Hyperparameters in `input.json` follow the same keyword convention as the
existing `gnn` / `faenet` model entry of `GAMPTrainer.set_model`.
"""
from bam_torch.group_averaging.training.ga_mp_trainer import GAMPTrainer
from bam_torch.group_averaging.model import MODEL_REGISTRY
from bam_torch.group_averaging.model.equiv_layer import EquivariantInterface


_SIMPLE_GNN_NAMES = {"sgnn", "simple_gnn", "scalable_gnn"}
_SAT_NAMES = {"sat", "sparse_attn_transformer", "sat_3d"}


class SimpleGNNTrainer(GAMPTrainer):
    """MPtraj-scale trainer for `SimpleScalableGNN` with prob_rot averaging.

    Defaults `ga_method` to `prob_rot` and `group_averaging` to `3D` if those
    keys are missing from the json config. All other behavior (pkl loader,
    EMA, scheduler, distributed loss) is inherited unchanged from
    `GAMPTrainer`.
    """

    def __init__(self, json_data, rank=0, world_size=1):
        if json_data.get("ga_method") is None:
            json_data["ga_method"] = "prob_rot"
        if json_data.get("group_averaging") is None:
            json_data["group_averaging"] = "3D"
        super().__init__(json_data, rank, world_size)

    def set_model(self):
        cfg = self.json_data
        model_name = cfg["model"].lower()

        if model_name not in _SIMPLE_GNN_NAMES and model_name not in _SAT_NAMES:
            return super().set_model()

        model_cls = MODEL_REGISTRY[model_name]

        regress_forces = cfg.get("regress_forces", "auto")
        if regress_forces is True:
            regress_forces = "autograd"
        elif regress_forces is False:
            regress_forces = "false"

        common_kwargs = dict(
            cutoff=cfg.get("cutoff", 6.0),
            num_species=cfg.get("num_species", 89),
            hidden_channels=cfg.get("hidden_channels", 256),
            features_dim=cfg.get("features_dim", 256),
            num_radial_basis=cfg.get("num_radial_basis", 8),
            nlayers=cfg.get("nlayers", 6),
            max_num_neighbors=cfg.get("max_neigh", 30),
            avg_num_neighbors=cfg.get("avg_num_neighbors", 30.0),
            regress_forces=regress_forces,
            compute_stress=cfg.get("compute_stress", True),
            compute_virials=cfg.get("compute_virials", True),
            radial_type=cfg.get("radial_type", "bessel"),
            num_polynomial_cutoff=cfg.get("num_polynomial_cutoff", 6),
            pbc=cfg.get("pbc", True),
        )

        if model_name in _SAT_NAMES:
            # Transformer-specific defaults: features_dim is FFN hidden
            # (typically 4 * d_model).
            common_kwargs["features_dim"] = cfg.get("features_dim", 1024)
            common_kwargs["nhead"] = cfg.get("nhead", 8)
            common_kwargs["edge_bias_hidden"] = cfg.get("edge_bias_hidden", 64)
            common_kwargs["dropout"] = cfg.get("dropout", 0.0)

        model = model_cls(**common_kwargs)

        # Equivariant interface that samples rotations for prob / prob_rot.
        ga_method = cfg.get("ga_method", "prob_rot").lower()
        if ga_method in {"prob", "probabilistic", "prob_rot"}:
            small_cfg = cfg.get("small_equiv", {})
            self.equiv_model = EquivariantInterface(
                symmetry=small_cfg.get("symmetry", "O3"),
                interface=small_cfg.get("interface", "prob"),
                fixed_noise=small_cfg.get("fixed_noise", False),
                noise_scale=small_cfg.get("noise_scale", 1),
                tau=small_cfg.get("tau", 0.01),
                hard=small_cfg.get("hard", True),
                cutoff=cfg.get("cutoff", 6.0),
                num_species=cfg.get("num_species", 89),
                avg_num_neighbors=cfg.get("avg_num_neighbors", 30.0),
                hidden_irreps=small_cfg.get(
                    "hidden_channels", "16x0e+8x1o+4x2e"
                ),
                features_dim=small_cfg.get("features_dim", 32),
                num_basis_func=small_cfg.get("num_radial_basis", 8),
                nlayers=small_cfg.get("nlayers", 1),
                max_ell=small_cfg.get("max_ell", 3),
                MLP_irreps=small_cfg.get("MLP_irreps", "16x0e"),
                output_irreps=small_cfg.get("output_channels", "3x1o"),
                gate=small_cfg.get("active_fn", "silu").lower(),
                cueq_config=small_cfg.get("cueq_config", None),
                radial_MLP=small_cfg.get("radial_MLP", [64, 64]),
            ).to(self.device)
            n_params = sum(
                p.numel() for p in self.equiv_model.parameters()
                if p.requires_grad
            )
            print(
                f"\n[SimpleGNNTrainer] equiv_model (prob_rot sampler) "
                f"parameters: {n_params}"
            )
        else:
            self.equiv_model = None

        return model
