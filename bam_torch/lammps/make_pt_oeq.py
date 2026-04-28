"""
Convert model pkl to pt format (OEQ-aware version).

Auto-detects the equivariant backend used during training and rebuilds
the model accordingly. Also supports forcing a specific backend, which
is useful when an OEQ-trained model needs to be exported for LAMMPS
(LAMMPS' libtorch_tp_jit class is not available inside the LAMMPS binary,
so the saved model.pt must be e3nn-shaped).

The trained weights themselves (interactions.X.conv_tp_weights.layer*.weight,
linear_*.weight, etc.) are identical between OEQ and e3nn — only the buffer
keys (`conv_tp.weight: (0,)`, `output_mask`, `_compiled_main_left_right._w3j_*`)
differ. So an OEQ checkpoint can be loaded into an e3nn-built model with
strict=False, and the e3nn buffers are auto-initialized at build time.

Usage:
    # Auto-detect (default)
    python -m bam_torch.lammps.make_pt_oeq --pkl model.pkl --pt model.pt

    # Force e3nn — required when target is LAMMPS and training used OEQ
    python -m bam_torch.lammps.make_pt_oeq --pkl model.pkl --pt model.pt --backend e3nn

    # Force OEQ (requires openequivariance installed)
    python -m bam_torch.lammps.make_pt_oeq --pkl model.pkl --pt model.pt --backend oeq

Or in Python:
    from bam_torch.lammps.make_pt_oeq import recreate_model_pt_from_pkl
    recreate_model_pt_from_pkl('model.pkl', 'model.pt', backend='e3nn')
"""

import argparse
import inspect
import torch
from e3nn import o3
from copy import deepcopy

from bam_torch.model.models import RACE
from bam_torch.model.wrapper_ops import (
    CuEquivarianceConfig,
    OEQConfig,
    CUET_AVAILABLE,
    OEQ_AVAILABLE,
)


def _trained_with_e3nn(state_dict):
    """Detect e3nn TensorProduct in the *interactions* layer.

    e3nn TensorProduct registers `.conv_tp.weight` (often shape (0,)) and
    `_compiled_main_left_right._w3j_*` buffers; OEQ/cueq backends do not.

    Scope the check to `interactions.` because `products.X.conv_tp` is
    always built with e3nn even when interactions use OEQ, so its keys
    cannot be used to identify the training backend.
    """
    for k in state_dict.keys():
        if not k.startswith('interactions.'):
            continue
        if '.conv_tp.weight' in k or '._compiled_main_left_right._w3j_' in k:
            return True
    return False


def _resolve_backend(state_dict, cfg, requested):
    """Decide which backend to build the rebuilt model with.

    requested: 'auto' | 'e3nn' | 'oeq' | 'cueq'
    """
    if requested == 'e3nn':
        return 'e3nn', None, None
    if requested == 'oeq':
        if not OEQ_AVAILABLE:
            raise RuntimeError(
                "Backend 'oeq' was requested but openequivariance is not installed."
            )
        return 'oeq', None, OEQConfig(enabled=True, optimize_all=True)
    if requested == 'cueq':
        if not CUET_AVAILABLE:
            raise RuntimeError(
                "Backend 'cueq' was requested but CuEquivariance is not installed."
            )
        return 'cueq', CuEquivarianceConfig(
            enabled=True, layout="ir_mul",
            group="O3_e3nn", optimize_all=True,
        ), None

    # requested == 'auto' — auto-detect from checkpoint then env availability
    if _trained_with_e3nn(state_dict):
        return 'e3nn (auto: forced by checkpoint)', None, None

    cueq_request = cfg.get('cueq_config')
    oeq_request = cfg.get('oeq_config')

    if (cueq_request is None or cueq_request) and CUET_AVAILABLE:
        return 'cueq (auto)', CuEquivarianceConfig(
            enabled=True, layout="ir_mul",
            group="O3_e3nn", optimize_all=True,
        ), None
    if oeq_request and OEQ_AVAILABLE:
        return 'oeq (auto)', None, OEQConfig(enabled=True, optimize_all=True)
    return 'e3nn (auto: fallback)', None, None


def recreate_model_pt_from_pkl(
    pkl_path='model.pkl',
    output_path='model.pt',
    backend='auto',
):
    """Generate model.pt from model.pkl

    Args:
        pkl_path:    Path to the model pkl file
        output_path: Path to save the pt file
        backend:     'auto' (detect from checkpoint, default),
                     'e3nn' (force e3nn — required for LAMMPS),
                     'oeq'  (force OpenEquivariance — fast Python inference),
                     'cueq' (force CuEquivariance)

    Returns:
        model: The loaded RACE model
    """
    # Load checkpoint
    pckl = torch.load(pkl_path, map_location='cpu', weights_only=False)
    cfg = pckl['input.json']

    # Strip DDP prefix once so backend detection sees the real keys.
    state_dict = pckl['params']
    if any(k.startswith('module.') for k in state_dict.keys()):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

    backend_used, cueq_config, oeq_config = _resolve_backend(
        state_dict, cfg, backend
    )

    # Parameter to choose force computation mode:
    regress_forces = cfg.get('regress_forces', 'auto')
    if regress_forces == True:    # forces computed via auto-gradient of energy
        regress_forces = "autograd"
    elif regress_forces == False:  # no force computation
        regress_forces = "false"

    model_kwargs = dict(
        cutoff=cfg['cutoff'],
        avg_num_neighbors=cfg['avg_num_neighbors'],
        num_species=cfg['num_species'],
        max_ell=cfg['max_ell'],
        num_basis_func=cfg['num_radial_basis'],
        hidden_irreps=o3.Irreps(cfg['hidden_channels']),
        nlayers=cfg['nlayers'],
        features_dim=cfg['features_dim'],
        output_irreps=o3.Irreps(cfg.get('output_channels', '1x0e')),
        active_fn=cfg.get('active_fn', 'identity'),
        regress_forces=regress_forces,
        cueq_config=cueq_config,
    )
    model_params = inspect.signature(RACE).parameters
    if 'oeq_config' in model_params:
        model_kwargs['oeq_config'] = oeq_config
    if 'l_separated_layer_norm' in model_params:
        model_kwargs['l_separated_layer_norm'] = cfg.get(
            'l_separated_layer_norm', False
        )
    if 'interaction_block' in model_params:
        model_kwargs['interaction_block'] = cfg.get('interaction_block', 'slow')

    model = RACE(**model_kwargs)

    # strict=False so e3nn auto-buffers (Wigner 3j cache, output_mask, ...) that
    # aren't in an OEQ checkpoint can be auto-initialized at build time.
    # All trained weights (the layer*.weight family) ARE shared between
    # backends and must match — guard against silently dropping any.
    result = model.load_state_dict(state_dict, strict=False)
    if result.unexpected_keys:
        raise RuntimeError(
            "Unexpected keys in checkpoint that the rebuilt model cannot "
            f"absorb: {result.unexpected_keys}"
        )
    auto_init_count = len(result.missing_keys)

    model.eval()
    torch.save(deepcopy(model), output_path)

    print(f"Successfully recreated {output_path} from {pkl_path}")
    print(f"  Model type:        RACE")
    print(f"  Backend:           {backend_used}")
    print(f"  Cutoff:            {cfg['cutoff']} A")
    print(f"  Total params:      {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Auto-init buffers: {auto_init_count} "
          f"(internal e3nn caches / Wigner 3j; not learned weights)")

    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert model pkl to pt format')
    parser.add_argument('--pkl', type=str, default='model.pkl',
                        help='Path to the model pkl file (default: model.pkl)')
    parser.add_argument('--pt', type=str, default='model.pt',
                        help='Path to save the pt file (default: model.pt)')
    parser.add_argument('--backend', type=str, default='auto',
                        choices=['auto', 'e3nn', 'oeq', 'cueq'],
                        help=(
                            "Equivariant backend to rebuild with: "
                            "'auto' (detect from checkpoint), "
                            "'e3nn' (force — required when target is LAMMPS), "
                            "'oeq' (force OpenEquivariance), "
                            "'cueq' (force CuEquivariance). Default: auto."
                        ))
    args = parser.parse_args()

    recreate_model_pt_from_pkl(args.pkl, args.pt, backend=args.backend)
