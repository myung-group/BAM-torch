"""
Convert a trained checkpoint (model.pkl) directly to a LAMMPS-compatible
TorchScript model (model-lammps.pt) in a single in-memory step.

This merges make_pt_oeq.py (--backend e3nn) and create_lammps.py, skipping
the intermediate model.pt file: with e3nn >= 0.6, o3.SphericalHarmonics
holds a non-picklable ScriptFunction, so torch.save()-ing the eager model
(what make_pt*.py does) fails. Building the model and compiling it to
TorchScript in one process avoids that pickle step entirely; the final
model-lammps.pt is identical.

The model is always rebuilt e3nn-shaped: LAMMPS' libtorch cannot run
OpenEquivariance/CuEquivariance kernels. Checkpoints trained with OEQ load
with strict=False - the trained weights are shared between backends, and
the e3nn-only buffers (Wigner 3j cache, output_mask, ...) are
auto-initialized at build time.

Usage:
    python -m bam_torch.lammps.convert_to_lammps
    python -m bam_torch.lammps.convert_to_lammps --pkl model.pkl --output model-lammps.pt

Or in Python:
    from bam_torch.lammps.convert_to_lammps import convert_pkl_to_lammps
    convert_pkl_to_lammps('model.pkl', 'model-lammps.pt')
"""

import os
os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
os.environ.setdefault("MASTER_ADDR", "localhost")
os.environ.setdefault("MASTER_PORT", "29500")
os.environ.setdefault("RANK", "0")
os.environ.setdefault("WORLD_SIZE", "1")

import argparse
import inspect

import torch
import torch.distributed as dist
from e3nn import o3
from e3nn.util import jit

from bam_torch.model import models as bam_models
from bam_torch.model.models import RACE, get_edge_relative_vectors_with_pbc_lammps
from bam_torch.lammps.lammps_bam import LAMMPS_BAM


def convert_pkl_to_lammps(pkl_path='model.pkl', output_path='model-lammps.pt'):
    """Rebuild the RACE model from a checkpoint and save it for LAMMPS.

    Args:
        pkl_path:    Path to the trained checkpoint (model.pkl)
        output_path: Path for the TorchScript model consumed by pair_style bam

    Returns:
        Path to the saved LAMMPS model
    """
    if not dist.is_initialized():
        dist.init_process_group(
            backend="nccl" if torch.cuda.is_available() else "gloo",
            init_method="env://", world_size=1, rank=0,
        )

    # ---- 1. Rebuild the RACE model from the checkpoint (e3nn backend) ----
    pckl = torch.load(pkl_path, map_location="cpu", weights_only=False)
    cfg = pckl["input.json"]

    state_dict = pckl["params"]
    if any(k.startswith("module.") for k in state_dict):
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

    regress_forces = cfg.get("regress_forces", "auto")
    if regress_forces is True:      # forces via auto-gradient of energy
        regress_forces = "auto"
    elif regress_forces is False:   # no force computation
        regress_forces = "false"

    model_kwargs = dict(
        cutoff=cfg["cutoff"],
        avg_num_neighbors=cfg["avg_num_neighbors"],
        num_species=cfg["num_species"],
        max_ell=cfg["max_ell"],
        num_basis_func=cfg["num_radial_basis"],
        hidden_irreps=o3.Irreps(cfg["hidden_channels"]),
        nlayers=cfg["nlayers"],
        features_dim=cfg["features_dim"],
        output_irreps=o3.Irreps(cfg.get("output_channels", "1x0e")),
        active_fn=cfg.get("active_fn", "identity"),
        regress_forces=regress_forces,
        cueq_config=None,           # LAMMPS export must be e3nn-shaped
    )
    model_params = inspect.signature(RACE).parameters
    if "oeq_config" in model_params:
        model_kwargs["oeq_config"] = None
    if "l_separated_layer_norm" in model_params:
        model_kwargs["l_separated_layer_norm"] = cfg.get("l_separated_layer_norm", False)
    if "interaction_block" in model_params:
        # "slow" and "fast" build different tensor-product paths, so the
        # block the checkpoint was trained with has to be reproduced here.
        model_kwargs["interaction_block"] = cfg.get("interaction_block", "slow")

    model = RACE(**model_kwargs)

    # strict=False: e3nn auto-buffers missing from an OEQ/cueq-trained
    # checkpoint are auto-initialized; guard against silently dropping
    # any trained weight.
    result = model.load_state_dict(state_dict, strict=False)
    if result.unexpected_keys:
        raise RuntimeError(
            "Unexpected keys in checkpoint that the rebuilt model cannot "
            f"absorb: {result.unexpected_keys}"
        )
    model.eval()

    # ---- 2. Attach LAMMPS metadata (same as create_lammps.py) ----
    uniq_element = pckl["uniq_element"]
    enr_avg_per_element = pckl["enr_avg_per_element"]

    # Newer checkpoints save valid_scale_shift as {class_idx: tensor};
    # older checkpoints saved a tensor/list. Handle both.
    vss = pckl["valid_scale_shift"]
    if isinstance(vss, dict):
        e_corr = torch.stack([v for v in vss.values()]).flatten().mean().item()
    else:
        e_corr = torch.tensor(vss).flatten().mean().item()

    print(f"enr_avg_per_element: {enr_avg_per_element}")
    print(f"e_corr: {e_corr}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model.atomic_numbers = torch.tensor(list(uniq_element.keys())).clone().detach()
    model.num_interactions = torch.tensor(cfg["nlayers"])
    model.r_max = torch.tensor(cfg["cutoff"])
    model = model.float().to(device)

    bam_models.get_edge_relative_vectors_with_pbc = get_edge_relative_vectors_with_pbc_lammps
    model.training_mode_for_lammps = True

    # Adjust variables for compatibility across model versions
    try:
        criterion = model.criterion
        model.regress_forces = "auto" if criterion < model.criterion_value else "direct"
    except AttributeError:
        model.criterion = None

    for module in model.modules():
        module.training_mode_for_lammps = True

    # ---- 3. Compile to TorchScript and save ----
    lammps_model = LAMMPS_BAM(
        model, enr_avg_per_element=enr_avg_per_element, e_corr=e_corr
    ).to(device)
    jit.compile(lammps_model).save(output_path)
    print(f"Successfully created LAMMPS model: {output_path}")

    if dist.is_initialized():
        dist.destroy_process_group()

    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert model.pkl directly to a LAMMPS-compatible "
                    "TorchScript model (no intermediate model.pt)."
    )
    parser.add_argument("--pkl", type=str, default="model.pkl",
                        help="Path to the model pkl file (default: model.pkl)")
    parser.add_argument("--output", type=str, default="model-lammps.pt",
                        help="Path to save the LAMMPS model (default: model-lammps.pt)")
    args = parser.parse_args()

    convert_pkl_to_lammps(args.pkl, args.output)
