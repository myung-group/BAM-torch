"""
Convert model.pt to LAMMPS-compatible TorchScript format.

Usage:
    python -m bam_torch.lammps.create_lammps --pkl model.pkl --pt model.pt --output model-lammps.pt

Or in Python:
    from bam_torch.lammps.create_lammps import create_lammps_model
    create_lammps_model('model.pkl', 'model.pt', 'model-lammps.pt')
"""

import os
os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "29500"
os.environ["RANK"] = "0"
os.environ["WORLD_SIZE"] = "1"

import ast
import argparse
import torch
import torch.distributed as dist
from e3nn.util import jit

from bam_torch.model import models as bam_models
from bam_torch.model.models import get_edge_relative_vectors_with_pbc_lammps
from bam_torch.lammps.lammps_bam import LAMMPS_BAM


def create_lammps_model(pkl_path='model.pkl', pt_path='model.pt', output_path=None,
                        enr_avg_file=None):
    """Convert model.pt to LAMMPS-compatible TorchScript format.

    Args:
        pkl_path: Path to the model pkl file
        pt_path: Path to the model pt file
        output_path: Path to save the LAMMPS model (default: {pt_path}-lammps.pt)
        enr_avg_file: Optional file containing enr_avg_per_element for universal potential

    Returns:
        Path to the saved LAMMPS model
    """
    if output_path is None:
        output_path = pt_path.replace('.pt', '-lammps.pt')

    # Initiate distributed process
    if not dist.is_initialized():
        dist.init_process_group(
            backend='nccl' if torch.cuda.is_available() else 'gloo',
            init_method='env://',
            world_size=1,
            rank=0
        )

    # Load pkl for configuration
    pckl = torch.load(pkl_path, map_location='cpu', weights_only=False)
    nlayers = pckl['input.json']['nlayers']
    cutoff = pckl['input.json']['cutoff']

    # Get energy averages
    if enr_avg_file is not None:
        with open(enr_avg_file, 'r', encoding='utf-8') as file:
            content = file.read()
        enr_avg_per_element, uniq_element = ast.literal_eval(content)
    else:
        uniq_element = pckl['uniq_element']
        enr_avg_per_element = pckl['enr_avg_per_element']

    # Get energy correction
    # Newer checkpoints save valid_scale_shift as {class_idx: tensor} dict;
    # older checkpoints saved it as a tensor/list. Handle both.
    vss = pckl['valid_scale_shift']
    if isinstance(vss, dict):
        e_corr = torch.stack([v for v in vss.values()]).flatten().mean().item()
    else:
        e_corr = torch.tensor(vss).flatten().mean().item()

    print(f"enr_avg_per_element: {enr_avg_per_element}")
    print(f"e_corr: {e_corr}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load model
    model = torch.load(pt_path, map_location='cpu', weights_only=False)

    # Remove DDP wrapper
    if hasattr(model, 'module'):
        model = model.module

    model.eval()
    species = torch.tensor(list(uniq_element.keys()))
    model.atomic_numbers = species.clone().detach()
    model.num_interactions = torch.tensor(nlayers)
    model.r_max = torch.tensor(cutoff)
    model = model.float().to(device)
    bam_models.get_edge_relative_vectors_with_pbc = get_edge_relative_vectors_with_pbc_lammps
    model.training_mode_for_lammps = True

    # Adjust variables for compatibility across model versions
    try:
        criterion = model.criterion
        criterion_value = model.criterion_value
        if criterion < criterion_value:
            regress_forces = "auto"
        else:
            regress_forces = "direct"
        model.regress_forces = regress_forces
    except:
        criterion = None
        model.criterion = criterion

    for module in model.modules():
        module.training_mode_for_lammps = True

    # Generate LAMMPS-BAM model
    lammps_model = LAMMPS_BAM(
        model,
        enr_avg_per_element=enr_avg_per_element,
        e_corr=e_corr
    )
    lammps_model = lammps_model.to(device)
    lammps_model_compiled = jit.compile(lammps_model)
    lammps_model_compiled.save(output_path)

    print(f"Successfully created LAMMPS model: {output_path}")

    if dist.is_initialized():
        dist.destroy_process_group()

    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Convert model.pt to LAMMPS-compatible TorchScript format'
    )
    parser.add_argument('--pkl', type=str, default='model.pkl',
                        help='Path to the model pkl file (default: model.pkl)')
    parser.add_argument('--pt', type=str, default='model.pt',
                        help='Path to the model pt file (default: model.pt)')
    parser.add_argument('--output', type=str, default=None,
                        help='Path to save the LAMMPS model (default: {pt}-lammps.pt)')
    parser.add_argument('--enr-avg-file', type=str, default=None,
                        help='Optional file containing enr_avg_per_element for universal potential')
    args = parser.parse_args()

    create_lammps_model(args.pkl, args.pt, args.output, args.enr_avg_file)
