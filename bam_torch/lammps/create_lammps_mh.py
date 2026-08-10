"""
Convert multihead model.pt to LAMMPS-compatible TorchScript format.

Usage:
    python -m bam_torch.lammps.create_lammps_mh --pkl model.pkl --pt model.pt --output model-lammps.pt --head head_name

Or in Python:
    from bam_torch.lammps.create_lammps_mh import create_lammps_model_mh
    create_lammps_model_mh('model.pkl', 'model.pt', 'model-lammps.pt', head='head_name')
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


def create_lammps_model_mh(pkl_path='model.pkl', pt_path='model.pt', output_path=None,
                           head=None, enr_avg_file=None):
    """Convert multihead model.pt to LAMMPS-compatible TorchScript format.

    Args:
        pkl_path: Path to the model pkl file
        pt_path: Path to the model pt file
        output_path: Path to save the LAMMPS model (default: {pt_path}-lammps.pt)
        head: Head name or index to use (default: first head)
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

    # Head selection for multihead models
    ckpt_input_json = pckl.get('input.json', {})
    multihead_config = ckpt_input_json.get('multihead', {})
    datasets = multihead_config.get('datasets', [])
    is_multihead = len(datasets) > 0

    if is_multihead:
        head_names = [ds.get('name', f'head_{i}') for i, ds in enumerate(datasets)]

        # Determine selected head
        if head is None:
            selected_head = head_names[0]
            selected_head_idx = 0
        elif isinstance(head, int):
            selected_head_idx = head
            selected_head = head_names[selected_head_idx]
        else:
            selected_head = head
            selected_head_idx = head_names.index(head) if head in head_names else 0

        print(f"Multihead model detected: {len(datasets)} heads")
        print(f"  Available heads: {head_names}")
        print(f"  Selected head: {selected_head} (index: {selected_head_idx})")

        # Use per-head E0s
        per_head_enr_avg = pckl.get('per_head_enr_avg', {})
        if selected_head_idx in per_head_enr_avg:
            enr_avg_per_element = per_head_enr_avg[selected_head_idx].get(
                'enr_avg_per_element', pckl['enr_avg_per_element']
            )
            print(f"  Using per-head E0s for head {selected_head_idx}")
        else:
            enr_avg_per_element = pckl['enr_avg_per_element']
            print(f"  No per-head E0s found, using global E0s")

        # Use per-head scale_shift
        per_head_scale_shift = pckl.get('per_head_scale_shift', {})
        if selected_head_idx in per_head_scale_shift and per_head_scale_shift[selected_head_idx]:
            head_shifts = per_head_scale_shift[selected_head_idx]
            e_corr = sum(head_shifts) / len(head_shifts)
            print(f"  Using per-head scale_shift: {e_corr:.4f}")
        else:
            e_corr = 0.0
            print(f"  No per-head scale_shift found, using 0")

        uniq_element = pckl['uniq_element']
    else:
        # Single-head model: use original logic
        selected_head = None
        if enr_avg_file is not None:
            with open(enr_avg_file, 'r', encoding='utf-8') as file:
                content = file.read()
            enr_avg_per_element, uniq_element = ast.literal_eval(content)
        else:
            uniq_element = pckl['uniq_element']
            enr_avg_per_element = pckl['enr_avg_per_element']

        e_corr = torch.tensor(pckl['valid_scale_shift'])
        e_corr = e_corr.flatten().mean().item()
        print("Single-head model detected")

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
        e_corr=e_corr,
        head=selected_head
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
        description='Convert multihead model.pt to LAMMPS-compatible TorchScript format'
    )
    parser.add_argument('--pkl', type=str, default='model.pkl',
                        help='Path to the model pkl file (default: model.pkl)')
    parser.add_argument('--pt', type=str, default='model.pt',
                        help='Path to the model pt file (default: model.pt)')
    parser.add_argument('--output', type=str, default=None,
                        help='Path to save the LAMMPS model (default: {pt}-lammps.pt)')
    parser.add_argument('--head', type=str, default=None,
                        help='Head name to use for multihead models (default: first head)')
    parser.add_argument('--enr-avg-file', type=str, default=None,
                        help='Optional file containing enr_avg_per_element for universal potential')
    args = parser.parse_args()

    create_lammps_model_mh(args.pkl, args.pt, args.output, args.head, args.enr_avg_file)
