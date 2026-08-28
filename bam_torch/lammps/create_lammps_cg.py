"""
Convert CG model to LAMMPS-compatible format.

Usage:
    python -m bam_torch.lammps.create_lammps_cg --pkl model_cg.pkl --pt model_cg.pt

Or in Python:
    from bam_torch.lammps.create_lammps_cg import create_lammps_cg_model
    create_lammps_cg_model('model_cg.pkl', 'model_cg.pt', 'model_cg-lammps.pt')
"""

import os
os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "29500"
os.environ["RANK"] = "0"
os.environ["WORLD_SIZE"] = "1"

import argparse
import torch
import torch.distributed as dist
from e3nn.util import jit

from ase.data import chemical_symbols

from bam_torch.model import models as bam_models
from bam_torch.model.models import get_edge_relative_vectors_with_pbc_lammps
from bam_torch.lammps.lammps_bam import LAMMPS_BAM


# Mapping CG bead types to dummy atomic numbers
# Dynamically generated: CG type i -> element i+1 (H=1, He=2, Li=3, ...)
# Supports up to 1000 CG types
# CG types beyond the periodic table (uMLP vocab >= 118) get a synthetic label X<n>;
# pair_bam only compares the integer Z, so any unique integer works.
CG_TYPE_TO_ELEMENT = {i: ((chemical_symbols[i + 1] if i + 1 < len(chemical_symbols) else f"X{i + 1}"), i + 1)
                      for i in range(1000)}


def create_lammps_cg_model(pkl_path, pt_path, output_path=None, type_mapping=None):
    """Convert CG model to LAMMPS-compatible format.

    Args:
        pkl_path: Path to the CG model pkl file
        pt_path: Path to the model pt file
        output_path: Output path for LAMMPS model (default: <pt_name>-lammps.pt)
        type_mapping: Optional dict mapping CG type -> element symbol
                      e.g., {0: 'H', 1: 'He'}
                      If None, uses default CG_TYPE_TO_ELEMENT

    Returns:
        output_path: Path where the LAMMPS model was saved
    """
    # Set default output path
    if output_path is None:
        output_path = pt_path.replace('.pt', '-lammps.pt')

    # Initialize distributed process
    if not dist.is_initialized():
        dist.init_process_group(
            backend='nccl' if torch.cuda.is_available() else 'gloo',
            init_method='env://',
            world_size=1,
            rank=0
        )

    # Load pkl file for metadata
    print(f"Loading CG model from {pkl_path}...")
    pckl = torch.load(pkl_path, map_location='cpu', weights_only=False)

    # Verify this is a CG model
    is_cg = pckl.get('is_cg_model', False)
    if is_cg:
        print("✓ Confirmed: This is a CG model")
    else:
        print("Warning: This doesn't appear to be a CG model. Proceeding anyway...")

    # Extract config
    cfg = pckl['input.json']
    nlayers = cfg['nlayers']
    cutoff = cfg['cutoff']
    num_cg_types = cfg.get('num_cg_types', 1)

    # Get CG type info (equivalent to uniq_element for all-atom)
    uniq_element = pckl.get('uniq_element', {0: 0})
    enr_avg_per_element = pckl.get('enr_avg_per_element', {0: 0.0})

    # Get energy correction from valid_scale_shift_origin (total energy shift)
    # NOTE: valid_scale_shift is per-atom scale factor, NOT the energy correction
    # valid_scale_shift_origin is the actual energy shift that should be added
    valid_scale_shift_origin = pckl.get('valid_scale_shift_origin', None)
    if valid_scale_shift_origin is not None:
        if isinstance(valid_scale_shift_origin, torch.Tensor):
            e_corr = valid_scale_shift_origin.item()
        else:
            e_corr = float(valid_scale_shift_origin)
        # Force-only models (enr_lambda=0) may have NaN e_corr
        if e_corr != e_corr:  # NaN check
            print("  Warning: e_corr is NaN (force-only model). Setting to 0.0")
            e_corr = 0.0
    else:
        e_corr = 0.0

    print(f"\nCG Model Configuration:")
    print(f"  - Number of CG types: {num_cg_types}")
    print(f"  - Cutoff: {cutoff} Å")
    print(f"  - Number of layers: {nlayers}")
    print(f"  - enr_avg_per_element: {enr_avg_per_element}")
    print(f"  - e_corr: {e_corr}")

    # Print CG mapping info
    cg_mapping = pckl.get('cg_mapping_config', {})
    if cg_mapping:
        print(f"  - CG mapping: {cg_mapping.get('formula', 'N/A')} -> {len(cg_mapping.get('beads', []))} bead(s)")
        print(f"  - Method: {cg_mapping.get('method', 'N/A')}")

    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nUsing device: {device}")

    # Load model
    print(f"Loading model from {pt_path}...")
    model = torch.load(pt_path, map_location='cpu', weights_only=False)

    # Remove DDP wrapper if present
    if hasattr(model, 'module'):
        model = model.module

    model.eval()

    # Set model attributes for LAMMPS compatibility
    # Map CG type indices to dummy atomic numbers via CG_TYPE_TO_ELEMENT
    # e.g., CG type 0 → H (Z=1), CG type 1 → He (Z=2), ...
    # Must use num_cg_types (not just uniq_element keys) to match
    # the model's embedding layer size (one-hot dimension)
    atomic_nums = [CG_TYPE_TO_ELEMENT[t][1] for t in range(num_cg_types)]
    model.atomic_numbers = torch.tensor(atomic_nums)
    model.num_interactions = torch.tensor(nlayers)
    model.r_max = torch.tensor(cutoff)
    model = model.float().to(device)

    # Replace edge vector function for LAMMPS
    bam_models.get_edge_relative_vectors_with_pbc = get_edge_relative_vectors_with_pbc_lammps
    model.training_mode_for_lammps = True

    # Adjust force regression mode
    try:
        criterion = model.criterion
        criterion_value = model.criterion_value
        if criterion < criterion_value:
            regress_forces = "auto"
        else:
            regress_forces = "direct"
        model.regress_forces = regress_forces
    except:
        model.criterion = None

    # Set LAMMPS mode for all modules
    for module in model.modules():
        module.training_mode_for_lammps = True

    # Get bond topology for CG bond flag (if use_bond_flag is enabled)
    bond_topology = None
    cg_config = cfg.get('cg_config', {})
    if cfg.get('use_bond_flag', False) and 'bond_topology' in cg_config:
        bond_topology = cg_config['bond_topology']
        print(f"  - Bond topology: {bond_topology['n_beads_per_mol']} beads/mol, "
              f"{len(bond_topology['bonds'])} bonds")

    # Generate LAMMPS-BAM model
    print(f"\nCreating LAMMPS-compatible model...")
    # Multihead ckpt: LAMMPS_BAM defaults to heads[-1] (MACE convention),
    # but cg_multihead puts the TARGET at datasets[0] -> deploy head 0.
    deploy_head_kw = {}
    if getattr(model, "heads", None) and len(getattr(model, "heads")) > 1 and model.heads[0] is not None:
        deploy_head_kw["head"] = model.heads[0]
        print(f"  - Multihead model: deploying head 0 = {model.heads[0]!r} (of {model.heads})")
    lammps_model = LAMMPS_BAM(
        model,
        enr_avg_per_element=enr_avg_per_element,
        e_corr=e_corr,
        bond_topology=bond_topology,
        **deploy_head_kw,
    )
    lammps_model = lammps_model.to(device)

    # Compile and save
    lammps_model_compiled = jit.compile(lammps_model)
    lammps_model_compiled.save(output_path)

    print(f"\n✓ LAMMPS CG model saved to: {output_path}")
    print(f"\nTo use in LAMMPS, you'll need:")
    print(f"  1. LAMMPS with ML-IAP/MLIAP package")
    print(f"  2. pair_style mliap model bam {output_path}")
    print(f"  3. CG data file with bead types matching the training")

    # Cleanup
    if dist.is_initialized():
        dist.destroy_process_group()

    return output_path


def main():
    parser = argparse.ArgumentParser(description='Convert CG model to LAMMPS format')
    parser.add_argument('--pkl', default='model_cg.pkl',
                        help='Path to the CG model pkl file')
    parser.add_argument('--pt', default='model_cg.pt',
                        help='Path to the model pt file')
    parser.add_argument('--output', default=None,
                        help='Output path for LAMMPS model (default: <pt_name>-lammps.pt)')
    args = parser.parse_args()

    create_lammps_cg_model(args.pkl, args.pt, args.output)


if __name__ == "__main__":
    main()
