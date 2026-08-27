"""
Convert Delta Learning CG model to LAMMPS-compatible format.

This script handles models trained with delta learning, where:
- Training: model learns F_delta = F_total - F_prior
- Inference: F_total = F_prior + F_delta (model output)

The LAMMPS wrapper includes prior force calculation (ZBL + D2).

Usage:
    python -m bam_torch.lammps.create_lammps_cg_delta --pkl model_cg_delta.pkl --pt model_cg_delta.pt

Or in Python:
    from bam_torch.lammps.create_lammps_cg_delta import create_lammps_cg_delta_model
    create_lammps_cg_delta_model('model_cg_delta.pkl', 'model_cg_delta.pt')
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
from typing import Dict, List, Optional

from ase.data import chemical_symbols

from bam_torch.model import models as bam_models
from bam_torch.model.models import get_edge_relative_vectors_with_pbc_lammps
from bam_torch.lammps.lammps_bam_delta import LAMMPS_BAM_Delta

# Mapping CG bead types to dummy atomic numbers
# Dynamically generated: CG type i -> element i+1 (H=1, He=2, Li=3, ...)
# Supports up to 100 CG types
CG_TYPE_TO_ELEMENT = {i: (chemical_symbols[i + 1], i + 1) for i in range(100)}


def create_lammps_cg_delta_model(
    pkl_path: str,
    pt_path: str,
    output_path: str = None,
    prior_config_override: Dict = None
) -> str:
    """Convert Delta Learning CG model to LAMMPS-compatible format.

    Args:
        pkl_path: Path to the CG model pkl file
        pt_path: Path to the model pt file
        output_path: Output path for LAMMPS model (default: <pt_name>-lammps-delta.pt)
        prior_config_override: Optional override for prior configuration.
                               If None, uses config from pkl file automatically.

    Returns:
        output_path: Path where the LAMMPS model was saved
    """
    # Set default output path
    if output_path is None:
        output_path = pt_path.replace('.pt', '-lammps-delta.pt')

    # Initialize distributed process
    if not dist.is_initialized():
        dist.init_process_group(
            backend='nccl' if torch.cuda.is_available() else 'gloo',
            init_method='env://',
            world_size=1,
            rank=0
        )

    # Load pkl file for metadata
    print(f"Loading Delta Learning CG model from {pkl_path}...")
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

    # Get CG type info
    uniq_element = pckl.get('uniq_element', {0: 0})
    enr_avg_per_element = pckl.get('enr_avg_per_element', {0: 0.0})

    # Get number of beads from training data info
    n_beads = pckl.get('n_beads', None)
    if n_beads is None:
        # Try to infer from ntrain data shape or use default
        n_beads = cfg.get('n_beads', 256)
    print(f"  - Number of CG beads: {n_beads}")

    # Get energy correction
    valid_scale_shift_origin = pckl.get('valid_scale_shift_origin', None)
    if valid_scale_shift_origin is not None:
        if isinstance(valid_scale_shift_origin, torch.Tensor):
            e_corr = valid_scale_shift_origin.item()
        else:
            e_corr = float(valid_scale_shift_origin)
        # Force-only models (enr_lambda=0) may have NaN e_corr
        # 무-prior create_lammps_cg.py와 동일한 가드 — 없으면 LAMMPS에서
        # E_pair = nan 이 되고 minimize가 첫 스텝에 멈춘다.
        if e_corr != e_corr:  # NaN check
            print("  Warning: e_corr is NaN (force-only model). Setting to 0.0")
            e_corr = 0.0
    else:
        e_corr = 0.0

    # Get prior configuration from pkl file
    cg_config = cfg.get('cg_config', {})
    prior_config = cg_config.get('prior', None)

    if prior_config is None:
        # Try alternative location
        delta_config = cfg.get('delta_learning', {})
        prior_config = delta_config.get('prior', None)

    # Apply overrides if provided
    if prior_config_override is not None:
        if prior_config is None:
            prior_config = prior_config_override
        else:
            # Merge: override takes precedence
            prior_config.update(prior_config_override)

    if prior_config is None:
        raise ValueError(
            "No prior configuration found in model pkl file.\n"
            "Please provide --zbl-cutoff, --d2-cutoff, --atomic-number options,\n"
            "or ensure the model was trained with delta_learning config."
        )

    prior_type = prior_config.get('type', 'universal')

    print(f"\nDelta Learning CG Model Configuration:")
    print(f"  - Number of CG types: {num_cg_types}")
    print(f"  - Cutoff: {cutoff} Å")
    print(f"  - Number of layers: {nlayers}")
    print(f"  - enr_avg_per_element: {enr_avg_per_element}")
    print(f"  - e_corr: {e_corr}")
    print(f"\nPrior Configuration:")
    print(f"  - Type: {prior_type}")

    if prior_type == 'harmonic_repulsive':
        # Harmonic + Repulsive prior
        topo = prior_config.get('bond_topology', {})
        n_beads_per_mol = topo.get('n_beads_per_mol', 21)
        harmonic_cfg = prior_config.get('harmonic', {})
        repulsive_cfg = prior_config.get('repulsive', {})
        print(f"  - Bond topology: {n_beads_per_mol} beads/mol, "
              f"{len(topo.get('bonds', []))} bonds/mol")
        print(f"  - Repulsive epsilon: {repulsive_cfg.get('epsilon', 0.001)} eV")
        print(f"  - Repulsive cutoff: {repulsive_cfg.get('cutoff', 10.0)} Å")
        print(f"  - Number of beads: {n_beads}")
    else:
        # Universal (ZBL + D2) prior
        atomic_numbers_base = prior_config.get('atomic_numbers', [8])
        zbl_cutoff = prior_config.get('zbl_cutoff', 5.0)
        d2_cutoff = prior_config.get('d2_cutoff', 20.0)

        # Expand atomic numbers to match n_beads
        if len(atomic_numbers_base) == 1:
            atomic_numbers = atomic_numbers_base * n_beads
        elif len(atomic_numbers_base) < n_beads:
            atomic_numbers = (atomic_numbers_base * (n_beads // len(atomic_numbers_base) + 1))[:n_beads]
        else:
            atomic_numbers = atomic_numbers_base[:n_beads]

        prior_config['atomic_numbers'] = atomic_numbers

        print(f"  - ZBL cutoff: {zbl_cutoff} Å")
        print(f"  - D2 cutoff: {d2_cutoff} Å")
        print(f"  - Number of beads: {len(atomic_numbers)}")

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
    # e.g., CG type 0 -> H (Z=1), CG type 1 -> He (Z=2), ...
    # Must match create_lammps_cg.py (no-prior path) so that
    # `pair_coeff * * <pt> H He Li Be B` resolves; using the raw 0-based
    # uniq_element keys makes LAMMPS type N unmappable.
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

    # Generate LAMMPS-BAM-Delta model
    print(f"\nCreating LAMMPS-compatible Delta Learning model...")

    # Extra kwargs for harmonic_repulsive prior
    extra_kwargs = {}
    if prior_type == 'harmonic_repulsive':
        # Build types array from uniq_element mapping
        # Need to load types from NPZ or reconstruct from bead info
        npz_path = cg_config.get('fname_npz', None)
        if npz_path and os.path.exists(npz_path):
            import numpy as np
            npz_data = np.load(npz_path, allow_pickle=True)
            types_array = npz_data['types'].tolist()
            # n_beads는 pkl/cfg에 없으면 기본 256으로 잡히는데, 그대로 쓰면
            # n_mol = n_beads // n_beads_per_mol 이 실제보다 작아져 뒤쪽 분자들에
            # bond prior와 분자내 exclusion이 통째로 빠진다. NPZ 실측으로 정정.
            if len(types_array) != n_beads:
                print(f"  ! n_beads 정정: {n_beads} -> {len(types_array)} (NPZ types 실측)")
                n_beads = len(types_array)
        else:
            # Reconstruct types: n_mol molecules × n_beads_per_mol types
            topo = prior_config.get('bond_topology', {})
            n_beads_per_mol = topo.get('n_beads_per_mol', 21)
            n_mol = n_beads // n_beads_per_mol
            types_array = list(range(n_beads_per_mol)) * n_mol
        extra_kwargs['n_atoms'] = n_beads
        extra_kwargs['types_array'] = types_array

    # Get bond topology for CG bond flag (if use_bond_flag is enabled)
    # 무-prior create_lammps_cg.py와 동일하게 처리해야 한다 — 빠뜨리면
    # radial MLP 입력이 1채널 모자라 run 0에서 shape mismatch로 죽는다.
    bond_topology = None
    if cfg.get('use_bond_flag', False) and 'bond_topology' in cg_config:
        bond_topology = cg_config['bond_topology']
        print(f"  - Bond topology: {bond_topology['n_beads_per_mol']} beads/mol, "
              f"{len(bond_topology['bonds'])} bonds")

    lammps_model = LAMMPS_BAM_Delta(
        model,
        enr_avg_per_element=enr_avg_per_element,
        e_corr=e_corr,
        prior_config=prior_config,
        bond_topology=bond_topology,
        **extra_kwargs
    )
    lammps_model = lammps_model.to(device)

    # Compile and save
    lammps_model_compiled = jit.compile(lammps_model)
    lammps_model_compiled.save(output_path)

    print(f"\n✓ LAMMPS Delta Learning CG model saved to: {output_path}")
    print(f"\nThis model includes:")
    print(f"  - ML model (predicts F_delta)")
    print(f"  - Prior force field ({prior_type})")
    print(f"  - At inference: F_total = F_prior + F_delta")
    print(f"\nTo use in LAMMPS:")
    print(f"  pair_style bam no_domain_decomposition cutoff {cutoff}")
    print(f"  pair_coeff * * {output_path} water")

    # Cleanup
    if dist.is_initialized():
        dist.destroy_process_group()

    return output_path


def main():
    parser = argparse.ArgumentParser(
        description='Convert Delta Learning CG model to LAMMPS format',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage (all settings read from pkl file automatically)
  python -m bam_torch.lammps.create_lammps_cg_delta --pkl model_cg_delta.pkl --pt model_cg_delta.pt

  # Override specific settings if needed
  python -m bam_torch.lammps.create_lammps_cg_delta --pkl model.pkl --pt model.pt \\
      --zbl-cutoff 4.0 --d2-cutoff 15.0

  # Custom output path
  python -m bam_torch.lammps.create_lammps_cg_delta --pkl model.pkl --pt model.pt \\
      --output my_lammps_model.pt

Note: Prior configuration (atomic_numbers, cutoffs) is automatically read from
the pkl file. Command line arguments are optional overrides.
"""
    )
    parser.add_argument('--pkl', default='model_cg_delta.pkl',
                        help='Path to the CG model pkl file')
    parser.add_argument('--pt', default='model_cg_delta.pt',
                        help='Path to the model pt file')
    parser.add_argument('--output', default=None,
                        help='Output path for LAMMPS model (default: <pt_name>-lammps-delta.pt)')
    parser.add_argument('--zbl-cutoff', type=float, default=None,
                        help='Override ZBL prior cutoff in Å (default: from pkl file)')
    parser.add_argument('--d2-cutoff', type=float, default=None,
                        help='Override D2 dispersion cutoff in Å (default: from pkl file)')
    parser.add_argument('--n-beads', type=int, default=None,
                        help='Override number of CG beads (default: from pkl file)')
    parser.add_argument('--atomic-number', type=int, default=None,
                        help='Override atomic number for CG beads (default: from pkl file)')
    args = parser.parse_args()

    # Build prior config override from command line args (only non-None values)
    prior_config_override = {}
    if args.zbl_cutoff is not None:
        prior_config_override['zbl_cutoff'] = args.zbl_cutoff
    if args.d2_cutoff is not None:
        prior_config_override['d2_cutoff'] = args.d2_cutoff
    if args.atomic_number is not None:
        prior_config_override['atomic_numbers'] = [args.atomic_number]
    if args.n_beads is not None:
        prior_config_override['n_beads'] = args.n_beads

    # Pass None if no overrides provided
    if not prior_config_override:
        prior_config_override = None

    create_lammps_cg_delta_model(
        args.pkl,
        args.pt,
        args.output,
        prior_config_override=prior_config_override
    )


if __name__ == "__main__":
    main()
