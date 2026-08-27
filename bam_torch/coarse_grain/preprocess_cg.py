#!/usr/bin/env python
"""
CG Dataset Preprocessing Script

This script preprocesses an atomistic trajectory into a CG representation
and saves it as an NPZ file for faster training.

Supports two source types:
- ase: ASE-readable trajectory (water, methane, etc. via molecule-based mapping)
- gromacs: GROMACS trajectory (lipid, DNA, RNA, protein via residue-based MARTINI mapping)

Usage:
    # ASE trajectory (molecule-based mapping)
    python preprocess_cg.py --input water.traj --output water_cg.npz --mapping water

    # GROMACS trajectory (residue-based MARTINI mapping)
    python preprocess_cg.py --source-type gromacs --topology step7_1.gro --trajectory rerun.trr \
        --energy ener.edr --output lipid_cg.npz --residues DLIPC --include-water

    # List all available mappings
    python preprocess_cg.py --list-presets
"""

import argparse
import sys
import os

# Add parent path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))


def main():
    parser = argparse.ArgumentParser(
        description='Preprocess atomistic trajectory to CG NPZ format',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # ASE trajectory (molecule-based mapping)
  python preprocess_cg.py -i water.traj -o water_cg.npz -m water
  python preprocess_cg.py -i water.traj -o water_cg.npz -m water -n 4000

  # GROMACS trajectory (residue-based MARTINI mapping)
  python preprocess_cg.py --source-type gromacs --topology system.gro \\
      --trajectory rerun.trr --energy ener.edr -o system_cg.npz \\
      --residues DLIPC --include-water --waters-per-bead 4

  # List all available mappings (molecule-based + MARTINI)
  python preprocess_cg.py --list-presets

  # Merge multiple NPZ files into one multi-system file
  python preprocess_cg.py --merge sys1.npz sys2.npz sys3.npz -o merged.npz

  # Split into train/valid/test (stratified by system)
  python preprocess_cg.py --split merged.npz -o ./output_dir --ntrain 900 --nvalid 100 --ntest 100

  # Show info about existing NPZ file
  python preprocess_cg.py --info water_cg.npz
"""
    )

    # Source type
    parser.add_argument('--source-type', '-s', type=str, default='ase',
                        choices=['ase', 'gromacs'],
                        help='Input source type: ase (ASE trajectory) or gromacs (default: ase)')

    # Common arguments
    parser.add_argument('--output', '-o', type=str,
                        help='Output NPZ file path')
    parser.add_argument('--n-frames', '-n', type=int, default=None,
                        help='Number of frames to process (default: all)')

    # ASE-specific arguments
    ase_group = parser.add_argument_group('ASE source options')
    ase_group.add_argument('--input', '-i', type=str,
                           help='Input ASE trajectory file')
    ase_group.add_argument('--mapping', '-m', type=str, default='water',
                           help='CG mapping preset name (default: water)')
    ase_group.add_argument('--start', type=int, default=0,
                           help='Starting frame index (default: 0)')
    ase_group.add_argument('--stride', type=int, default=1,
                           help='Frame stride (default: 1)')

    # GROMACS-specific arguments
    gro_group = parser.add_argument_group('GROMACS source options')
    gro_group.add_argument('--topology', '-t', type=str,
                           help='GROMACS topology file (.gro, .pdb, .tpr)')
    gro_group.add_argument('--trajectory', type=str,
                           help='GROMACS trajectory file (.trr, .xtc)')
    gro_group.add_argument('--energy', type=str,
                           help='GROMACS energy file (.edr)')
    gro_group.add_argument('--residues', nargs='*', default=None,
                           help='Residue names to map (default: auto-detect)')
    gro_group.add_argument('--include-water', action='store_true',
                           help='Include water in CG mapping')
    gro_group.add_argument('--waters-per-bead', type=int, default=4,
                           help='Number of waters per CG bead (default: 4)')
    gro_group.add_argument('--ignore-residues', nargs='*', default=None,
                           help='Residue names to ignore (e.g., NA CL)')
    gro_group.add_argument('--subsample', type=int, default=1,
                           help='Subsample every N frames (default: 1)')

    # Merge command
    merge_group = parser.add_argument_group('Merge options')
    merge_group.add_argument('--merge', nargs='+', metavar='NPZ_FILE',
                             help='Merge multiple CG NPZ files into one multi-system file')

    # Split command
    split_group = parser.add_argument_group('Split options')
    split_group.add_argument('--split', type=str, metavar='NPZ_FILE',
                             help='Split a multi-system NPZ into train/valid/test sets')
    split_group.add_argument('--ntrain', type=int, default=0,
                             help='Number of training frames')
    split_group.add_argument('--nvalid', type=int, default=0,
                             help='Number of validation frames')
    split_group.add_argument('--ntest', type=int, default=0,
                             help='Number of test frames')
    split_group.add_argument('--seed', type=int, default=42,
                             help='Random seed for split (default: 42)')

    # Info/listing commands
    parser.add_argument('--list-presets', action='store_true',
                        help='List available CG mapping presets')
    parser.add_argument('--list-martini', action='store_true',
                        help='List available MARTINI residue mappings')
    parser.add_argument('--info', type=str, metavar='NPZ_FILE',
                        help='Show info about an NPZ file')

    args = parser.parse_args()

    # Handle special commands
    if args.list_presets:
        from bam_torch.utils.cg_mapping import print_available_presets
        print_available_presets()
        print("\nFor MARTINI residue mappings, use --list-martini")
        return

    if args.list_martini:
        from bam_torch.utils.martini_mappings import print_martini_registry
        print_martini_registry()
        return

    if args.info:
        from bam_torch.utils.cg_dataset import print_npz_info
        print_npz_info(args.info)
        return

    if args.merge:
        from bam_torch.utils.cg_dataset import merge_cg_npz
        if not args.output:
            print("Error: --output is required for merge")
            sys.exit(1)
        merge_cg_npz(npz_paths=args.merge, output_path=args.output)
        return

    if args.split:
        from bam_torch.utils.cg_dataset import split_cg_npz
        if not args.output:
            print("Error: --output (output directory) is required for split")
            sys.exit(1)
        if args.ntrain == 0 and args.nvalid == 0:
            print("Error: at least --ntrain or --nvalid must be > 0")
            sys.exit(1)
        split_cg_npz(
            npz_path=args.split,
            output_dir=args.output,
            ntrain=args.ntrain,
            nvalid=args.nvalid,
            ntest=args.ntest,
            random_seed=args.seed,
        )
        return

    # Route to appropriate source handler
    if args.source_type == 'gromacs':
        _run_gromacs_preprocessing(args)
    else:
        _run_ase_preprocessing(args)


def _run_ase_preprocessing(args):
    """Run ASE-based preprocessing (molecule-based mapping)."""
    from bam_torch.utils.cg_dataset import preprocess_to_cg

    if not args.input or not args.output:
        print("Error: --input and --output are required for ASE preprocessing")
        sys.exit(1)

    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}")
        sys.exit(1)

    preprocess_to_cg(
        input_path=args.input,
        output_path=args.output,
        mapping_config=args.mapping,
        n_frames=args.n_frames,
        start_frame=args.start,
        stride=args.stride,
        show_progress=True
    )

    print(f"\nPreprocessing complete!")
    print(f"Use this NPZ file in your input_cg.json:")
    print(f"""
{{
    "trainer": "cg",
    "cg_config": {{
        "fname_npz": "{args.output}",
        "cutoff": 10.0
    }},
    ...
}}
""")


def _run_gromacs_preprocessing(args):
    """Run GROMACS-based preprocessing (residue-based MARTINI mapping)."""
    from bam_torch.coarse_grain.gromacs_to_cg import convert_gromacs_to_cg

    if not args.topology or not args.trajectory or not args.output:
        print("Error: --topology, --trajectory, and --output are required for GROMACS preprocessing")
        sys.exit(1)

    if not os.path.exists(args.topology):
        print(f"Error: Topology file not found: {args.topology}")
        sys.exit(1)

    if not os.path.exists(args.trajectory):
        print(f"Error: Trajectory file not found: {args.trajectory}")
        sys.exit(1)

    convert_gromacs_to_cg(
        topology=args.topology,
        trajectory=args.trajectory,
        output=args.output,
        energy=args.energy,
        residues=args.residues,
        include_water=args.include_water,
        waters_per_bead=args.waters_per_bead,
        ignore_residues=args.ignore_residues,
        n_frames=args.n_frames,
        subsample=args.subsample,
    )

    print(f"\nGROMACS preprocessing complete!")
    print(f"Use this NPZ file in your input_cg.json:")
    print(f"""
{{
    "trainer": "cg",
    "cg_config": {{
        "fname_npz": "{args.output}",
        "cutoff": 15.0
    }},
    ...
}}
""")


if __name__ == '__main__':
    main()
