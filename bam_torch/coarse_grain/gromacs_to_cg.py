"""
Generic GROMACS All-Atom trajectory to Coarse-Grained NPZ converter.

Supports any combination of biomolecular systems:
- Lipids (DLIPC, DPPC, POPC, DOPC, ...)
- DNA nucleotides (DA, DT, DG, DC)
- RNA nucleotides (RA, RU, RG, RC)
- Proteins (20 standard amino acids)
- Water (MARTINI 4:1 mapping with spatial clustering)
- Mixed systems (e.g., protein + lipid bilayer + water)

Usage:
    python -m bam_torch.coarse_grain.gromacs_to_cg \\
        --topology step7_1.gro \\
        --trajectory rerun.trr \\
        --energy ener.edr \\
        --output system_cg.npz \\
        --residues DLIPC \\
        --include-water

    python -m bam_torch.coarse_grain.gromacs_to_cg --list-mappings
"""

import os
import sys
import argparse
import numpy as np
from typing import Optional, List

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from bam_torch.utils.residue_cg_mapping import (
    ResidueBasedCGMapper,
    FORCE_CONVERSION,
    KJ_MOL_TO_EV,
    _lazy_import_mda,
)
from bam_torch.utils.martini_mappings import print_martini_registry


def extract_energies(edr_path: str, n_frames: Optional[int] = None) -> Optional[np.ndarray]:
    """Extract potential energies from GROMACS EDR file.

    Uses 'gmx energy' command. Energies are converted from kJ/mol to eV.

    Args:
        edr_path: Path to .edr file
        n_frames: Max number of frames (truncate if needed)

    Returns:
        Energies array in eV, or None if extraction fails.
    """
    import subprocess
    import tempfile

    try:
        print(f"\nExtracting energies from {edr_path}...")

        with tempfile.NamedTemporaryFile(mode='w', suffix='.xvg', delete=False) as tmp:
            tmp_path = tmp.name

        cmd = f'echo "Potential" | gmx energy -f {edr_path} -o {tmp_path} 2>/dev/null'
        subprocess.run(cmd, shell=True, check=True)

        energies = []
        with open(tmp_path, 'r') as f:
            for line in f:
                if not line.startswith(('#', '@')):
                    parts = line.split()
                    if len(parts) >= 2:
                        energies.append(float(parts[1]))

        os.unlink(tmp_path)

        energies = np.array(energies)
        print(f"  Extracted {len(energies)} energy values")
        print(f"  Energy range: [{energies.min():.2f}, {energies.max():.2f}] kJ/mol")

        energies_ev = energies * KJ_MOL_TO_EV

        if n_frames is not None and len(energies) > n_frames:
            energies_ev = energies_ev[:n_frames]

        return energies_ev

    except Exception as e:
        print(f"  Warning: Could not extract energies: {e}")
        print("  Using zero energies (force-only training)")
        return None


def convert_gromacs_to_cg(
    topology: str,
    trajectory: str,
    output: str,
    energy: Optional[str] = None,
    residues: Optional[List[str]] = None,
    include_water: bool = False,
    waters_per_bead: int = 4,
    ignore_residues: Optional[List[str]] = None,
    custom_mappings: Optional[dict] = None,
    n_frames: Optional[int] = None,
    subsample: int = 1,
):
    """Convert GROMACS AA trajectory to CG NPZ format.

    Args:
        topology: Path to topology file (.gro or .tpr)
        trajectory: Path to trajectory file (.trr or .xtc)
        output: Output NPZ file path
        energy: Path to energy file (.edr), optional
        residues: List of residue names to map (None = auto-detect)
        include_water: Include water as W beads
        waters_per_bead: Waters per W bead (default: 4)
        ignore_residues: Residues to skip (e.g., ['NA', 'CL'])
        custom_mappings: Custom mapping overrides
        n_frames: Number of frames to process (None = all)
        subsample: Process every N-th frame
    """
    mda = _lazy_import_mda()

    print("=" * 70)
    print("GROMACS AA -> CG Conversion (MARTINI-style)")
    print("=" * 70)

    # Validate input files
    if not os.path.exists(trajectory):
        raise FileNotFoundError(f"Trajectory file not found: {trajectory}")
    if not os.path.exists(topology):
        raise FileNotFoundError(f"Topology file not found: {topology}")

    print(f"\nInput files:")
    print(f"  Topology:   {topology}")
    print(f"  Trajectory: {trajectory}")
    if energy:
        print(f"  Energy:     {energy}")
    print(f"\nOptions:")
    print(f"  Include water: {include_water}")
    if include_water:
        print(f"  Waters per bead: {waters_per_bead}")
    if ignore_residues:
        print(f"  Ignore residues: {ignore_residues}")

    # Load universe
    print("\nLoading trajectory...")
    u = mda.Universe(topology, trajectory)
    total_frames = len(u.trajectory)
    print(f"  Total frames: {total_frames}")

    # Determine frame indices
    if subsample > 1:
        frame_indices = list(range(0, total_frames, subsample))
    else:
        frame_indices = list(range(total_frames))

    if n_frames is not None:
        frame_indices = frame_indices[:n_frames]

    n_process = len(frame_indices)
    print(f"  Processing {n_process} frames (subsample={subsample})")

    # Create mapper
    mapper = ResidueBasedCGMapper(
        universe=u,
        residue_names=residues,
        custom_mappings=custom_mappings,
        include_water=include_water,
        waters_per_bead=waters_per_bead,
        ignore_residues=ignore_residues,
    )

    # Allocate output arrays
    n_beads = mapper.n_beads
    cg_positions = np.zeros((n_process, n_beads, 3), dtype=np.float32)
    cg_forces = np.zeros((n_process, n_beads, 3), dtype=np.float32)
    cg_cells = np.zeros((n_process, 3, 3), dtype=np.float32)

    # Process frames
    print("\nProcessing frames...")
    for i, frame_idx in enumerate(frame_indices):
        if (i + 1) % 100 == 0 or i == 0:
            print(f"  Frame {i + 1}/{n_process}")

        ts = u.trajectory[frame_idx]
        positions = ts.positions  # Angstrom

        if not ts.has_forces:
            raise ValueError(
                f"Frame {frame_idx} has no force data. "
                "Use 'gmx mdrun -rerun' to generate forces."
            )

        # Convert forces from kJ/(mol*nm) to eV/Angstrom
        forces = ts.forces * FORCE_CONVERSION

        # Get box dimensions for PBC-aware COM
        box = ts.dimensions[:3] if ts.dimensions is not None else None

        pos, frc, _ = mapper.map_frame(positions, forces, box)
        cg_positions[i] = pos
        cg_forces[i] = frc

        if box is not None:
            cg_cells[i] = np.diag(box)

    # Extract energies
    if energy and os.path.exists(energy):
        energies = extract_energies(energy, n_process)
    else:
        energies = None

    if energies is None:
        energies = np.zeros(n_process, dtype=np.float64)

    # Bead types
    bead_types = mapper.types

    # Metadata
    metadata = {
        'source': 'gromacs_to_cg',
        'mapping': 'MARTINI-style',
        'n_frames': n_process,
        'n_total_beads': n_beads,
        'n_bead_types': mapper.n_bead_types,
        'bead_type_names': mapper.bead_type_names,
        'residue_config': mapper.residue_config,
        'bond_topology': mapper.bond_topology,
        'include_water': include_water,
        'waters_per_bead': waters_per_bead if include_water else 0,
        'subsample': subsample,
        'bead_masses': mapper.bead_masses.tolist(),
        'unit_position': 'Angstrom',
        'unit_force': 'eV/Angstrom',
        'unit_energy': 'eV',
    }

    # Save NPZ
    print(f"\nSaving to {output}...")
    np.savez(
        output,
        positions=cg_positions,
        forces=cg_forces,
        energies=energies,
        types=bead_types,
        cells=cg_cells,
        metadata=np.array(metadata, dtype=object),
    )

    file_size = os.path.getsize(output) / 1e6

    print(f"\n{'='*70}")
    print("Conversion complete!")
    print(f"{'='*70}")
    print(f"  Output: {output} ({file_size:.1f} MB)")
    print(f"  Frames: {n_process}")
    print(f"  Total beads per frame: {n_beads}")
    print(f"  Bead types: {mapper.n_bead_types}")
    print(f"\n  Position range: [{cg_positions.min():.2f}, {cg_positions.max():.2f}] A")
    print(f"  Force range: [{cg_forces.min():.4f}, {cg_forces.max():.4f}] eV/A")
    print(f"  |F| mean: {np.linalg.norm(cg_forces, axis=-1).mean():.4f} eV/A")
    print(f"\n  Bead type mapping:")
    for type_id, name in sorted(mapper.bead_type_names.items()):
        count = (bead_types == type_id).sum()
        print(f"    {type_id}: {name} ({count} beads)")
    print(f"{'='*70}")


def main():
    parser = argparse.ArgumentParser(
        description='Convert GROMACS AA trajectory to CG NPZ (MARTINI-style)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Lipid system
  python -m bam_torch.coarse_grain.gromacs_to_cg \\
      --topology step7_1.gro --trajectory rerun.trr \\
      --energy ener.edr --output dlipc_cg.npz \\
      --residues DLIPC --include-water

  # Protein system
  python -m bam_torch.coarse_grain.gromacs_to_cg \\
      --topology protein.gro --trajectory rerun.trr \\
      --output protein_cg.npz --ignore-residues NA CL

  # DNA system
  python -m bam_torch.coarse_grain.gromacs_to_cg \\
      --topology dna.gro --trajectory rerun.trr \\
      --output dna_cg.npz --residues DA DT DG DC

  # List available mappings
  python -m bam_torch.coarse_grain.gromacs_to_cg --list-mappings
"""
    )

    parser.add_argument('--topology', '-t', type=str,
                        help='Topology file (.gro or .tpr)')
    parser.add_argument('--trajectory', '-f', type=str,
                        help='Trajectory file (.trr or .xtc)')
    parser.add_argument('--energy', '-e', type=str, default=None,
                        help='Energy file (.edr)')
    parser.add_argument('--output', '-o', type=str, default='system_cg.npz',
                        help='Output NPZ file (default: system_cg.npz)')
    parser.add_argument('--residues', '-r', nargs='+', default=None,
                        help='Residue names to map (default: auto-detect)')
    parser.add_argument('--include-water', action='store_true',
                        help='Include water as W beads (4:1 MARTINI)')
    parser.add_argument('--waters-per-bead', type=int, default=4,
                        help='Waters per W bead (default: 4)')
    parser.add_argument('--ignore-residues', nargs='+', default=None,
                        help='Residues to skip (e.g., NA CL)')
    parser.add_argument('--n-frames', '-n', type=int, default=None,
                        help='Number of frames to process (default: all)')
    parser.add_argument('--subsample', '-s', type=int, default=1,
                        help='Process every N-th frame (default: 1)')
    parser.add_argument('--list-mappings', action='store_true',
                        help='List available MARTINI mappings')

    args = parser.parse_args()

    if args.list_mappings:
        print_martini_registry()
        return

    if not args.topology or not args.trajectory:
        parser.print_help()
        print("\nError: --topology and --trajectory are required")
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


if __name__ == '__main__':
    main()
