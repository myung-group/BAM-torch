"""
CG (Coarse-Grained) simulation utilities for BAM-torch

Supports: water, lipid, DNA, RNA, protein, and custom molecular systems.

Scripts:
    - preprocess_cg.py: AA trajectory -> CG NPZ conversion (ASE or GROMACS source)
    - gromacs_to_cg.py: GROMACS AA trajectory -> CG NPZ (MARTINI-style residue mapping)
    - npz_to_lammps_data.py: NPZ -> LAMMPS data file (multi-type support)
    - aa_to_cg_data.py: AA LAMMPS data -> CG LAMMPS data (configurable atoms/molecule)
    - make_pt_cg.py: PKL model -> LAMMPS TorchScript (.pt)
    - evaluate_cg_model.py: Evaluate trained CG model (traj or NPZ input)
    - lammpsout_to_traj.py: LAMMPS dump+log -> ASE trajectory (auto element mapping)

Related utilities (bam_torch/utils/):
    - cg_mapping.py: Molecule-based CG mapping (water, methane, etc.)
    - residue_cg_mapping.py: Residue-based CG mapping (lipid, DNA, RNA, protein)
    - martini_mappings.py: MARTINI mapping registry for all supported residue types

See README_CG_WORKFLOW.md for detailed workflow documentation.
"""
