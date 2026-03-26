"""
BAM-torch LAMMPS interface module.

This module provides tools for converting BAM models to LAMMPS-compatible format
and utilities for working with LAMMPS trajectory data.

Main conversion scripts:
- create_lammps.py: All-atom models
- create_lammps_cg.py: CG models (total learning)
- create_lammps_cg_delta.py: CG models (delta learning with prior)

LAMMPS wrappers:
- lammps_bam.py: Standard LAMMPS_BAM wrapper
- lammps_bam_delta.py: LAMMPS_BAM_Delta wrapper with prior force calculation
"""

from bam_torch.lammps.lammps_bam import LAMMPS_BAM
from bam_torch.lammps.lammps_bam_delta import (
    LAMMPS_BAM_Delta,
    ZBLPriorTorch,
    D2PriorTorch,
    UniversalPriorTorch,
)

__all__ = [
    'LAMMPS_BAM',
    'LAMMPS_BAM_Delta',
    'ZBLPriorTorch',
    'D2PriorTorch',
    'UniversalPriorTorch',
]
