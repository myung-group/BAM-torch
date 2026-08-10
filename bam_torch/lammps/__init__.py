"""
BAM-torch LAMMPS interface module.

This module provides tools for converting BAM models to LAMMPS-compatible format
and utilities for working with LAMMPS trajectory data.

Main conversion scripts:
- create_lammps.py: All-atom models

LAMMPS wrappers:
- lammps_bam.py: Standard LAMMPS_BAM wrapper
"""

from bam_torch.lammps.lammps_bam import LAMMPS_BAM

__all__ = [
    'LAMMPS_BAM',
]
