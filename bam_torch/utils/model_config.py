"""
Centralized model configuration utilities for BAM-torch.

Resolves the naming inconsistency between config JSON keys and
model constructor parameters (Issue #5, Phase 3.5):

    Config key          →  Constructor param
    ──────────────────     ─────────────────
    num_radial_basis    →  num_basis_func
    hidden_channels     →  hidden_irreps      (string → o3.Irreps)
    output_channels     →  output_irreps      (string, model converts)
    regress_forces      →  regress_forces     (bool → "autograd"/"false")

Usage:
    from bam_torch.utils.model_config import (
        parse_model_config,
        parse_cueq_config,
        parse_charge_config,
    )

    kwargs = parse_model_config(json_data)
    kwargs['cueq_config'] = parse_cueq_config(json_data)
    model = ModelClass(**kwargs)
"""

from e3nn import o3


def parse_model_config(json_data):
    """Parse JSON config into model constructor kwargs.

    Handles the key renaming and type conversions that are duplicated
    across BaseTrainer.set_model(), CDTrainer.set_model(),
    CDTrainerV3.set_model(), and CDRACECalculator._build_model().

    Returns:
        dict with constructor-ready keyword arguments:
            cutoff, num_species, avg_num_neighbors, max_ell,
            num_basis_func, hidden_irreps, nlayers, features_dim,
            output_irreps, active_fn, regress_forces
    """
    hidden_irreps = o3.Irreps(
        json_data.get('hidden_channels', "64x0e+64x1o+64x2e")
    )

    regress_forces = json_data.get('regress_forces', "auto")
    if regress_forces is True:
        regress_forces = "autograd"
    elif regress_forces is False:
        regress_forces = "false"

    return {
        'cutoff': json_data.get('cutoff', 6.0),
        'num_species': json_data.get('num_species', 4),
        'avg_num_neighbors': json_data.get('avg_num_neighbors', 30),
        'max_ell': json_data.get('max_ell', 3),
        'num_basis_func': json_data.get('num_radial_basis', 8),
        'hidden_irreps': hidden_irreps,
        'nlayers': json_data.get('nlayers', 3),
        'features_dim': json_data.get('features_dim', 64),
        'output_irreps': json_data.get('output_channels', "1x0e"),
        'active_fn': json_data.get('active_fn', "identity"),
        'regress_forces': regress_forces,
    }


def parse_cueq_config(json_data):
    """Parse CuEquivariance configuration.

    Returns:
        CuEquivarianceConfig or None
    """
    cueq_config = json_data.get('cueq_config')
    if cueq_config is None or cueq_config:
        try:
            import cuequivariance as cue  # noqa: F401
            import cuequivariance_torch as cuet  # noqa: F401
            CUET_AVAILABLE = True
        except ImportError:
            CUET_AVAILABLE = False
        if CUET_AVAILABLE:
            from bam_torch.model.wrapper_ops import CuEquivarianceConfig
            return CuEquivarianceConfig(
                enabled=True,
                layout="ir_mul",
                group="O3_e3nn",
                optimize_all=True,
            )
    return None


def parse_charge_config(json_data):
    """Parse charge-dependent model configuration.

    Returns:
        dict with charge-specific constructor kwargs:
            cep_hidden_dim, charge_type, use_cent_energy
    """
    charge_config = json_data.get('charge', {})
    return {
        'cep_hidden_dim': charge_config.get('cep_hidden_dim', 64),
        'charge_type': charge_config.get('charge_type', 'npa'),
        'use_cent_energy': charge_config.get('use_cent_energy', False),
    }
