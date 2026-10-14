"""Train-split-only reference energies, scales and immutable provenance for MVE."""

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
from ase.stress import full_3x3_to_voigt_6_stress

from bam_torch.utils.utils import get_enr_avg_per_element
from .loss import MVE_UNITS, stress_voigt_to_irrep

SCHEMA = 'bam_torch.mve_train_stats.v1'


def stats_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def geometry_digest(atoms):
    return stats_digest({'numbers': [int(z) for z in atoms.numbers],
                         'positions': np.asarray(atoms.positions).tolist(),
                         'cell': np.asarray(atoms.cell).tolist(),
                         'pbc': np.asarray(atoms.pbc, dtype=bool).tolist()})


def _original_labels(atoms):
    results = atoms.calc.results if atoms.calc is not None else {}
    virial = results.get('virial')
    if virial is None:
        virial = atoms.info.get('virial')
    labels = {}
    for key in ('energy', 'forces', 'stress'):
        value = results.get(key)
        if value is not None:
            labels[key] = np.asarray(value, dtype=float)
    if 'stress' not in labels and virial is not None:
        volume = float(abs(np.linalg.det(np.asarray(atoms.cell))))
        if not math.isfinite(volume) or volume <= 0:
            raise ValueError('observed virial requires positive original cell volume')
        value = np.asarray(virial, dtype=float)
        labels['stress'] = - (full_3x3_to_voigt_6_stress(value) if value.shape == (3, 3) else value) / volume
    for key, value in labels.items():
        if not np.isfinite(value).all():
            raise ValueError(f'observed {key} must be finite before train-only fitting')
    return labels


def train_provenance(frames, ids):
    if len(frames) != len(ids) or len(set(ids)) != len(ids):
        raise ValueError('training frame IDs must be unique and aligned')
    entries = []
    for atoms, frame_id in zip(frames, ids):
        labels = _original_labels(atoms)
        entries.append({'id': int(frame_id), 'numbers': [int(z) for z in atoms.numbers],
                        'positions': np.asarray(atoms.positions).tolist(),
                        'cell': np.asarray(atoms.cell).tolist(),
                        'pbc': np.asarray(atoms.pbc, dtype=bool).tolist(),
                        'labels': {key: value.tolist() for key, value in labels.items()}})
    return {'split_digest': stats_digest([int(frame_id) for frame_id in ids]),
            'source_digest': stats_digest(entries)}


def fit_train_stats(frames, ids, enabled=False, external=None):
    if type(enabled) is not bool:
        raise ValueError('train normalization enabled must be boolean')
    provenance = train_provenance(frames, ids)
    elements = sorted({int(z) for atoms in frames for z in atoms.numbers})
    energy_frames = [atoms for atoms in frames if 'energy' in _original_labels(atoms)]
    if not energy_frames:
        raise ValueError('train split needs an original energy label to fit atomic references')
    refs, species, _ = get_enr_avg_per_element(energy_frames, elements)
    references = {int(k): float(v) for k, v in refs.items()}
    species = {int(k): int(v) for k, v in species.items()}
    for value in references.values():
        if not math.isfinite(value):
            raise ValueError('atomic references must be finite')
    q = stress_voigt_to_irrep(dtype=torch.float64).numpy()
    residuals = {'energy': [], 'forces': [], 'stress': []}
    for atoms in frames:
        labels = _original_labels(atoms)
        if 'energy' in labels:
            ref = sum(references[species[int(z)]] for z in atoms.numbers)
            residuals['energy'].append((float(labels['energy']) - ref) / len(atoms))
        if 'forces' in labels:
            value = labels['forces']
            if value.shape != (len(atoms), 3):
                raise ValueError('original forces must be [atoms,3]')
            residuals['forces'].extend(value.reshape(-1).tolist())
        if 'stress' in labels:
            value = labels['stress']
            if value.shape != (6,):
                raise ValueError('original stress must be ASE Voigt[6]')
            residuals['stress'].extend((q @ value).tolist())
    scales = {}
    for key, values in residuals.items():
        # A missing task has no fitted variance; unit scale leaves it inert.
        scale = float(np.sqrt(np.mean(np.square(values)))) if values else 1.0
        scales[key] = max(scale, 1e-12)
    computed = {'schema': SCHEMA, 'version': 1, 'units': MVE_UNITS,
                'normalization_enabled': enabled,
                **provenance, 'species': species, 'references': references, 'scales': scales}
    if external is not None:
        supplied = json.loads(Path(external).read_text())
        # JSON object keys are strings; the in-memory species/reference maps
        # use integer atomic numbers and model indices for graph construction.
        canonical = json.loads(json.dumps(computed, allow_nan=False))
        if supplied != canonical:
            raise ValueError('external MVE statistics schema/units/train split/source/reference/scales mismatch')
        return computed
    return computed
