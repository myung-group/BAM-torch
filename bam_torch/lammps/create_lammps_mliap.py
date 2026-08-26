"""Export a RACE checkpoint for LAMMPS ``pair_style mliap unified``.

Usage:
    python -m bam_torch.lammps.create_lammps_mliap --pkl model.pkl \
        --backend oeq --output bam_mliap_oeq.pt

    # checkpoints trained on 0-based z-table species (omol/opoly datasets):
    python -m bam_torch.lammps.create_lammps_mliap --pkl model.pkl \
        --backend oeq --zbased --output bam_mliap_oeq.pt

    # or an explicit species-ordered element list:
    ... --elements Li P S Cl

    # multi-head checkpoint: pick which head to deploy
    ... --head target

The saved .pt stores a rebuild callable (not the module weights), so loading
requires this package on PYTHONPATH and, on torch>=2.6,
TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 in the LAMMPS environment.
"""
import argparse

import torch
from ase.data import chemical_symbols

from bam_torch.lammps.lammps_mliap_bam import rebuild_bam_mliap


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--pkl', default='model.pkl')
    p.add_argument('--backend', default='e3nn', choices=['e3nn', 'oeq'])
    p.add_argument('--output', default=None)
    p.add_argument('--elements', nargs='+', default=None,
                   help='chemical symbols in species order '
                        '(overrides checkpoint uniq_element)')
    p.add_argument('--head', default=None,
                   help='head name or index for multi-head checkpoints '
                        '(default: first head)')
    p.add_argument('--zbased', action='store_true',
                   help='species are 0-based z-table indices (H=0, He=1, ...) '
                        'as in omol/opoly-style datasets')
    args = p.parse_args()

    elements = args.elements
    if args.zbased:
        ck = torch.load(args.pkl, map_location='cpu', weights_only=False)
        n = ck['input.json']['num_species']
        elements = chemical_symbols[1:n + 1]

    head = args.head
    if head is not None and head.lstrip('-').isdigit():
        head = int(head)
    obj = rebuild_bam_mliap(args.pkl, backend=args.backend, elements=elements,
                            head=head)
    out = args.output or ('bam_mliap_%s.pt' % args.backend)
    torch.save(obj, out)
    print('saved:', out)
    print('elements[:8]:', obj.element_types[:8], '... rcutfac:', obj.rcutfac)


if __name__ == '__main__':
    main()
