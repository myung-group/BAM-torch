"""
Convert CG model pkl to pt format.

Usage:
    python -m bam_torch.lammps.make_pt_cg --pkl model_cg.pkl --pt model_cg.pt

Or in Python:
    from bam_torch.lammps.make_pt_cg import recreate_cg_model_pt_from_pkl
    recreate_cg_model_pt_from_pkl('model_cg.pkl', 'model_cg.pt')
"""

import torch
from e3nn import o3
from copy import deepcopy
import argparse

from bam_torch.model.models import RACE


def recreate_cg_model_pt_from_pkl(pkl_path='model_cg.pkl', output_path='model_cg.pt'):
    """Generate model.pt from CG model.pkl

    Args:
        pkl_path: Path to the CG model pkl file
        output_path: Path to save the pt file

    Returns:
        model: The loaded RACE model
    """
    # Load checkpoint
    pckl = torch.load(pkl_path, map_location='cpu', weights_only=False)
    cfg = pckl['input.json']

    # Verify this is a CG model
    is_cg = pckl.get('is_cg_model', False)
    if not is_cg:
        print("Warning: This doesn't appear to be a CG model. Proceeding anyway...")

    # Parameter to choose force computation mode
    regress_forces = cfg.get('regress_forces', 'direct')
    if regress_forces == True:
        regress_forces = "direct"
    elif regress_forces == False:
        regress_forces = "false"

    # CG uses num_cg_types instead of num_species
    num_species = cfg.get('num_cg_types', cfg.get('num_species', 1))

    # Multihead finetune ckpt (cg_multihead trainer): rebuild RACEUnified with
    # the SAME heads. Inference uses head 0 (= datasets[0], the target head) via
    # the model fallback when data has no "head" key — so LAMMPS wrappers need
    # no change, but datasets[0] MUST be the deployment target by convention.
    mh = cfg.get("multihead", {})
    if mh.get("enabled", False):
        from bam_torch.model.models import RACEUnified
        heads = [d.get("name", f"head_{i}") for i, d in enumerate(mh.get("datasets", []))]
        assert not cfg.get("use_bond_flag", False), "RACEUnified path does not support use_bond_flag"
        regress = cfg.get("regress_forces", "auto")
        if regress is True: regress = "autograd"
        elif regress is False: regress = "false"
        model = RACEUnified(
            cutoff=cfg["cutoff"],
            avg_num_neighbors=cfg["avg_num_neighbors"],
            num_species=num_species,
            max_ell=cfg["max_ell"],
            num_basis_func=cfg["num_radial_basis"],
            hidden_irreps=o3.Irreps(cfg["hidden_channels"]),
            nlayers=cfg["nlayers"],
            features_dim=cfg["features_dim"],
            output_irreps=o3.Irreps(cfg.get("output_channels", "1x0e")),
            active_fn=cfg.get("active_fn", "identity"),
            regress_forces=regress,
            cueq_config=None,
            interaction_block=cfg.get("interaction_block") or "slow",
            heads=heads,
        )
        state_dict = pckl["params"]
        if any(k.startswith("module.") for k in state_dict.keys()):
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)
        model.eval()
        torch.save(deepcopy(model), output_path)
        print(f"✓ Multihead CG model -> {output_path} | heads {heads} | inference head 0 = {heads[0]}")
        return model

    # Create model with exact config
    model = RACE(
        cutoff=cfg['cutoff'],
        avg_num_neighbors=cfg['avg_num_neighbors'],
        num_species=num_species,
        max_ell=cfg['max_ell'],
        num_basis_func=cfg['num_radial_basis'],
        hidden_irreps=o3.Irreps(cfg['hidden_channels']),
        nlayers=cfg['nlayers'],
        features_dim=cfg['features_dim'],
        output_irreps=o3.Irreps(cfg.get('output_channels', '1x0e')),
        active_fn=cfg.get('active_fn', 'identity'),
        regress_forces=regress_forces,
        use_bond_flag=cfg.get('use_bond_flag', False),
        # 훈련에 쓴 블록을 그대로 재현해야 한다. 빠뜨리면 fast 로 훈련한 모델을
        # slow 구조에 얹게 되어 load_state_dict 가 깨진다.
        interaction_block=cfg.get('interaction_block', 'slow'),
    )

    # Load weights (remove DDP prefix if exists)
    state_dict = pckl['params']
    if any(k.startswith('module.') for k in state_dict.keys()):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

    model.load_state_dict(state_dict)
    model.eval()

    # Save model
    torch.save(deepcopy(model), output_path)

    print(f"✓ Successfully recreated {output_path} from {pkl_path}")
    print(f"  Model type: RACE (CG)")
    print(f"  Number of CG types: {num_species}")
    print(f"  Cutoff: {cfg['cutoff']} Å")
    print(f"  Total params: {sum(p.numel() for p in model.parameters()):,}")

    # Print CG info if available
    cg_mapping = pckl.get('cg_mapping_config', {})
    if cg_mapping:
        print(f"  CG mapping: {cg_mapping.get('formula', 'N/A')} -> {len(cg_mapping.get('beads', []))} bead(s)")

    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert CG model pkl to pt format')
    parser.add_argument('--pkl', type=str, default='model_cg.pkl',
                        help='Path to the CG model pkl file (default: model_cg.pkl)')
    parser.add_argument('--pt', type=str, default='model_cg.pt',
                        help='Path to save the pt file (default: model_cg.pt)')
    args = parser.parse_args()

    recreate_cg_model_pt_from_pkl(args.pkl, args.pt)
