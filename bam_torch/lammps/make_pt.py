import torch
from bam_torch.model.models import RACE
from e3nn import o3
from copy import deepcopy

def recreate_model_pt_from_pkl(pkl_path='model.pkl', output_path='model.pt'):
    """Generate model.pt from model.pkl
    """
    # Load checkpoint
    pckl = torch.load(pkl_path, map_location='cpu', weights_only=False)
    cfg = pckl['input.json']
    
    # Parameter to choose force computation mode:
    regress_forces = cfg.get('regress_forces', 'auto')
    if regress_forces == True:    # forces computed via auto-gradient of energy
        regress_forces = "auto"
    elif regress_forces == False: # no force computation 
        regress_forces = "false" 
        
    # Create model with exact config
    model = RACE(
        cutoff=cfg['cutoff'],
        avg_num_neighbors=cfg['avg_num_neighbors'],
        num_species=cfg['num_species'],
        max_ell=cfg['max_ell'],
        num_basis_func=cfg['num_radial_basis'],
        hidden_irreps=o3.Irreps(cfg['hidden_channels']),
        nlayers=cfg['nlayers'],
        features_dim=cfg['features_dim'],
        output_irreps=o3.Irreps(cfg['output_channels']),
        active_fn=cfg.get('active_fn', 'identity'),
        regress_forces=regress_forces,
    )
    
    # Load weights (remove DDP prefix if exists)
    state_dict = pckl['params']
    if any(k.startswith('module.') for k in state_dict.keys()):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    
    model.load_state_dict(state_dict)
    model.eval()
    
    # Save exactly as in training: deepcopy + save
    # This matches: torch.save(deepcopy(self.model), 'model.pt')
    torch.save(deepcopy(model), output_path)
    
    print(f"✓ Successfully recreated {output_path} from {pkl_path}")
    print(f"  Model type: {type(model)}")
    print(f"  Total params: {sum(p.numel() for p in model.parameters()):,}")
    
    return model

if __name__ == '__main__':
    recreate_model_pt_from_pkl('model.pkl', 'model.pt')
