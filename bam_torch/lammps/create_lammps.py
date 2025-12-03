import os
os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "29500"
os.environ["RANK"] = "0"
os.environ["WORLD_SIZE"] = "1"

import ast
import json
import torch
import torch.distributed as dist
from e3nn.util import jit

from bam_torch.model import models as bam_models
from bam_torch.model.models import get_edge_relative_vectors_with_pbc_lammps
from bam_torch.utils.utils import find_input_json, extract_species
from bam_torch.lammps.lammps_bam import LAMMPS_BAM


with open('input.json') as f:
    cfg = json.load(f)

def main():
    # Initiate distributed process
    if not dist.is_initialized():
        dist.init_process_group(backend='nccl' if torch.cuda.is_available() else 'gloo',
                              init_method='env://',
                              world_size=1,
                              rank=0)

    input_json_path = find_input_json()
    with open(input_json_path) as f:
         json_data = json.load(f)
    model_path = json_data.get("model_path", "model.pt")
    pckl = torch.load('model.pkl', weights_only=False)
    nlayers = pckl['input.json']['nlayers']
    cutoff = pckl['input.json']['cutoff']
    
    if cfg.get('enr_avg_per_element') is not None: # for universal potential
        with open(cfg['enr_avg_per_element'], 'r', encoding='utf-8') as file:
            content = file.read()
        enr_avg_per_element, uniq_element = ast.literal_eval(content)
    else: # for general case
        uniq_element = pckl['uniq_element']
        enr_avg_per_element = pckl['enr_avg_per_element']

    e_corr = torch.tensor(pckl['valid_scale_shift'])
    e_corr = e_corr.flatten().mean().item()

    print(f"enr_avg_per_element: {enr_avg_per_element}")
    print(f"e_corr: {e_corr}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load model
    model = torch.load("model.pt", map_location='cpu', weights_only=False)

    # Remove DDP wrapper
    if hasattr(model, 'module'):
        model = model.module

    model.eval()
    species = torch.tensor(list(uniq_element.keys()))
    model.atomic_numbers = species.clone().detach()
    model.num_interactions = torch.tensor(nlayers)
    model.r_max = torch.tensor(cutoff)
    model = model.float().to(device)
    bam_models.get_edge_relative_vectors_with_pbc = get_edge_relative_vectors_with_pbc_lammps
    model.training_mode_for_lammps = True

    # Adjust variables for compatibility across model versions
    try:
        criterion = model.criterion
        criterion_value = model.criterion_value
        if criterion < criterion_value:
            regress_forces = "auto"
        else:
            regress_forces = "direct"
        model.regress_forces = regress_forces
    except:
        criterion = None
        model.criterion = criterion

    for module in model.modules():
        module.training_mode_for_lammps = True

    # Generate LAMMPS-BAM model
    lammps_model = LAMMPS_BAM(
        model,
        enr_avg_per_element=enr_avg_per_element,
        e_corr=e_corr
    )
    lammps_model = lammps_model.to(device)
    lammps_model_compiled = jit.compile(lammps_model)
    lammps_model_compiled.save(f"{model_path.split('.')[0]}" + "-lammps.pt")

    print("LAMMPS model created successfully with node_enr_avg + e_corr!")

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

