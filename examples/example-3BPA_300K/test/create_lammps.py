import argparse
import os

os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"

from bam_torch.utils.utils import find_input_json, date
import json
import torch
from e3nn.util import jit

from lammps_bam import LAMMPS_BAM
from bam_torch.training.base_trainer import BaseTrainer
from bam_torch.model.models import RACE
import pickle
from bam_torch.utils.utils import extract_species

def main():
    input_json_path = find_input_json()
    with open(input_json_path) as f:
         json_data = json.load(f)

    # JSON에서 필요한 정보 추출
    model_path = json_data.get("model_path", "model.pt")
    head = json_data.get("head", None)

    pckl = torch.load('model.pkl', weights_only=False)
    nlayers = pckl['input.json']['nlayers']
    cutoff = pckl['input.json']['cutoff']

    model = torch.load("model.pt", weights_only=False)
    model.eval()
    
    species = extract_species("train_300K.xyz")
    model.atomic_numbers = species.clone().detach()
    model.num_species = len(species)
    model.num_interactions = torch.tensor(nlayers)
    model.r_max = torch.tensor(cutoff)
    print(f"atomic_numbers: {model.atomic_numbers}")
    print(f"r_max: {model.r_max}")
    print(f"num_interactions: {model.num_interactions}")
    print(f'species: {species}')

    # 데이터 타입을 float32로 강제 설정
    print("Converting model to float32 for LAMMPS compatibility.")
    model = model.float().to("cpu")

    # 헤드 처리
    if head is None and hasattr(model, "heads") and len(model.heads) > 1:
        print("Multiple heads detected, but no head specified in JSON. Using the last head.")
        head = model.heads[-1]
    if head is not None:
        print(f"Using head: {head}")

    lammps_model = (
        LAMMPS_BAM(model, head=head) if head is not None else LAMMPS_BAM(model)
    )
    lammps_model_compiled = jit.compile(lammps_model)
    lammps_model_compiled.save(model_path + "-lammps.pt")

if __name__ == "__main__":
    main()
