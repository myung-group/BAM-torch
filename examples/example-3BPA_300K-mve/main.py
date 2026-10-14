import torch
import json

from bam_torch.training import TRAINER_REGISTRY
from bam_torch.utils.utils import find_input_json, date


if __name__ == '__main__':
    print(date()) 
    input_json_path = find_input_json()
    with open(input_json_path) as f:
        json_data = json.load(f)
    trainer = TRAINER_REGISTRY[json_data['trainer']](json_data, 0, 1)
    trainer.train()
    
    print(date())
