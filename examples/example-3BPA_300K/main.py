from bam_torch.utils.utils import find_input_json, date
import json
import torch

if __name__ == '__main__':
    print(date()) 
    input_json_path = find_input_json()
    torch.cuda.empty_cache()

    with open(input_json_path) as f:
        json_data = json.load(f)

        if json_data['trainer'] in ['base']:
            from bam_torch.training.base_trainer import BaseTrainer
            base_trainer = BaseTrainer(json_data)
            base_trainer.train()
        else:
            print('we are making')

    print(date())
