from bam_torch.predicting.evaluator import Evaluator
from bam_torch.utils import find_input_json, date
import json
import torch


if __name__ == '__main__':
    print(date()) 
    input_json_path = find_input_json()
    torch.cuda.empty_cache()

    with open(input_json_path) as f:
        json_data = json.load(f)

        if json_data['predict']:
            evaluator = Evaluator(json_data)
            evaluator.evaluate()
        else:
            print('we are making')

    print(date())
