from bam_torch.predicting.evaluator import Evaluator
from bam_torch.utils.utils import date
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MVE v1 CPU checkpoint prediction')
    parser.add_argument('checkpoint')
    parser.add_argument('prediction_file', help='ASE-readable geometries; labels optional')
    parser.add_argument('--output', default='mve_predictions.json')
    args = parser.parse_args()
    print(date())
    evaluator = Evaluator({'trainer': 'mve', 'predict': {
        'model': args.checkpoint, 'fname_traj': args.prediction_file,
        'uncertainty_output': args.output,
    }})
    evaluator.evaluate()
    print(date())
