"""Entry point for distillation training.

Defaults read settings from ``input.json``. CLI flags override individual
fields so a λ-ablation is three one-line invocations::

    python main.py --lambda-dft 1.0      # pure DFT (compression baseline)
    python main.py --lambda-dft 0.5      # hybrid
    python main.py --lambda-dft 0.0      # pure teacher (distillation)

A ``--lambda-dft X`` invocation automatically reroutes outputs into
``runs/ldft_X/`` so the three ablation runs don't clobber each other.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", default=os.path.join(HERE, "input.json"))
    p.add_argument("--lambda-dft", type=float, default=None,
                   help="override NN.distill.lambda_dft and reroute outputs into runs/ldft_X/")
    p.add_argument("--nepoch", type=int, default=None, help="override NN.nepoch")
    p.add_argument("--run-name", default=None,
                   help="explicit subdirectory under runs/ for outputs (overrides the auto-name)")
    return p.parse_args()


def reroute_outputs(json_data: dict, run_name: str) -> None:
    run_dir = os.path.join(HERE, "runs", run_name)
    os.makedirs(run_dir, exist_ok=True)
    json_data["NN"]["fname_pkl"] = os.path.join(run_dir, "student_runtime.pkl")
    json_data["train"] = {"fname_log": os.path.join(run_dir, "loss_train.out")}
    print(f"[main] outputs rerouted to {run_dir}")


def main() -> None:
    args = parse_args()
    if not os.path.exists(args.input):
        sys.exit(f"input.json not found: {args.input}")

    with open(args.input) as f:
        json_data = json.load(f)

    if args.lambda_dft is not None:
        json_data["NN"]["distill"]["lambda_dft"] = args.lambda_dft

    if args.nepoch is not None:
        json_data["NN"]["nepoch"] = args.nepoch

    if args.run_name is not None:
        reroute_outputs(json_data, args.run_name)
    elif args.lambda_dft is not None:
        # Auto-name from lambda so the 3 ablation runs separate cleanly.
        # Use Python's default float repr (e.g. 0.0, 0.5, 1.0) so the dir
        # name matches the literal string passed by run_ablation.sh.
        reroute_outputs(json_data, f"ldft_{args.lambda_dft}")

    # DistillTrainer is registered as 'distill' in TRAINER_REGISTRY by
    # bam_torch/training/__init__.py once bam_torch is installed.
    from bam_torch.training import TRAINER_REGISTRY

    trainer_name = json_data.get("trainer", "distill")
    if trainer_name not in ("distill", "distillation"):
        sys.exit(f"this script only supports trainer='distill', got '{trainer_name}'")
    trainer_cls = TRAINER_REGISTRY[trainer_name]

    if json_data.get("gpu-parallel"):
        print("[main] forcing gpu-parallel=false for phase-1 single-GPU run")
        json_data["gpu-parallel"] = False

    print(f"[main] lambda_dft = {json_data['NN']['distill']['lambda_dft']}")
    print(f"[main] nepoch     = {json_data['NN']['nepoch']}")
    print(f"[main] ckpt       = {json_data['NN']['fname_pkl']}")

    trainer = trainer_cls(json_data, rank=0, world_size=1)
    trainer.train()
    print("[main] training finished")


if __name__ == "__main__":
    main()
