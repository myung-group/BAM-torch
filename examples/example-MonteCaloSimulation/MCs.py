import heapq
import json
import math
import os
import pickle
import random
from dataclasses import dataclass
from multiprocessing import Process, Queue, set_start_method
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from ase import Atoms
from ase.io import read

from bam_torch.tase.base_calculator import RACECalculator

# Paths
# Set paths based on tutorial directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = SCRIPT_DIR  # tutorial directory is project root
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_energy(calc: RACECalculator, atoms: Atoms) -> float:
    atoms.calc = calc
    return float(atoms.get_potential_energy())


@dataclass
class MonteCarloConfig:
    temperature: float
    n_steps: int
    sampling_interval: int
    output_interval: int
    structure_save_interval: int = 5000


@dataclass
class SwapSettings:
    cutoff: float
    swaps_per_step: int


class TMSwapMonteCarlo:
    def __init__(
        self,
        structure_template: str,
        model_config: Dict,
        swap_settings: SwapSettings,
        seed: Optional[int] = None,
    ) -> None:
        self.seed = seed
        self.log_prefix = f"[Seed {seed}] " if seed else ""
        self.swap_settings = swap_settings

        # Load structure and identify atoms
        self.base_atoms = read(structure_template)
        self.tm_indices = self._identify_tm_atoms(self.base_atoms)

        # Initialize calculator
        self.calc = RACECalculator(model_config)
        if seed:
            set_random_seed(seed)  # Re-seed after calculator init
        
        # Tracking structures
        self.top_structures: List[Tuple[float, int, Atoms]] = []
        self.max_top_structures = 10
        self._last_swaps: List[Tuple[int, int]] = []

        # Initialize log file
        self.log_file = None
        if seed:
            os.makedirs(RESULTS_DIR, exist_ok=True)
            self.log_file = open(os.path.join(RESULTS_DIR, f"tm_swap_mc_seed_{seed}.log"), "w", buffering=1)

    @staticmethod
    def _identify_tm_atoms(atoms: Atoms) -> List[int]:
        tm_symbols = {
            "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
            "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd",
            "La", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
        }
        return [i for i, s in enumerate(atoms.get_chemical_symbols()) if s in tm_symbols]

    def _log(self, msg: str) -> None:
        """Print and write to log file."""
        print(msg, flush=True)
        if self.log_file:
            self.log_file.write(msg + "\n")
            self.log_file.flush()

    def _select_tm_pair(self, atoms: Atoms, banned: Optional[set] = None) -> Optional[Tuple[int, int, float]]:
        """Select a TM pair to swap within cutoff distance."""
        if len(self.tm_indices) < 2:
            return None

        candidates = self.tm_indices.copy()
        random.shuffle(candidates)
        banned_pairs = {tuple(sorted(pair)) for pair in self._last_swaps}
        if banned:
            banned_pairs.update(banned)

        for anchor in candidates:
            distances = atoms.get_distances(anchor, self.tm_indices, mic=True)
            neighbours = [
                self.tm_indices[i] for i, dist in enumerate(distances)
                if self.tm_indices[i] != anchor and 0.0 < dist <= self.swap_settings.cutoff
            ]
            
            random.shuffle(neighbours)
            for partner in neighbours:
                # Skip same element swap
                if self.base_atoms.get_chemical_symbols()[anchor] == self.base_atoms.get_chemical_symbols()[partner]:
                    continue
                key = tuple(sorted((anchor, partner)))
                if key not in banned_pairs:
                    return anchor, partner, float(atoms.get_distance(anchor, partner, mic=True))
        return None

    def _propose_swap(self, atoms: Atoms, banned: Optional[set] = None) -> Optional[Tuple[Atoms, Dict, List]]:
        """Propose TM swap(s)."""
        swaps = []
        proposal = atoms.copy()
        combined_banned = set(tuple(sorted(pair)) for pair in (banned or []))
        new_pairs = []

        for _ in range(self.swap_settings.swaps_per_step):
            pair = self._select_tm_pair(proposal, combined_banned)
            if not pair:
                break

            i, j, distance = pair
            positions = proposal.positions.copy()
            positions[[i, j]] = positions[[j, i]]
            proposal.positions = positions

            swaps.append({
                "type": "TM swap",
                "atom_indices": [int(i), int(j)],
                "elements": [self.base_atoms.get_chemical_symbols()[i], self.base_atoms.get_chemical_symbols()[j]],
                "distance": distance,
            })

            pair_key = tuple(sorted((i, j)))
            combined_banned.add(pair_key)
            new_pairs.append(pair_key)

        if not swaps:
            return None

        proposal.info["last_swap"] = swaps
        return proposal, {"tm_move": swaps}, new_pairs

    def run(self, mc_config: MonteCarloConfig) -> Dict:
        """Run Monte Carlo simulation."""
        beta = 1.0 / (8.617e-5 * mc_config.temperature) if mc_config.temperature > 0 else float("inf")
        
        # Initialize simulation state
        current_atoms = self.base_atoms.copy()
        current_energy = compute_energy(self.calc, current_atoms)
        best_energy = current_energy
        
        # Save initial structure
        self._update_top_structures(current_energy, 0, current_atoms.copy())

        # Statistics tracking
        total_steps = mc_config.n_steps
        accept_total = 0
        accept_sampled = 0
        proposals_sampled = 0
        energy_trace = []
        sampled_energies = []
        move_history = []

        # Main MC loop
        for step in range(total_steps):
            step_banned = set()
            step_moves = []
            step_accepted = False
            attempt_count = 0

            # Try proposing swaps until accepted or no more options
            while True:
                proposal = self._propose_swap(current_atoms, step_banned)
                
                if not proposal:
                    if not step_moves and step % mc_config.output_interval == 0:
                        self._log(f"{self.log_prefix}Step {step:6d}: no valid TM swap found")
                    break

                candidate_atoms, move_details, attempted_pairs = proposal
                candidate_energy = compute_energy(self.calc, candidate_atoms)
                delta_e = candidate_energy - current_energy

                # Metropolis acceptance criterion
                accept = delta_e <= 0.0 or (math.isfinite(beta) and math.log(random.random()) < -beta * delta_e)
                
                # Record move details
                final_energy = candidate_energy if accept else current_energy
                move_details["accepted"] = accept
                move_details["delta_e"] = delta_e
                move_details["energy"] = final_energy
                step_moves.append(move_details)

                # Log output
                if step % mc_config.output_interval == 0:
                    log_delta_e = 0 if accept else delta_e
                    self._log_step(step, attempt_count, accept, final_energy, best_energy, move_details, log_delta_e)

                if accept:
                    current_atoms = candidate_atoms
                    current_energy = candidate_energy
                    accept_total += 1
                    step_accepted = True
                    self._last_swaps = attempted_pairs
                    
                    if current_energy < best_energy:
                        best_energy = current_energy
                    
                    self._update_top_structures(current_energy, step, current_atoms.copy())
                    break
                
                # Reject: ban this pair and try another
                step_banned.update(attempted_pairs)
                self._last_swaps = attempted_pairs
                attempt_count += 1

            move_history.extend(step_moves)

            # Periodic structure save
            if mc_config.structure_save_interval > 0 and step > 0 and step % mc_config.structure_save_interval == 0:
                self._save_top_structures(step)

            # Collect statistics
            if True:  # Always collect statistics (equilibration removed)
                energy_trace.append(current_energy)
                if step % mc_config.sampling_interval == 0:
                    proposals_sampled += 1
                    if step_accepted:
                        accept_sampled += 1
                    sampled_energies.append(current_energy)

        # Finalize
        stats = {
            "best_energy": best_energy,
            "mean_energy": float(np.mean(sampled_energies)) if sampled_energies else current_energy,
            "std_energy": float(np.std(sampled_energies)) if sampled_energies else 0.0,
            "accept_ratio_total": accept_total / total_steps if total_steps else 0.0,
            "accept_ratio_sampled": accept_sampled / proposals_sampled if proposals_sampled else 0.0,
            "energy_trace": energy_trace,
            "move_history": move_history,
        }

        self._save_top_structures(total_steps - 1)
        self._log_summary(stats)
        
        if self.log_file:
            self.log_file.close()
            self.log_file = None

        return stats

    def _log_step(self, step: int, retry: int, accept: bool, energy: float, 
                  best_energy: float, move_details: Dict, delta_e: float) -> None:
        """Log a single step."""
        label = f"Step {step:6d}" if retry == 0 else f"       retry {retry:2d}"
        tm_info = move_details.get("tm_move", [])
        
        if tm_info:
            swaps_str = []
            for pair in tm_info:
                idx_a, idx_b = pair["atom_indices"]
                elem_a, elem_b = pair["elements"]
                swaps_str.append(f"{idx_a}({elem_a})↔{idx_b}({elem_b})")
            
            if len(tm_info) > 1:
                msg_detail = f"TM swap ({len(tm_info)}): {', '.join(swaps_str)}"
            else:
                pair = tm_info[0]
                msg_detail = f"TM swap {swaps_str[0]} (dist={pair['distance']:.3f} Å)"
        else:
            msg_detail = "No swap"
        
        if accept:
            tail = ""
            energy_str = f"E={energy:10.4f} eV"
        else:
            tail = f" | ΔE={delta_e:+.6e} eV"
            # User Request: Log candidate energy (swapped energy) even when rejected
            candidate_energy = energy + delta_e
            energy_str = f"E={candidate_energy:10.4f} eV"

        msg = (f"{self.log_prefix}{label}: {'accepted' if accept else 'rejected':<9} "
               f"{energy_str} | best={best_energy:10.4f} eV | {msg_detail}{tail}")
        self._log(msg)

    def _log_summary(self, stats: Dict) -> None:
        """Log final summary."""
        if self.log_file:
            summary = (
                f"\n{'=' * 70}\n"
                f"Seed {self.seed} Completed\n"
                f"Best energy: {stats['best_energy']:.6f} eV\n"
                f"Mean energy: {stats['mean_energy']:.6f} eV\n"
                f"Accept ratio (total): {stats['accept_ratio_total']:.4f}\n"
                f"Accept ratio (sampled): {stats['accept_ratio_sampled']:.4f}\n"
                f"{'=' * 70}\n"
            )
            self.log_file.write(summary)

    def _update_top_structures(self, energy: float, step: int, atoms: Atoms) -> None:
        """Update top 10 structures list."""
        entry = (-energy, step, atoms)
        if len(self.top_structures) < self.max_top_structures:
            heapq.heappush(self.top_structures, entry)
        elif energy < -self.top_structures[0][0]:
            heapq.heapreplace(self.top_structures, entry)

    def _save_top_structures(self, step: int) -> None:
        """Save top 10 structures to files."""
        if not self.top_structures:
            return

        from ase.calculators.singlepoint import SinglePointCalculator
        from ase.io import write

        sorted_structures = sorted(self.top_structures, key=lambda x: x[0])
        step_dir = os.path.join(RESULTS_DIR, f"step_{step:06d}")
        os.makedirs(step_dir, exist_ok=True)

        for rank, (neg_energy, struct_step, atoms) in enumerate(sorted_structures):
            energy = -neg_energy
            filename = os.path.join(
                step_dir,
                f"tm_swap_seed_{self.seed}_rank_{rank+1:02d}_step_{struct_step:06d}_energy_{energy:.4f}.extxyz"
            )
            atoms = atoms.copy()
            atoms.set_calculator(SinglePointCalculator(atoms, energy=energy))
            atoms.info['rank'] = rank + 1
            atoms.info['mc_step'] = struct_step
            write(filename, atoms, format="extxyz")

        self._log(f"{self.log_prefix}💾 Top {len(sorted_structures)} structures saved (step {step})")


def run_seed_on_gpu(seed: int, gpu_idx: int, structure_path: str, model_config: Dict,
                    swap_settings: SwapSettings, mc_config: MonteCarloConfig, result_queue: Queue) -> None:
    """Run a seed on a specific GPU.

    Note: With spawn start method, this function runs in a fresh Python interpreter,
    so setting CUDA_VISIBLE_DEVICES here will work properly.

    gpu_idx should be the actual physical GPU ID (e.g., 0, 1, 2).
    We set CUDA_VISIBLE_DEVICES to only show this one GPU, and PyTorch will see it as device 0.
    """
    import sys
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)

    # Set CUDA_VISIBLE_DEVICES to single GPU for this process
    # This MUST be done before any CUDA initialization
    # After this, PyTorch will see this GPU as cuda:0
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_idx)

    # Also explicitly clear any GPU selection from parent process
    if 'CUDA_DEVICE' in os.environ:
        del os.environ['CUDA_DEVICE']

    print(f"\n{'=' * 70}\n[Physical GPU {gpu_idx}] Seed {seed} Start (will appear as cuda:0)\n{'=' * 70}\n", flush=True)

    # Debug: Check what CUDA sees
    if torch.cuda.is_available():
        print(f"CUDA available: {torch.cuda.device_count()} device(s)")
        print(f"Current device: {torch.cuda.current_device()}")
        print(f"Device name: {torch.cuda.get_device_name(0)}")
    else:
        print("WARNING: CUDA not available!")

    set_random_seed(seed)
    mc = TMSwapMonteCarlo(structure_path, model_config, swap_settings, seed=seed)
    results = mc.run(mc_config)
    print(f"\n[GPU {gpu_idx}] Seed {seed} Completed", flush=True)
    result_queue.put((seed, results))


def main() -> None:
    # Load configuration
    input_json = os.path.join(PROJECT_ROOT, "input.json")
    with open(input_json, "r") as f:
        config = json.load(f)

    mc_cfg = config["monte_carlo"]
    flex_cfg = mc_cfg.get("flexible_config", {}) or {}

    # Setup configurations
    swap_settings = SwapSettings(
        cutoff=float(flex_cfg.get("tm_swap_cutoff", config.get("cutoff", 6.0))),
        swaps_per_step=int(flex_cfg.get("tm_swaps_per_step", 1)),
    )
    
    mc_config = MonteCarloConfig(
        temperature=float(mc_cfg["temperature"]),
        n_steps=int(mc_cfg["n_steps"]),
        sampling_interval=int(mc_cfg["sampling_interval"]),
        output_interval=int(mc_cfg.get("output_interval", 100)),
        structure_save_interval=int(mc_cfg.get("structure_save_interval", 5000)),
    )

    seeds = [
        config.get("NN", {}).get("data_seed", 1300) + i for i in range(3)
    ]

    # Handle structure_template path: absolute or relative to data/ folder
    structure_template = mc_cfg["structure_template"]
    if os.path.isabs(structure_template):
        structure_path = structure_template
    else:
        structure_path = os.path.join(PROJECT_ROOT, "data", structure_template)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print(f"    Structure : {structure_path}")
    print(f"    Cutoff    : {swap_settings.cutoff} Å")
    print(f"    Swaps/step: {swap_settings.swaps_per_step}")

    # MultiGPU or sequential execution
    use_multigpu = os.environ.get("USE_MULTIGPU", "false").lower() == "true"
    gpu_ids = [g.strip() for g in os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")]

    if use_multigpu and len(gpu_ids) >= len(seeds):
        print(f"🚀 MultiGPU mode: {len(seeds)} seeds on {len(gpu_ids)} GPUs")

        # Use subprocess instead of multiprocessing to properly set CUDA_VISIBLE_DEVICES
        import subprocess
        import sys
        import tempfile

        processes = []
        temp_files = []

        for idx, seed in enumerate(seeds):
            gpu_idx = int(gpu_ids[idx % len(gpu_ids)])

            # Create a temporary file to store results
            temp_file = tempfile.NamedTemporaryFile(mode='w+b', delete=False, suffix='.pkl')
            temp_file.close()
            temp_files.append(temp_file.name)

            # Create a runner script that will be executed with specific CUDA_VISIBLE_DEVICES
            runner_code = f'''
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "{gpu_idx}"

import sys
sys.path.insert(0, "{PROJECT_ROOT}/src")

import pickle
from test_1027 import TMSwapMonteCarlo, SwapSettings, MonteCarloConfig, set_random_seed

# Load config
import json
with open("{input_json}", "r") as f:
    config = json.load(f)

mc_cfg = config["monte_carlo"]
flex_cfg = mc_cfg.get("flexible_config", {{}}) or {{}}

swap_settings = SwapSettings(
    cutoff=float(flex_cfg.get("tm_swap_cutoff", config.get("cutoff", 6.0))),
    swaps_per_step=int(flex_cfg.get("tm_swaps_per_step", 1)),
)

mc_config = MonteCarloConfig(
    temperature=float(mc_cfg["temperature"]),
    n_steps=int(mc_cfg["n_steps"]),
    sampling_interval=int(mc_cfg["sampling_interval"]),
    output_interval=int(mc_cfg.get("output_interval", 100)),
    structure_save_interval=int(mc_cfg.get("structure_save_interval", 5000)),
)

print(f"\\n{{'=' * 70}}\\n[Physical GPU {gpu_idx}] Seed {seed} Start\\n{{'=' * 70}}\\n", flush=True)

set_random_seed({seed})
mc = TMSwapMonteCarlo("{structure_path}", config, swap_settings, seed={seed})
results = mc.run(mc_config)

# Save results to temp file
with open("{temp_file.name}", "wb") as f:
    pickle.dump({{"seed": {seed}, "results": results}}, f)

print(f"\\n[GPU {gpu_idx}] Seed {seed} Completed", flush=True)
'''

            # Launch subprocess with specific GPU
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_idx)

            p = subprocess.Popen(
                [sys.executable, "-c", runner_code],
                env=env,
                stdout=sys.stdout,
                stderr=sys.stderr
            )
            processes.append(p)
            print(f"  Seed {seed} → GPU {gpu_idx} (PID: {p.pid})")

        # Wait for all processes
        for p in processes:
            p.wait()

        # Collect results from temp files
        all_results = {}
        for temp_file in temp_files:
            with open(temp_file, "rb") as f:
                data = pickle.load(f)
                all_results[f"seed_{data['seed']}"] = data['results']
            os.unlink(temp_file)

    else:
        if use_multigpu:
            print(f"⚠️  MultiGPU requested but insufficient GPUs, running sequentially")
        
        all_results = {}
        for seed in seeds:
            print(f"\n{'=' * 70}\nSeed {seed}\n{'=' * 70}", flush=True)
            set_random_seed(seed)
            mc = TMSwapMonteCarlo(structure_path, config, swap_settings, seed=seed)
            results = mc.run(mc_config)
            all_results[f"seed_{seed}"] = results

    # Save results
    results_path = os.path.join(RESULTS_DIR, "tm_swap_mc_results.pkl")
    with open(results_path, "wb") as f:
        pickle.dump(all_results, f)
    print(f"\n✅ Monte Carlo sweep completed. Results saved to {results_path}")


if __name__ == "__main__":
    set_start_method('spawn', force=True)
    main()
