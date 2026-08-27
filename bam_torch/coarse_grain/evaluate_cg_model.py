"""
CG Model Evaluation Script

Compares CG model predictions with reference values:
- Energy: CG predicted vs Atomistic reference
- Force: CG predicted vs CG-mapped atomistic forces

Supports two input modes:
- traj: Atomistic trajectory + CG mapping (original workflow)
- npz: Pre-processed CG NPZ file (no mapping needed)

Usage:
    # From atomistic trajectory (water)
    python evaluate_cg_model.py --model model_cg.pkl --data water.traj --ntest 100

    # From atomistic trajectory (custom mapping)
    python evaluate_cg_model.py --model model_cg.pkl --data system.traj --mapping water --ntest 100

    # From pre-processed NPZ (any system)
    python evaluate_cg_model.py --model model_cg.pkl --data system_cg.npz --input-type npz --ntest 100
"""

import os
import sys
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt

# Add parent path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from bam_torch.utils.utils import get_graphset_cg, get_cg_enr_avg_per_type
from torch_geometric.loader import DataLoader


def load_model(model_path, device):
    """Load trained CG model from checkpoint."""
    ckpt = torch.load(model_path, map_location=device, weights_only=False)

    # Get model config from checkpoint
    json_data = ckpt.get('input.json', {})

    # Import and create model
    from bam_torch.model import MODEL_REGISTRY
    from e3nn import o3

    model_name = json_data.get('model', 'race').lower()
    model_cls = MODEL_REGISTRY.get(model_name)

    num_cg_types = json_data.get('num_cg_types', 1)
    cutoff = json_data.get('cutoff', 10.0)
    hidden_irreps = o3.Irreps(json_data.get('hidden_channels', "64x0e+32x1o+16x2e"))

    model = model_cls(
        cutoff=cutoff,
        avg_num_neighbors=json_data.get('avg_num_neighbors', 50),
        num_species=num_cg_types,
        max_ell=json_data.get('max_ell', 2),
        num_basis_func=json_data.get('num_radial_basis', 8),
        hidden_irreps=hidden_irreps,
        nlayers=json_data.get('nlayers', 3),
        features_dim=json_data.get('features_dim', 64),
        output_irreps=json_data.get('output_channels', "1x0e"),
        active_fn=json_data.get('active_fn', "identity"),
        regress_forces=json_data.get('regress_forces', "direct"),
        cueq_config=None,
        use_bond_flag=json_data.get('use_bond_flag', False),
        interaction_block=json_data.get('interaction_block') or 'slow',
    )

    model.load_state_dict(ckpt['params'])
    model.set_criterion(None, None)  # No loss computation needed for evaluation
    model.to(device)
    model.eval()

    return model, ckpt


def evaluate(model, dataloader, device, enr_avg_per_type, uniq_type):
    """Run evaluation and collect predictions vs references."""
    model.eval()

    all_energy_pred = []
    all_energy_ref = []
    all_force_pred = []
    all_force_ref = []

    for data in dataloader:
            # Move to device
            data = {k: v.to(device) if hasattr(v, 'to') else v
                    for k, v in data.to_dict().items()}

            # Model prediction (need grad for F=-dE/dr when regress_forces="auto")
            preds = model(data, backprop=False)

            # Energy (add back the offset)
            energy_pred = preds['energy'].detach().cpu().numpy()
            energy_ref = data['energy'].detach().cpu().numpy()

            # Add back energy offset for comparison
            batch = data['batch'].cpu().numpy()
            species = data['species'].cpu().numpy()

            num_graphs = data['ptr'].numel() - 1
            for g in range(num_graphs):
                mask = batch == g
                types_in_graph = species[mask]
                offset = sum(enr_avg_per_type[uniq_type[t]] for t in types_in_graph)
                energy_pred[g] += offset
                energy_ref[g] += offset

            all_energy_pred.extend(energy_pred)
            all_energy_ref.extend(energy_ref)

            # Forces
            force_pred = preds['forces'].detach().cpu().numpy()
            force_ref = data['forces'].detach().cpu().numpy()

            all_force_pred.extend(force_pred.flatten())
            all_force_ref.extend(force_ref.flatten())

    return (np.array(all_energy_pred), np.array(all_energy_ref),
            np.array(all_force_pred), np.array(all_force_ref))


def calculate_metrics(pred, ref):
    """Calculate RMSE, MAE, and R^2."""
    rmse = np.sqrt(np.mean((pred - ref) ** 2))
    mae = np.mean(np.abs(pred - ref))

    ss_res = np.sum((ref - pred) ** 2)
    ss_tot = np.sum((ref - np.mean(ref)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

    return rmse, mae, r2


def plot_scatter(pred, ref, title, xlabel, ylabel, save_path, unit=""):
    """Create scatter plot with diagonal line."""
    fig, ax = plt.subplots(figsize=(6, 6))

    ax.scatter(ref, pred, alpha=0.5, s=10, c='blue')

    # Diagonal line
    min_val = min(ref.min(), pred.min())
    max_val = max(ref.max(), pred.max())
    margin = (max_val - min_val) * 0.05
    ax.plot([min_val - margin, max_val + margin],
            [min_val - margin, max_val + margin],
            'r--', linewidth=1, label='y = x')

    # Calculate metrics
    rmse, mae, r2 = calculate_metrics(pred, ref)

    # Add metrics text
    metrics_text = f'RMSE: {rmse:.4f} {unit}\nMAE: {mae:.4f} {unit}\nR²: {r2:.4f}'
    ax.text(0.05, 0.95, metrics_text, transform=ax.transAxes,
            verticalalignment='top', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_xlim(min_val - margin, max_val + margin)
    ax.set_ylim(min_val - margin, max_val + margin)
    ax.set_aspect('equal')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

    print(f"Saved: {save_path}")
    return rmse, mae, r2


def load_npz_as_cg_traj(npz_path, ntest=None):
    """Load CG NPZ file and convert to dict list for get_graphset_cg.

    Supports both fixed-size and multi-system flat NPZ (frame_offsets/frame_sizes).

    Args:
        npz_path: Path to CG NPZ file
        ntest: Number of test frames (from the end). None = all frames.

    Returns:
        cg_traj: List of dicts with keys: positions, types, forces, energy, cell
        atomistic_energies: Array of energies
    """
    data = np.load(npz_path, allow_pickle=True)
    energies = data['energies']
    cells = data['cells']
    is_multi = 'frame_offsets' in data

    n_frames = len(energies)
    if ntest is not None:
        indices = list(range(max(0, n_frames - ntest), n_frames))
    else:
        indices = list(range(n_frames))

    cg_traj = []

    if is_multi:
        positions_flat = data['positions']
        forces_flat = data['forces']
        types_flat = data['types']
        frame_offsets = data['frame_offsets']
        frame_sizes = data['frame_sizes']

        for idx in indices:
            off = int(frame_offsets[idx])
            sz = int(frame_sizes[idx])
            cg_traj.append({
                'positions': positions_flat[off:off + sz],
                'forces': forces_flat[off:off + sz],
                'types': types_flat[off:off + sz],
                'energy': float(energies[idx]),
                'cell': cells[idx],
            })
    else:
        positions = data['positions']
        forces = data['forces']
        types = data['types']

        for idx in indices:
            cg_traj.append({
                'positions': positions[idx],
                'forces': forces[idx],
                'types': types[idx] if types.ndim > 1 else types,
                'energy': float(energies[idx]),
                'cell': cells[idx],
            })

    selected_energies = energies[indices]
    return cg_traj, selected_energies


def main():
    parser = argparse.ArgumentParser(description='Evaluate CG Model')
    parser.add_argument('--model', type=str, default='model_cg.pkl',
                        help='Path to trained model checkpoint')
    parser.add_argument('--data', type=str, default='water.traj',
                        help='Path to atomistic trajectory or CG NPZ file')
    parser.add_argument('--input-type', type=str, default='auto',
                        choices=['auto', 'traj', 'npz'],
                        help='Input data type: traj (atomistic trajectory), npz (CG NPZ), '
                             'or auto (detect from extension)')
    parser.add_argument('--mapping', type=str, default='water',
                        help='CG mapping preset (for traj input). Options: water, methane, etc.')
    parser.add_argument('--ntest', type=int, default=100,
                        help='Number of test frames')
    parser.add_argument('--cutoff', type=float, default=10.0,
                        help='CG cutoff distance')
    parser.add_argument('--output', type=str, default='cg_evaluation',
                        help='Output prefix for plots')
    args = parser.parse_args()

    # Auto-detect input type
    input_type = args.input_type
    if input_type == 'auto':
        if args.data.endswith('.npz'):
            input_type = 'npz'
        else:
            input_type = 'traj'

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load model
    print(f"\nLoading model: {args.model}")
    model, ckpt = load_model(args.model, device)

    # Get saved parameters
    uniq_type = ckpt.get('uniq_element', {0: 0})
    enr_avg_per_type = ckpt.get('enr_avg_per_element', {0: 0.0})
    json_data = ckpt.get('input.json', {})

    print(f"Energy offset per CG type: {enr_avg_per_type}")
    print(f"Number of CG types: {len(uniq_type)}")

    if input_type == 'npz':
        # NPZ mode: data is already in CG representation
        print(f"\nLoading CG NPZ data: {args.data} (last {args.ntest} frames)")
        cg_traj, atomistic_energies = load_npz_as_cg_traj(args.data, ntest=args.ntest)
        print(f"Loaded {len(cg_traj)} CG frames")

    else:
        # Trajectory mode: need CG mapping
        from ase.io import read
        from bam_torch.utils.cg_mapping import CGMapping

        print(f"\nLoading test data: {args.data} (last {args.ntest} frames)")
        traj = read(args.data, index=slice(-args.ntest, None))
        print(f"Loaded {len(traj)} frames")

        # CG mapping (CGMapping accepts preset name strings directly)
        print(f"Applying CG mapping: {args.mapping}")
        mapping = CGMapping(args.mapping)

        # Convert to CG
        print("Converting to CG representation...")
        from bam_torch.utils.cg_mapping import convert_trajectory_to_cg
        cg_traj = convert_trajectory_to_cg(traj, mapping, show_progress=True)

        # Store original atomistic energies
        atomistic_energies = np.array([atoms.get_potential_energy() for atoms in traj])

    # Create graph dataset
    # Get bond_topology from checkpoint config or NPZ metadata
    bond_topology = None
    cg_config = json_data.get('cg_config', {})
    if cg_config.get('bond_topology'):
        bond_topology = cg_config['bond_topology']
    elif hasattr(args, 'data') and args.data.endswith('.npz'):
        try:
            npz_meta = np.load(args.data, allow_pickle=True)
            if 'metadata' in npz_meta:
                meta = npz_meta['metadata'].item()
                if isinstance(meta, dict) and 'bond_topology' in meta:
                    bt = meta['bond_topology']
                    bond_topology = bt.get('global', None) if isinstance(bt, dict) else None
        except Exception:
            pass

    print("Building CG graphs...")
    graphset = get_graphset_cg(
        cg_traj, args.cutoff, uniq_type, enr_avg_per_type, 1.0,
        regress_forces=True, max_neigh=None, show_progress=True,
        bond_topology=bond_topology,
    )

    dataloader = DataLoader(graphset, batch_size=4, shuffle=False)

    # Evaluate
    print("\nRunning evaluation...")
    energy_pred, energy_ref, force_pred, force_ref = evaluate(
        model, dataloader, device, enr_avg_per_type, uniq_type
    )

    # Print metrics
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)

    e_rmse, e_mae, e_r2 = calculate_metrics(energy_pred, energy_ref)
    print(f"\nEnergy:")
    print(f"  RMSE: {e_rmse:.4f} eV")
    print(f"  MAE:  {e_mae:.4f} eV")
    print(f"  R²:   {e_r2:.4f}")

    f_rmse, f_mae, f_r2 = calculate_metrics(force_pred, force_ref)
    print(f"\nForce (CG):")
    print(f"  RMSE: {f_rmse:.4f} eV/A")
    print(f"  MAE:  {f_mae:.4f} eV/A")
    print(f"  R²:   {f_r2:.4f}")

    # Create scatter plots
    print("\nGenerating plots...")

    plot_scatter(
        energy_pred, energy_ref,
        "CG Energy: Prediction vs Reference",
        "Reference Energy (eV)",
        "Predicted Energy (eV)",
        f"{args.output}_energy.png",
        unit="eV"
    )

    plot_scatter(
        force_pred, force_ref,
        "CG Force: Prediction vs Reference",
        "Reference Force (eV/A)",
        "Predicted Force (eV/A)",
        f"{args.output}_force.png",
        unit="eV/A"
    )

    # Also compare with atomistic energies
    if atomistic_energies is not None and len(atomistic_energies) == len(energy_pred):
        print("\n" + "="*60)
        print("COMPARISON WITH ATOMISTIC ENERGIES")
        print("="*60)

        ae_rmse, ae_mae, ae_r2 = calculate_metrics(energy_pred, atomistic_energies)
        print(f"\nCG Predicted vs Atomistic Reference:")
        print(f"  RMSE: {ae_rmse:.4f} eV")
        print(f"  MAE:  {ae_mae:.4f} eV")
        print(f"  R²:   {ae_r2:.4f}")

        plot_scatter(
            energy_pred, atomistic_energies,
            "CG Predicted vs Atomistic Energy",
            "Atomistic Energy (eV)",
            "CG Predicted Energy (eV)",
            f"{args.output}_energy_vs_atomistic.png",
            unit="eV"
        )

    print("\n" + "="*60)
    print("Evaluation complete!")
    print("="*60)


if __name__ == '__main__':
    main()
