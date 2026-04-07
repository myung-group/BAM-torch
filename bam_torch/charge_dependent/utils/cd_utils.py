"""
Charge-dependent data utilities for BAM-torch.

Utilities for converting QM9star data into PyG Data objects with charge information.
Also supports Wiggle150 test set loading.

Data sources:
  - Extended xyz files converted by qm9star_preprocessor.py
  - Or general xyz/traj files containing charge info
"""

import os
import hashlib
import numpy as np
from ase.io import read
from matscipy.neighbours import neighbour_list
from tqdm import tqdm

import torch
import torch.distributed as dist
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch.utils.data.distributed import DistributedSampler

from bam_torch.utils.utils import (
    get_enr_avg_per_element,
    get_relative_vector,
)

# =============================================================
# Unit conversion constants
# =============================================================
HARTREE_TO_EV = 27.211386245988       # eV/Hartree
BOHR_TO_ANGSTROM = 0.529177249        # Ang/bohr
HARTREE_BOHR_TO_EV_ANG = HARTREE_TO_EV / BOHR_TO_ANGSTROM  # ~ 51.4221


def _safe_get_forces(atoms):
    """Safely retrieve forces from an atoms object.

    In ASE 3.27+, 'forces' is stored in the internal calculator, not in arrays.
    """
    # (1) Read directly from arrays (older ASE)
    if 'forces' in atoms.arrays:
        return atoms.arrays['forces']
    # (2) ASE 3.27+: call get_forces() directly
    try:
        f = atoms.get_forces()
        if f is not None:
            return f
    except Exception:
        pass
    # (3) Read from calculator results
    if atoms.calc is not None:
        try:
            return atoms.get_forces()
        except Exception:
            pass
    return np.zeros((len(atoms), 3))


def _safe_get_energy(atoms):
    """Safely retrieve energy from an atoms object.

    In ASE 3.27+, 'energy' is accessed via get_potential_energy(), not info dict.
    """
    # (1) Read directly from info (older ASE)
    if 'energy' in atoms.info:
        return float(atoms.info['energy'])
    # (2) ASE 3.27+: call get_potential_energy() directly
    try:
        e = atoms.get_potential_energy()
        if e is not None:
            return float(e)
    except Exception:
        pass
    # (3) Read from calculator results
    if atoms.calc is not None:
        try:
            return float(atoms.get_potential_energy())
        except Exception:
            pass
    raise ValueError("Could not find energy. "
                     "Extended xyz or calculator is required.")


def _safe_get_stress(atoms):
    """Safely retrieve stress from an atoms object."""
    if atoms.calc is not None:
        try:
            if 'stress' in atoms.calc.results:
                return atoms.get_stress()
        except Exception:
            pass
    return np.zeros(6)


def _safe_get_volume(atoms):
    """Safely retrieve volume from an atoms object."""
    try:
        vol = atoms.get_volume()
        if vol > 0:
            return vol
    except Exception:
        pass
    return np.zeros(1)


def _extract_charges(atoms, charge_key="charges",
                     total_charge_key="total_charge"):
    """Extract charge information from an atoms object.

    In extended xyz format:
      - atoms.arrays['charges'] -> per-atom charges
      - atoms.info['total_charge'] -> system total charge

    Returns:
        atomic_charges: np.ndarray [n_atoms]
        total_charge: float
    """
    n_atoms = len(atoms)

    # --- Atomic charges ---
    atomic_charges = None

    # (1) Find in arrays (older ASE: Properties=...charges:R:1)
    if charge_key in atoms.arrays:
        atomic_charges = np.array(atoms.arrays[charge_key], dtype=float)
    # (1b) ASE 3.27+: 'charges' accessed via get_charges()
    elif charge_key == 'charges':
        try:
            q = atoms.get_charges()
            if q is not None and len(q) == n_atoms:
                atomic_charges = np.array(q, dtype=float)
        except Exception:
            pass
    # (2) Find in info
    elif charge_key in atoms.info:
        val = atoms.info[charge_key]
        if isinstance(val, (list, np.ndarray)):
            atomic_charges = np.array(val, dtype=float)
        elif isinstance(val, str):
            atomic_charges = np.array(
                [float(x) for x in val.split()], dtype=float
            )

    # (3) Try other common key names
    if atomic_charges is None:
        for alt_key in ['charge', 'npa_charges', 'mulliken_charge',
                        'hirshfeld_charges', 'formal_charges',
                        'initial_charges']:
            if alt_key in atoms.arrays:
                atomic_charges = np.array(
                    atoms.arrays[alt_key], dtype=float
                )
                break
            elif alt_key in atoms.info:
                val = atoms.info[alt_key]
                if isinstance(val, (list, np.ndarray)):
                    atomic_charges = np.array(val, dtype=float)
                    break

    if atomic_charges is None:
        atomic_charges = np.zeros(n_atoms)

    # --- Total charge ---
    if total_charge_key in atoms.info:
        total_charge = float(atoms.info[total_charge_key])
    else:
        total_charge = float(np.sum(atomic_charges))

    # --- Total multiplicity ---
    mult_key = "total_multiplicity"
    if mult_key in atoms.info:
        total_multiplicity = int(atoms.info[mult_key])
    else:
        total_multiplicity = 1  # default: singlet

    return atomic_charges, total_charge, total_multiplicity


def get_graphset_charge(
    data, cutoff, uniq_element, enr_avg_per_element,
    enr_var, regress_forces=True, max_neigh=None,
    show_progress=False, desc="Converting",
    charge_key="charges",
    total_charge_key="total_charge",
):
    """
    Create graph dataset with charge information (extension of get_graphset).

    Supports both extended xyz (qm9star_preprocessor.py output) and general ASE files.

    Args:
        data: list of ASE atoms objects
        cutoff: distance cutoff (Angstrom)
        uniq_element: {atomic_number: species_index} dictionary
        enr_avg_per_element: per-element average energy
        enr_var: energy variance
        regress_forces: whether to regress forces
        max_neigh: maximum number of neighbors
        show_progress: whether to show tqdm progress bar
        desc: progress bar description
        charge_key: atomic charge key name
        total_charge_key: total charge key name

    Returns:
        graph_list: list of PyG Data objects
    """
    graph_list = []
    iterator = tqdm(data, desc=desc, leave=False) if show_progress else data

    for atoms in iterator:
        crds = atoms.get_positions()
        node_enr_avg = np.array([
            enr_avg_per_element[uniq_element[iz]]
            for iz in atoms.numbers
        ])

        # Energy (safe read)
        enr = _safe_get_energy(atoms) - node_enr_avg.sum()

        # Forces (safe read)
        if regress_forces or regress_forces == 'direct':
            frc = _safe_get_forces(atoms)
            volume = _safe_get_volume(atoms)
        else:
            frc = np.zeros((len(atoms), 3))
            volume = np.zeros(1)

        # Cell handling
        cell = np.array(atoms.get_cell())
        if np.all(cell == 0.0):
            cell = np.diag([30., 30., 30.])
            atoms.set_cell(cell)

        # Stress
        stress = _safe_get_stress(atoms)
        if np.all(stress == 0):
            volume = np.zeros(1)

        # Neighbor list
        iatoms, jatoms, Sij = neighbour_list(
            quantities='ijS', atoms=atoms, cutoff=cutoff
        )
        species = np.array([uniq_element[iz] for iz in atoms.numbers])
        num_nodes = crds.shape[0]
        num_edges = iatoms.shape[0]

        # Maximum neighbor count limit
        if max_neigh is not None:
            Rij, dist = get_relative_vector(atoms, iatoms, jatoms, Sij)
            nonmax_idx = []
            for i in range(len(atoms)):
                idx_i = (iatoms == i).nonzero()[0]
                idx_sorted = np.argsort(dist[idx_i])[:max_neigh]
                nonmax_idx.append(idx_i[idx_sorted])
            nonmax_idx = np.concatenate(nonmax_idx)
            iatoms = iatoms[nonmax_idx]
            jatoms = jatoms[nonmax_idx]
            num_edges = iatoms.shape[0]
            Sij = Sij[nonmax_idx]

        # Extract charge information
        atomic_charges, total_charge, total_multiplicity = _extract_charges(
            atoms, charge_key, total_charge_key
        )

        # Create PyG Data object
        graph = Data(
            positions=torch.tensor(crds, dtype=torch.float32),
            species=torch.tensor(species, dtype=torch.long),
            forces=torch.tensor(frc, dtype=torch.float32),
            edges=torch.tensor(Sij, dtype=torch.float32),
            num_nodes=num_nodes,
            num_edges=num_edges,
            energy=torch.tensor(enr, dtype=torch.float32),
            cell=torch.tensor(
                np.array(cell), dtype=torch.float32
            ).view(1, 3, 3),
            edge_index=torch.tensor(
                np.array([iatoms, jatoms]), dtype=torch.long
            ),
            stress=torch.tensor(stress, dtype=torch.float32),
            volume=torch.tensor(volume),
            # Charge-related fields
            atomic_charges=torch.tensor(
                atomic_charges, dtype=torch.float32
            ),
            total_charge=torch.tensor(
                total_charge, dtype=torch.float32
            ),
            total_multiplicity=torch.tensor(
                total_multiplicity, dtype=torch.long
            ),
        )
        graph_list.append(graph)

    return graph_list


def _get_cache_path(fname, ntrain, nvalid, random_seed, cutoff,
                    charge_key, regress_forces, max_neigh):
    """Generate deterministic cache file path based on data parameters."""
    fsize = os.path.getsize(fname)
    key = (f"{os.path.abspath(fname)}|{fsize}|{ntrain}|{nvalid}|"
           f"{random_seed}|{cutoff}|{charge_key}|{regress_forces}|{max_neigh}")
    h = hashlib.md5(key.encode()).hexdigest()[:12]
    basename = os.path.splitext(os.path.basename(fname))[0]
    cache_dir = os.path.dirname(os.path.abspath(fname))
    return os.path.join(cache_dir, f".cache_{basename}_{h}.pt")


def _wait_for_cache(cache_path, rank, poll_interval=10):
    """Wait for rank 0 to finish writing the cache file.

    Uses file-based synchronization instead of dist.barrier() to avoid
    NCCL timeout when rank 0 takes a long time to convert large datasets.
    """
    import time
    ready_path = cache_path + '.ready'
    if rank == 0:
        # Signal that cache is ready
        with open(ready_path, 'w') as f:
            f.write('ready')
    else:
        # Poll for the sentinel file
        while not os.path.exists(ready_path):
            time.sleep(poll_interval)
        time.sleep(1)  # ensure file is fully flushed


def get_dataloader_charge(
    fname, ntrain, nvalid, nbatch, cutoff, random_seed,
    element=None, regress_forces=True, max_neigh=None,
    charge_key="charges", total_charge_key="total_charge",
    rank=0, world_size=1,
):
    """
    Create DataLoader with charge information.

    Accepts extended xyz converted by qm9star_preprocessor.py or
    general ASE files as input.

    For DDP (world_size > 1), only rank 0 reads and converts the raw data,
    then saves to a .pt cache file. Other ranks load from the cache after
    a barrier, reducing peak CPU memory from N×full to 1×full + N×graphs.

    Args:
        fname: structure file path (.xyz, .traj, etc.)
        ntrain: number of training samples
        nvalid: number of validation samples
        nbatch: batch size
        cutoff: distance cutoff
        random_seed: random seed
        element: element list ("auto" or list)
        regress_forces: whether to regress forces
        max_neigh: maximum number of neighbors
        charge_key: atomic charge key name
        total_charge_key: total charge key name
        rank: DDP rank
        world_size: DDP world_size

    Returns:
        train_loader, valid_loader, uniq_element, enr_avg_per_element
    """
    cache_path = _get_cache_path(
        fname, ntrain, nvalid, random_seed, cutoff,
        charge_key, regress_forces, max_neigh,
    )

    if rank == 0:
        if os.path.exists(cache_path):
            # Load from existing cache
            print(f"\033[32mLoading cached dataset: {cache_path}\033[0m")
            cached = torch.load(cache_path, weights_only=False)
            train_graphs = cached['train_graphs']
            valid_graphs = cached['valid_graphs']
            uniq_element = cached['uniq_element']
            enr_avg_per_element = cached['enr_avg_per_element']
        else:
            # Read and convert from scratch
            traj = read(fname, index=':')

            if element is None or element == "auto":
                element = sorted(
                    list(set(int(atom.number)
                             for atoms in traj for atom in atoms))
                )
            elif isinstance(element, str):
                element = [int(e) for e in element.split()]

            enr_avg_per_element, uniq_element, enr_var = \
                get_enr_avg_per_element(traj, element)

            # Stratified train/valid split by (total_charge, total_multiplicity)
            # Ensures each charge/spin group is proportionally represented
            rng = np.random.RandomState(random_seed)

            groups = {}
            for i, atoms in enumerate(traj):
                tc = atoms.info.get(total_charge_key, 0.0)
                tm = atoms.info.get('total_multiplicity', 1)
                gk = (float(tc), int(tm) if isinstance(tm, (int, float)) else 1)
                groups.setdefault(gk, []).append(i)

            if len(groups) > 1:
                # Stratified: allocate train/valid proportionally per group
                train_idx = []
                valid_idx = []
                for gk in sorted(groups):
                    g_idx = np.array(groups[gk])
                    rng.shuffle(g_idx)
                    frac = len(g_idx) / len(traj)
                    n_tr = max(1, int(round(ntrain * frac)))
                    n_va = max(1, int(round(nvalid * frac)))
                    n_tr = min(n_tr, len(g_idx))
                    n_va = min(n_va, len(g_idx) - n_tr)
                    train_idx.extend(g_idx[:n_tr])
                    valid_idx.extend(g_idx[n_tr:n_tr + n_va])
                    print(f"  \033[36mGroup charge={gk[0]:+.0f} mult={gk[1]}: "
                          f"{len(g_idx)} total -> "
                          f"{n_tr} train / {n_va} valid\033[0m")
                train_data = [traj[i] for i in train_idx]
                valid_data = [traj[i] for i in valid_idx]
            else:
                # Single group: fallback to plain shuffle
                idx = rng.permutation(len(traj))
                train_data = [traj[i] for i in idx[:ntrain]]
                valid_data = [traj[i] for i in idx[ntrain:ntrain + nvalid]]

            del traj, groups  # free ASE trajectory memory

            print(f"\n\033[32mConverting training data ({ntrain}) "
                  f"with charge info...\033[0m")
            train_graphs = get_graphset_charge(
                train_data, cutoff, uniq_element,
                enr_avg_per_element, enr_var,
                regress_forces=regress_forces,
                max_neigh=max_neigh,
                show_progress=True,
                desc="Train",
                charge_key=charge_key,
                total_charge_key=total_charge_key,
            )
            del train_data

            print(f"\033[32mConverting validation data ({nvalid}) "
                  f"with charge info...\033[0m")
            valid_graphs = get_graphset_charge(
                valid_data, cutoff, uniq_element,
                enr_avg_per_element, enr_var,
                regress_forces=regress_forces,
                max_neigh=max_neigh,
                show_progress=True,
                desc="Valid",
                charge_key=charge_key,
                total_charge_key=total_charge_key,
            )
            del valid_data

            # Save cache for other ranks (and future reruns)
            if world_size > 1:
                print(f"\033[32mSaving dataset cache: {cache_path}\033[0m")
                torch.save({
                    'train_graphs': train_graphs,
                    'valid_graphs': valid_graphs,
                    'uniq_element': uniq_element,
                    'enr_avg_per_element': enr_avg_per_element,
                }, cache_path)

    # File-based sync: avoids NCCL timeout during long conversions
    if world_size > 1:
        _wait_for_cache(cache_path, rank)

    # Non-rank-0: load from cache
    if rank != 0:
        cached = torch.load(cache_path, weights_only=False)
        train_graphs = cached['train_graphs']
        valid_graphs = cached['valid_graphs']
        uniq_element = cached['uniq_element']
        enr_avg_per_element = cached['enr_avg_per_element']
        del cached

    # Create DataLoader (DDP: partition data with DistributedSampler)
    train_sampler = None
    valid_sampler = None
    if world_size > 1:
        train_sampler = DistributedSampler(
            train_graphs, num_replicas=world_size, rank=rank, shuffle=True
        )
        valid_sampler = DistributedSampler(
            valid_graphs, num_replicas=world_size, rank=rank, shuffle=False
        )

    train_loader = DataLoader(
        train_graphs, batch_size=nbatch,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        pin_memory=True,
        num_workers=min(4, os.cpu_count() or 1),
        drop_last=False,
    )
    valid_loader = DataLoader(
        valid_graphs, batch_size=nbatch,
        shuffle=False,
        sampler=valid_sampler,
        pin_memory=True,
        num_workers=min(4, os.cpu_count() or 1),
        drop_last=False,
    )

    return train_loader, valid_loader, uniq_element, enr_avg_per_element


def get_dataloader_charge_to_predict(
    fname, ndata, nbatch, cutoff, model_ckpt,
    regress_forces=True, max_neigh=None,
    charge_key="charges", total_charge_key="total_charge",
):
    """
    Create evaluation DataLoader (with charge information).

    Args:
        fname: structure file path
        ndata: number of data (file path or integer)
        nbatch: batch size
        cutoff: distance cutoff
        model_ckpt: trained model checkpoint
        regress_forces: whether to regress forces
        max_neigh: maximum number of neighbors
        charge_key: atomic charge key name
        total_charge_key: total charge key name

    Returns:
        data_loader, uniq_element, enr_avg_per_element
    """
    # Read data
    if isinstance(ndata, str):
        traj = read(ndata, index=':')
    else:
        traj = read(fname, index=f':{ndata}')

    # Restore element information from checkpoint
    uniq_element = model_ckpt['uniq_element']
    enr_avg_per_element = model_ckpt['enr_avg_per_element']
    enr_var = 1.0  # Variance not used for normalization during prediction

    # Create graph dataset
    graphs = get_graphset_charge(
        traj, cutoff, uniq_element,
        enr_avg_per_element, enr_var,
        regress_forces=regress_forces,
        max_neigh=max_neigh,
        show_progress=True,
        desc="Predict",
        charge_key=charge_key,
        total_charge_key=total_charge_key,
    )

    data_loader = DataLoader(graphs, batch_size=nbatch, shuffle=False)
    return data_loader, uniq_element, enr_avg_per_element
