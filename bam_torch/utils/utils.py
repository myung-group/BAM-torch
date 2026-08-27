import numpy as np
from ase.io import read
from scipy.optimize import minimize
from matscipy.neighbours import neighbour_list
from tqdm import tqdm

import torch
from torch import vmap
from torch_geometric.data import Data
from torch_geometric.data import Batch as DataBatch
from torch_geometric.loader import DataLoader
from torch.utils.data.distributed import DistributedSampler

import os
import pprint
from copy import deepcopy
from datetime import datetime
from .sampler import DistributedBalancedAtomCountBatchSampler


def get_enr_avg_per_element (traj, element):
    tgt_enr = np.array([atoms.get_potential_energy()
                    for atoms in traj])
    uniq_element = {int(e): i for i, e in enumerate(element)}
    element_counts = {i: np.array([ (atoms.numbers == e).sum()
                                   for atoms in traj])
                                for e, i in uniq_element.items()}
    c0 = np.array ([element_counts[i] for i in element_counts.keys()])
    m0 = tgt_enr.sum()/c0.sum()
    w0 = np.array ([m0 for _ in element], dtype=np.float64)

    def loss_fn (weight, count):
        # weight:  (nspec)
        # count:  (nspec, ndata)
        def objective_mean (w0, c0):
            # w0: weight (nspec)
            # c0: count  (nspec, ndata)
            return np.einsum('i,ij->j', w0, c0)
        prd_enr = objective_mean (weight, count)
        diff = (tgt_enr - prd_enr)
        return (diff*diff).mean()

    results = minimize (loss_fn, x0=w0, args=(c0,), method='BFGS')
    w0 = results.x
    enr_avg_per_element = {}
    for i, e in enumerate(element):
        enr_avg_per_element[i] = w0[i]

    return enr_avg_per_element, uniq_element, np.var(tgt_enr)


def get_enr_avg_per_element_with_ztable(traj, element, uniq_element, foundation_enr_avg):
    """
    Compute per-element E0s using the foundation model's z_table (MACE-style)
    
    Args:
        traj: List of atomic structures
        element: List of atomic numbers belonging to this head (e.g., [3, 15, 16, 17])
        uniq_element: Foundation z_table mapping {1: 0, 2: 1, 3: 2, ...}
        foundation_enr_avg: Foundation E0s dictionary {0: -3.66, 1: -1.34, ...}
    
    Returns:
        enr_avg_per_element: Updated E0s based on the foundation z_table
        enr_var: Variance of target energies
    """
    tgt_enr = np.array([atoms.get_potential_energy() for atoms in traj])
    
    # Retrieve species indices for the elements of this head
    species_indices = []
    for z in element:
        if z in uniq_element:
            species_indices.append(uniq_element[z])
        else:
            raise ValueError(f"Element {z} not in foundation z_table!")
    
    # Count occurrences of each element across the trajectory (indexed by foundation z_table)
    element_counts = {}
    for z in element:
        species_idx = uniq_element[z]
        element_counts[species_idx] = np.array([
            (atoms.numbers == z).sum() for atoms in traj
        ])
    
    # Optimization: compute E0s for the elements of this head
    c0 = np.array([element_counts[idx] for idx in species_indices])
    m0 = tgt_enr.sum() / c0.sum() if c0.sum() > 0 else 0.0
    w0 = np.array([m0 for _ in element], dtype=np.float64)
    
    def loss_fn(weight, count):
        prd_enr = np.einsum('i,ij->j', weight, count)
        diff = tgt_enr - prd_enr
        return (diff * diff).mean()
    
    results = minimize(loss_fn, x0=w0, args=(c0,), method='BFGS')
    optimized_weights = results.x
    
    # Copy foundation E0s and update only the elements belonging to this head
    enr_avg_per_element = dict(foundation_enr_avg) 
    for i, z in enumerate(element):
        species_idx = uniq_element[z]
        enr_avg_per_element[species_idx] = optimized_weights[i]
    
    return enr_avg_per_element, np.var(tgt_enr)
    

def get_graphset(data, cutoff, uniq_element, enr_avg_per_element, 
                 enr_var, regress_forces=True, max_neigh=None, 
                 show_progress=False, desc="Converting"):
    graph_list = []
    iterator = tqdm(data, desc=desc, leave=False) if show_progress else data
    for atoms in iterator:
        crds = atoms.get_positions()
        node_enr_avg = np.array([enr_avg_per_element[uniq_element[iz]]
                                  for iz in atoms.numbers])
        #enr = (atoms.get_potential_energy() - node_enr_avg.sum()) / enr_var
        enr = atoms.get_potential_energy() - node_enr_avg.sum()

        if regress_forces or regress_forces == 'direct':
            frc = atoms.get_forces()
            stress = np.zeros(6)
        else:
            frc = np.zeros((len(atoms), 3))

        cell = atoms.get_cell()
        if np.all(cell == [0.0, 0.0, 0.0]):
            cell = np.diag([30., 30., 30.])
            atoms.set_cell(cell)
        
        calculator = atoms.calc
        has_stress = calculator is not None and 'stress' in calculator.results
        if has_stress:
            stress = atoms.get_stress()
            volume = atoms.get_volume()
        else:
            stress = np.zeros(6)
            volume = np.zeros(1)
    
        iatoms, jatoms, Sij = neighbour_list(quantities='ijS',
                                             atoms=atoms,
                                             cutoff=cutoff)
        species = np.array([uniq_element[iz] for iz in atoms.numbers])
        num_nodes = crds.shape[0] 
        num_edges = iatoms.shape[0]

        # Sort neighbors by distance, remove edges larger than max_neighbors
        if max_neigh != None:
            Rij, dist = get_relative_vector(atoms, iatoms, jatoms, Sij)
            nonmax_idx = []
            for i in range(len(atoms)):
                idx_i = (iatoms == i).nonzero()[0]
                idx_sorted = np.argsort(dist[idx_i])[: max_neigh]
                nonmax_idx.append(idx_i[idx_sorted])
            nonmax_idx = np.concatenate(nonmax_idx)
            iatoms = iatoms[nonmax_idx]
            jatoms = jatoms[nonmax_idx]
            num_edges = iatoms.shape[0]
            Sij = Sij[nonmax_idx]
        
        # Generate Graph data set
        graph = Data(
            positions=torch.tensor(crds, dtype=torch.float32),   # node features
            species=torch.tensor(species, dtype=torch.long),
            forces=torch.tensor(frc, dtype=torch.float32),
            edges=torch.tensor(Sij, dtype=torch.float32),# edge features
            num_nodes=num_nodes,             
            num_edges=num_edges,
            energy=torch.tensor([enr], dtype=torch.float32),
            cell=torch.tensor(np.array(cell), dtype=torch.float32).view(1, 3, 3),
            edge_index=torch.tensor(np.array([iatoms, jatoms]), dtype=torch.long),  # senders, recerivers
            stress=torch.tensor(stress, dtype=torch.float32),
            stress_valid=torch.tensor([has_stress], dtype=torch.bool),
            volume=torch.tensor([volume] if np.isscalar(volume) else volume, dtype=torch.float32)
        )                          
        graph_list.append(graph)

    return graph_list


def get_graphset_with_pad(graphset, pad_nodes_to, pad_edges_to):
    graph_list = []
    for data in graphset:
        n_nodes = data.num_nodes
        original_n_nodes = n_nodes
        if n_nodes < pad_nodes_to:
            padding = torch.zeros((pad_nodes_to - n_nodes, data.positions.size(1)))
            data.positions = torch.cat([data.positions, padding], dim=0)
            data.forces = torch.cat([data.forces, padding], dim=0)
            data.species = torch.cat([data.species, padding[:, 0]], dim=0).to(torch.long)
            data.num_nodes = pad_nodes_to
            node_mask = torch.cat([torch.ones(original_n_nodes), 
                                   torch.zeros(pad_nodes_to - original_n_nodes)])
        else:
            node_mask = torch.ones(n_nodes)   
        # pad edges (attr)
        n_edges = data.num_edges
        original_n_edges = n_edges
        if n_edges < pad_edges_to:
            edge_attr_padding = torch.zeros((pad_edges_to - n_edges, 3))
            edge_index_padding = torch.zeros((2, pad_edges_to - n_edges), dtype=torch.long)
            data.edges = torch.cat([data.edges, edge_attr_padding], dim=0)
            data.num_edges = pad_edges_to
            data.edge_index = torch.cat([data.edge_index, edge_index_padding], dim=1)
            edge_mask = torch.cat([torch.ones(original_n_edges), 
                                   torch.zeros(pad_edges_to - original_n_edges)])
        else:
            edge_mask = torch.ones(n_edges)
        data.node_mask = node_mask
        data.edge_mask = edge_mask
        data.natoms = torch.tensor(pad_nodes_to).long()

        graph_list.append(data)
    return graph_list


def get_dataloader(fname, ntrain, nvalid, 
                   nbatch, cutoff, random_seed, 
                   element=None, regress_forces=True,
                   max_neigh=None,
                   rank=0, world_size=1):
    msg = ''
    if type(ntrain) == str: 
        train_data = read(ntrain, index=slice(None))
        valid_data = read(nvalid, index=slice(None))
        msg += 'number of data:\n'
        msg += f'\033[33m -- training      {len(train_data)}\n'
        msg += f' -- validation    {len(valid_data)}\033[0m\n\n'
        traj = train_data + valid_data
    else:
        nsamp = ntrain + nvalid
        traj = read(fname, index=slice(None))[-nsamp:]
        torch.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)
        idx = torch.arange(nsamp)
        idx = idx[torch.randperm(nsamp)] 
        idx_train = idx[:ntrain]
        idx_valid = idx[ntrain:]   
        train_data = [traj[i] for i in idx_train]
        valid_data = [traj[i] for i in idx_valid]
        msg += 'number of data:\n'
        msg += f'\033[33m -- training      {len(train_data)}\n'
        msg += f' -- validation    {len(valid_data)}\033[0m\n\n'

    if element == None or element == 'auto':
        element = sorted(
            list(set(atom.number for atoms in traj 
                                  for atom in atoms))
        )  # traj: ase.Atoms
    enr_avg_per_element, uniq_element, enr_var = get_enr_avg_per_element (traj, element) 
    msg += f'mean energy per element:\n {enr_avg_per_element}\n'
    if rank == 0:
        print(msg)
    
    loaders = []
    for dataset in [train_data, valid_data]:
        graphset = get_graphset(dataset, cutoff, uniq_element,
                                enr_avg_per_element, enr_var,
                                regress_forces, max_neigh)
        data_sampler = None
        if world_size > 1:
            data_sampler = DistributedSampler(
                                graphset, num_replicas=world_size, rank=rank
                        )
        loader = DataLoader(graphset,
                            nbatch,
                            shuffle=False,
                            drop_last=False,
                            pin_memory=True,
                            num_workers=0,
                            collate_fn=None,
                            sampler=data_sampler)
        loaders.append(loader)
        # train_loader, test_loader
    return loaders[0], loaders[1], uniq_element, enr_avg_per_element  


def get_dataloader_multihead(datasets_config, cutoff, nbatch, regress_forces=True,
                              max_neigh=None, foundation_enr_avg=None, 
                              foundation_uniq_element=None, rank=0, world_size=1,
                              smoke_config=None):
    """
    Create dataloaders for multi-head training
    
    Args:
        datasets_config: list of dicts with keys:
            - name: head name
            - ntrain: train data path
            - nvalid: valid data path
            - loss_weight: weight for this head's loss
            - use_foundation_e0s: if True, use foundation model's enr_avg_per_element
        cutoff: cutoff radius
        nbatch: batch size
        regress_forces: whether to regress forces
        max_neigh: max neighbors
        foundation_enr_avg: foundation model's enr_avg_per_element (for replay)
        foundation_uniq_element: foundation model's uniq_element (for replay)
        rank: process rank
        world_size: number of processes
        smoke_config: smoke test config (enabled, max_samples)
    
    Returns:
        train_loader, valid_loader, uniq_element, enr_avg_per_element
    """
    from ase.io import read
    import random
    
    # Smoke test config
    smoke_enabled = smoke_config.get('enabled', False) if smoke_config else False
    smoke_max_samples = smoke_config.get('max_samples', 100) if smoke_config else 100
    
    if rank == 0 and smoke_enabled:
        print(f"\n⚠️ SMOKE TEST: Limiting data to {smoke_max_samples} samples per dataset")
    
    all_train_graphs = []
    all_valid_graphs = []
    global_uniq_element = None
    global_enr_avg_per_element = None
    
    # Per-head E0s storage for checkpoint
    per_head_enr_avg = {}
    

    for head_idx, ds_config in enumerate(datasets_config):
        head_name = ds_config.get("name", f"head_{head_idx}")
        train_path = ds_config.get("ntrain")
        valid_path = ds_config.get("nvalid")
        loss_weight = ds_config.get("loss_weight", 1.0)
        use_foundation_e0s = ds_config.get("use_foundation_e0s", False)
        
        if rank == 0:
            print(f"\n[Head {head_idx}: {head_name}]")
            print(f"  - Train: {train_path}")
            print(f"  - Valid: {valid_path}")
            print(f"  - Use foundation E0s: {use_foundation_e0s}")
        
        # Load data
        train_data = read(train_path, index=':') if train_path else []
        valid_data = read(valid_path, index=':') if valid_path else []
        
        # Smoke test: limit number of samples
        if smoke_enabled:
            train_data = train_data[:smoke_max_samples]
            valid_data = valid_data[:smoke_max_samples]
            if rank == 0:
                print(f"  - [SMOKE] Limited to {len(train_data)} train, {len(valid_data)} valid")
        
        traj = train_data + valid_data
        
        if not traj:
            continue
        
        # Extract element list (convert to int to avoid np.int64)
        element = sorted([int(z) for z in set(atom.number for atoms in traj for atom in atoms)])
        
        # MACE-style: all heads share foundation uniq_element, but each head gets its own E0s
        if foundation_uniq_element is not None:
            # Use foundation uniq_element (shared across heads)
            uniq_element = foundation_uniq_element
            
            if use_foundation_e0s and foundation_enr_avg is not None:
                # Use foundation E0s
                enr_avg_per_element = foundation_enr_avg
                enr_var = 1.0
                if rank == 0:
                    print(f"  - Using foundation E0s (foundation z_table)")
            else:
                # Compute E0s for this head, based on foundation z_table
                from bam_torch.utils.utils import get_enr_avg_per_element_with_ztable
                enr_avg_per_element, enr_var = get_enr_avg_per_element_with_ztable(
                    traj, element, uniq_element, foundation_enr_avg
                )
                if rank == 0:
                    print(f"  - Calculated E0s for this head (foundation z_table)")
        else:
            # No foundation -> compute E0s using original method
            enr_avg_per_element, uniq_element, enr_var = get_enr_avg_per_element(traj, element)
            if rank == 0:
                print(f"  - Calculated new enr_avg_per_element (no foundation)")
        
        if rank == 0:
            print(f"  - Elements: {element}")
        
        # Store per-head E0s for checkpoint
        per_head_enr_avg[head_idx] = {
            'name': head_name,
            'enr_avg_per_element': dict(enr_avg_per_element),
            'use_foundation_e0s': use_foundation_e0s,
            'elements': element
        }
        
        # First head becomes global (for backward compatibility)
        if head_idx == 0:
            global_uniq_element = uniq_element
            global_enr_avg_per_element = enr_avg_per_element

        # Generate graphs and attach head information
        for data, graphs_list, data_type in [(train_data, all_train_graphs, 'train'), 
                                              (valid_data, all_valid_graphs, 'valid')]:
            if data:
                desc = f"[{head_name}] {data_type}"
                graphs = get_graphset(data, cutoff, uniq_element, 
                                      enr_avg_per_element, enr_var,
                                      regress_forces, max_neigh,
                                      show_progress=(rank == 0), desc=desc)
                for g in graphs:
                    g.head = torch.tensor([head_idx], dtype=torch.long)
                    g.config_head = torch.tensor([head_idx], dtype=torch.long)
                    g.weight = torch.tensor([loss_weight], dtype=torch.float)
                graphs_list.extend(graphs)
        
        if rank == 0:
            print(f"  - Train: {len(train_data)}, Valid: {len(valid_data)} graphs")
    
    # --- Per-head DataLoaders (JAX-style separate streams) ---
    per_head_train_loaders = {}
    per_head_valid_loaders = {}
    per_head_train_graphs = {}
    per_head_valid_graphs = {}

    # Separate graphs by head index
    for g in all_train_graphs:
        h = int(g.head.item())
        per_head_train_graphs.setdefault(h, []).append(g)
    for g in all_valid_graphs:
        h = int(g.head.item())
        per_head_valid_graphs.setdefault(h, []).append(g)

    for h in sorted(set(list(per_head_train_graphs.keys()) + list(per_head_valid_graphs.keys()))):
        # Train loader per head
        train_gs = per_head_train_graphs.get(h, [])
        random.shuffle(train_gs)
        sampler_t = DistributedSampler(train_gs, num_replicas=world_size, rank=rank) if world_size > 1 else None
        per_head_train_loaders[h] = DataLoader(
            train_gs, nbatch, shuffle=(sampler_t is None),
            drop_last=False, pin_memory=True, sampler=sampler_t
        )
        # Valid loader per head
        valid_gs = per_head_valid_graphs.get(h, [])
        sampler_v = DistributedSampler(valid_gs, num_replicas=world_size, rank=rank) if world_size > 1 else None
        per_head_valid_loaders[h] = DataLoader(
            valid_gs, nbatch, shuffle=False,
            drop_last=False, pin_memory=True, sampler=sampler_v
        )

    # --- Mixed DataLoaders (backward-compatible) ---
    random.shuffle(all_train_graphs)
    data_sampler_train = None
    data_sampler_valid = None
    if world_size > 1:
        data_sampler_train = DistributedSampler(all_train_graphs, num_replicas=world_size, rank=rank)
        data_sampler_valid = DistributedSampler(all_valid_graphs, num_replicas=world_size, rank=rank)

    train_loader = DataLoader(all_train_graphs, nbatch, shuffle=(data_sampler_train is None),
                              drop_last=False, pin_memory=True, sampler=data_sampler_train)
    valid_loader = DataLoader(all_valid_graphs, nbatch, shuffle=False,
                              drop_last=False, pin_memory=True, sampler=data_sampler_valid)

    if rank == 0:
        print(f"\n✓ Multihead Dataloaders created")
        print(f"  - Total train: {len(all_train_graphs)}, valid: {len(all_valid_graphs)}")
        for h in sorted(per_head_train_loaders.keys()):
            n_t = len(per_head_train_graphs.get(h, []))
            n_v = len(per_head_valid_graphs.get(h, []))
            print(f"  - Head {h}: train {n_t}, valid {n_v}")
        print(f"  - Batch size: {nbatch}")
        print(f"  - Per-head E0s stored for {len(per_head_enr_avg)} heads")

    return (train_loader, valid_loader,
            per_head_train_loaders, per_head_valid_loaders,
            global_uniq_element, global_enr_avg_per_element, per_head_enr_avg)


def get_graphset_to_predict(data, cutoff, uniq_element, 
                            regress_forces=True, max_neigh=None):
    graph_list = []
    for atoms in data:
        if atoms.calc:
            enr = atoms.get_potential_energy()
        else:
            enr = 0.0
        crds = atoms.get_positions()

        if (regress_forces or regress_forces == 'direct') and atoms.calc:
            frc = atoms.get_forces()
            volume = atoms.get_volume()
            stress = np.zeros(6)
            if 'stress' in atoms._calc.results.keys():
                stress = atoms.get_stress()
        else:
            frc = np.zeros((len(atoms), 3))
            stress = np.zeros(6)
            volume = np.zeros(1)

        cell = atoms.get_cell()
        if np.all(cell == [0.0, 0.0, 0.0]):
            cell = np.diag([30., 30., 30.])
            atoms.set_cell(cell)
        
        iatoms, jatoms, Sij = neighbour_list(quantities='ijS',
                                             atoms=atoms,
                                             cutoff=cutoff)
        _, neighbors = torch.unique(torch.tensor(iatoms, dtype=torch.long), 
                                    return_counts=True)
        species = np.array([uniq_element[iz] for iz in atoms.numbers])
        num_nodes = crds.shape[0]
        num_edges = iatoms.shape[0]
        
        # Sort neighbors by distance, remove edges larger than max_neighbors
        if max_neigh != None:
            Rij, dist = get_relative_vector(atoms, iatoms, jatoms, Sij)
            nonmax_idx = []
            for i in range(len(atoms)):
                idx_i = (iatoms == i).nonzero()[0]
                idx_sorted = np.argsort(dist[idx_i])[: max_neigh]
                nonmax_idx.append(idx_i[idx_sorted])
            nonmax_idx = np.concatenate(nonmax_idx)
            iatoms = iatoms[nonmax_idx]
            jatoms = jatoms[nonmax_idx]
            num_edges = iatoms.shape[0]
            Sij = Sij[nonmax_idx]
        
        # Generate Graph data set
        graph = Data(
            positions=torch.tensor(crds, dtype=torch.float32),   # node features
            species=torch.tensor(species, dtype=torch.long),
            forces=torch.tensor(frc, dtype=torch.float32),
            edges=torch.tensor(Sij, dtype=torch.float32),# edge features
            num_nodes=num_nodes,             
            num_edges=num_edges,
            energy=torch.tensor(enr, dtype=torch.float32),
            cell=torch.tensor(np.array(cell), dtype=torch.float32).view(1, 3, 3),
            edge_index=torch.tensor(np.array([iatoms, jatoms]), dtype=torch.long)
        )   
        graph["positions"].requires_grad_(True)
        graph["cell"].requires_grad_(True)
        graph_list.append(graph)

        del atoms
    return graph_list


def get_dataloader_to_predict(fname, ndata, nbatch, 
                              cutoff, model_ckpt,
                              regress_forces=True, 
                              max_neigh=None):
    if type(ndata) == str:
        traj = read(fname, index=slice(None))
        print('number of data:')
        print(f'\033[33m -- test          {len(traj)}\033[0m\n')
    else: 
        traj = read(fname, index=slice(None))[:ndata]
        print('number of data:')
        print(f'\033[33m -- test          {len(traj)}\033[0m\n')

    uniq_element = model_ckpt['uniq_element']
    enr_avg_per_element = model_ckpt['enr_avg_per_element']

    graphset = get_graphset_to_predict(traj, cutoff, uniq_element, 
                                       regress_forces, max_neigh)
    #pad_nodes_to = 0 # nbatch * max_nodes 
    #pad_edges_to = 0 # nbatch * max_edges
    #for graph in graphset:
    #    pad_nodes_to = max(graph.num_nodes, pad_nodes_to)
    #    pad_edges_to = max(graph.num_edges, pad_edges_to)
    #padded_graphset = get_graphset_with_pad(graphset, pad_nodes_to, pad_edges_to)
    padded_graphset = graphset
    loader = DataLoader(padded_graphset,
                        nbatch,
                        shuffle=False,
                        drop_last=True,
                        pin_memory=True,
                        num_workers=0,
                        collate_fn=None)
    return loader, uniq_element, enr_avg_per_element


def get_edge_relative_vectors(data):
    pos = torch.tensor(data.x, dtype=torch.float32)
    b, n, _ = pos.shape
    edges = torch.tensor(data.edge_idx, dtype=torch.float32)
    edges0 = edges[:, 0, :, None].expand(-1, -1, 3).long()
    edges1 = edges[:, 1, :, None].expand(-1, -1, 3).long()
    loc0 = torch.gather(pos, dim=1, index=edges0)
    loc1 = torch.gather(pos, dim=1, index=edges1)

    # Consider PBC
    Sij = torch.tensor(data.edge_attr, dtype=torch.float32).squeeze(1)
    cell = torch.tensor(data.cell, dtype=torch.float32)
    expanded_cell = torch.repeat_interleave(cell, repeats=edges.size(-1), dim=0)
    expanded_cell = expanded_cell.reshape(b, edges.size(-1), 3, 3)
    shift_v = torch.einsum('bni,bnij->bnj', Sij, expanded_cell)
    Rij = loc1 - loc0 + shift_v

    return Rij


def get_relative_vector(atoms, iatoms, jatoms, Sij):
    R = torch.tensor(atoms.get_positions())
    cell = torch.tensor(np.array(atoms.get_cell()))
    Sij = torch.tensor(Sij, dtype=torch.float32)
    shift_v = torch.einsum('ij,kj->ij', Sij, cell)   
    Rij = R[jatoms] - R[iatoms] + shift_v
    dist = torch.norm(Rij, dim=1)
    #print(dist)
    return Rij, dist


def data_to_dict(data):
    data_dict = data.to_dict() if isinstance(data, DataBatch) else data
    data_dict = {k: (torch.tensor(v) if isinstance(v, int) else v) 
                    for k, v in data_dict.items()}
    data_dict = {k: (torch.tensor(v) if isinstance(v, list) else v) 
                    for k, v in data_dict.items()}
    return data_dict


def extract_species(data):
    atoms = read(data, index=0)
    atoms_numbers = atoms.get_atomic_numbers()
    species = torch.unique(torch.tensor(atoms_numbers))

    return species


def apply_along_axis(func1d, axis: int, arr: torch.Tensor):
    num_dims = arr.ndim
    axis = axis % num_dims  # canonicalize

    func = func1d
    for i in range(1, num_dims - axis):
        func = vmap(func, in_dims=i, out_dims=-1)
    for i in range(axis):
        func = vmap(func, in_dims=0, out_dims=0)

    return func(arr)


def find_input_json():
    current_dir = os.getcwd()
    input_json_path = os.path.join(current_dir, 'input.json')
    if os.path.exists(input_json_path):
        return input_json_path
    else:
        return None


def date(fmt="%m/%d/%Y %H:%M:%S"):
    return datetime.now().strftime(fmt)


def on_exit(fout, separator_bottom, n_params, json_data, date1):
    print(separator_bottom, file=fout)
    print(f'\n* NUMBER OF PARAMETERS: ', file=fout)
    print(f' - {"MODEL(TOTAL)":14} {n_params}', file=fout)
    #print(f' -- {"EQUIV. MODEL":13} {interface_n_params}', file=fout)
    #print(f' -- {"BACKBONE":13} {backbone_n_params}', file=fout)
    print(f' --- {"HIDDEN.":12} {json_data["hidden_channels"]}', file=fout)
    print(f' --- {"FEATS. DIM.":12} {json_data["features_dim"]}', file=fout)
    print(f' --- {"RADI. BASIS.":12} {json_data["num_radial_basis"]}', file=fout)
    if json_data.get("nsamples"):
        print(f'\n* NUMBER OF "g" PER DATA:\n   {" ":14} {json_data.get("nsamples")}', file=fout)
    print(f'\n* SEED NUMBER:', file=fout)
    print(f' - {"DATA_SEED":14} {json_data["NN"]["data_seed"]}', file=fout)
    print(f' - {"INIT_SEED":14} {json_data["NN"]["init_seed"]}', file=fout)

    ntrain_val = json_data.get("ntrain")
    nvalid_val = json_data.get("nvalid")
    if ntrain_val is not None and nvalid_val is not None:
        if type(ntrain_val) == str:
            from ase.io import read
            try:
                train = read(ntrain_val, index=slice(None))
                ntrain = len(train)
                valid = read(nvalid_val, index=slice(None))
                nvalid = len(valid)
            except:
                ntrain = 0
                nvalid = 0
        else:
            ntrain = ntrain_val
            nvalid = nvalid_val
        print(f'\n* DATA INFO:\n - {"N(TRAIN)":14} {ntrain}\n - {"N(VALID)":14} {nvalid}', file=fout)
    else:
        print(f'\n* DATA INFO: (multihead mode, see datasets config)', file=fout)
    print(f' - {"BATCH":14} {json_data["nbatch"]}', file=fout)
    print(f' - {"CUTOFF":14} {json_data["cutoff"]}', file=fout)
    print(f' - {"AVG. NEIGH.":14} {json_data["avg_num_neighbors"]}', file=fout)

    date2 = date()
    day, days, hours, minutes, seconds = calculate_time_difference(date1, date2)
    print(f'\n* ELAPSED TIME:', file=fout)
    print(f' - {day}', file=fout)
    print(f' -- {"DAYS":13} {days:<15.9g}', file=fout)
    print(f' -- {"HOURS":13} {hours:<15.9g}', file=fout)
    print(f' -- {"MINUTES":13} {minutes:<15.9g}', file=fout)
    print(f' -- {"SECONDS":13} {seconds:<15.9g}\n', file=fout)
    print(separator_bottom, file=fout)

    print(' ', file=fout)
    pprint.pprint(json_data, stream=fout)
    fout.flush()
    fout.close()


def calculate_time_difference(date1, date2):
    date1 = datetime.strptime(date1, "%m/%d/%Y %H:%M:%S")
    date2 = datetime.strptime(date2, "%m/%d/%Y %H:%M:%S")
    time_diff = date2 - date1
    
    d = time_diff.days
    s = time_diff.seconds
    h = s // 3600
    m = (s % 3600) // 60
    sec = s % 60
    
    day = f'{d} DAYS, {h} HOURS, {m} MINUTES, and {sec} SECONDS'

    days = d + h/24 + m/1440 + sec/86400
    hours = d*24 + h + m/60 + sec/3600
    minutes = d*24*60 + h*60 + m + sec/60
    seconds = d*24*60*60 + h*60*60 + m*60 + sec

    return day, days, hours, minutes, seconds


# =============================================================================
# Coarse-Grained (CG) Data Loading Functions
# =============================================================================

def get_cg_enr_avg_per_type(cg_traj, num_cg_types):
    """
    Calculate average energy per CG bead type.

    Args:
        cg_traj: List of CG data dictionaries with 'energy' and 'types'
        num_cg_types: Number of unique CG bead types

    Returns:
        enr_avg_per_type: Dictionary mapping type_id to average energy
        uniq_type: Dictionary mapping type_id to index
        enr_var: Variance of energies
    """
    from scipy.optimize import minimize

    tgt_enr = np.array([frame['energy'] for frame in cg_traj])

    # Create unique type mapping
    uniq_type = {i: i for i in range(num_cg_types)}

    # Count occurrences of each type per frame
    type_counts = {}
    for type_id in range(num_cg_types):
        type_counts[type_id] = np.array([
            np.sum(frame['types'] == type_id) for frame in cg_traj
        ])

    c0 = np.array([type_counts[i] for i in range(num_cg_types)])

    # Handle case where some types have zero counts
    valid_mask = c0.sum(axis=1) > 0
    if not valid_mask.all():
        print(f"Warning: Some CG types have zero counts, using subset")
        c0_valid = c0[valid_mask]
    else:
        c0_valid = c0

    m0 = tgt_enr.sum() / c0_valid.sum() if c0_valid.sum() > 0 else 0.0
    w0 = np.array([m0 for _ in range(num_cg_types)], dtype=np.float64)

    def loss_fn(weight, count):
        prd_enr = np.einsum('i,ij->j', weight, count)
        diff = tgt_enr - prd_enr
        return (diff * diff).mean()

    results = minimize(loss_fn, x0=w0, args=(c0,), method='BFGS')
    w0 = results.x

    enr_avg_per_type = {i: w0[i] for i in range(num_cg_types)}

    return enr_avg_per_type, uniq_type, np.var(tgt_enr)


def get_graphset_cg(cg_traj, cutoff, uniq_type, enr_avg_per_type,
                            enr_var, regress_forces=True, max_neigh=None,
                            show_progress=False, desc="Converting CG",
                            bond_topology=None):
    from ase import Atoms

    graph_list = []
    iterator = tqdm(cg_traj, desc=desc, leave=False) if show_progress else cg_traj

    for cg_data in iterator:
        positions = cg_data['positions']
        types = cg_data['types']
        cell = cg_data['cell']
        energy = cg_data['energy']
        forces = cg_data.get('forces', np.zeros_like(positions))
        stress_arr = cg_data.get('stress', None)  # Optional, (3,3) eV/A^3 or (6,) Voigt

        # Calculate energy offset
        node_enr_avg = np.array([enr_avg_per_type[uniq_type[t]] for t in types])
        enr = energy - node_enr_avg.sum()

        # Create dummy ASE Atoms for neighbor list calculation
        # Use 'X' as placeholder element for CG beads
        n_sites = len(positions)

        # Detect non-periodic systems (zero cell matrix)
        use_pbc = cell is not None and np.abs(cell).sum() > 1e-6
        original_cell = cell.copy() if cell is not None else np.zeros((3, 3))
        if not use_pbc:
            # Create a large box enclosing all positions with buffer
            pos_min = positions.min(axis=0)
            pos_max = positions.max(axis=0)
            box_size = pos_max - pos_min + 2 * cutoff + 10.0
            nlist_cell = np.diag(box_size)
            # Center positions in box
            positions_shifted = positions - pos_min + cutoff + 5.0
        else:
            nlist_cell = cell
            positions_shifted = positions

        cg_atoms = Atoms(
            symbols=['X'] * n_sites,
            positions=positions_shifted,
            cell=nlist_cell,
            pbc=use_pbc
        )

        # Calculate neighbor list
        iatoms, jatoms, Sij = neighbour_list(
            quantities='ijS',
            atoms=cg_atoms,
            cutoff=cutoff
        )

        num_nodes = n_sites
        num_edges = len(iatoms)

        # Sort neighbors by distance, limit to max_neigh
        if max_neigh is not None and num_edges > 0:
            Rij_vec = positions[jatoms] - positions[iatoms]
            # Apply PBC shift
            shift_v = np.einsum('ij,jk->ik', Sij, cell)
            Rij_vec = Rij_vec + shift_v
            dist = np.linalg.norm(Rij_vec, axis=1)

            nonmax_idx = []
            for i in range(n_sites):
                idx_i = np.where(iatoms == i)[0]
                if len(idx_i) > max_neigh:
                    idx_sorted = idx_i[np.argsort(dist[idx_i])[:max_neigh]]
                else:
                    idx_sorted = idx_i
                nonmax_idx.append(idx_sorted)
            nonmax_idx = np.concatenate(nonmax_idx) if nonmax_idx else np.array([], dtype=int)

            iatoms = iatoms[nonmax_idx]
            jatoms = jatoms[nonmax_idx]
            Sij = Sij[nonmax_idx]
            num_edges = len(iatoms)

        # Build stress tensor for Data object.
        # Model outputs Voigt 6-component: (σ_xx, σ_yy, σ_zz, σ_yz, σ_xz, σ_xy).
        # If given (3, 3) full tensor, convert to Voigt 6-vector here.
        if stress_arr is not None:
            stress_np = np.asarray(stress_arr, dtype=np.float32)
            if stress_np.shape == (3, 3):
                voigt = np.array([
                    stress_np[0, 0],  # xx
                    stress_np[1, 1],  # yy
                    stress_np[2, 2],  # zz
                    stress_np[1, 2],  # yz
                    stress_np[0, 2],  # xz
                    stress_np[0, 1],  # xy
                ], dtype=np.float32)
                stress_tensor = torch.tensor(voigt, dtype=torch.float32)
            elif stress_np.shape == (6,):
                stress_tensor = torch.tensor(stress_np, dtype=torch.float32)
            else:
                stress_tensor = torch.tensor(np.zeros(6), dtype=torch.float32)
        else:
            stress_tensor = torch.tensor(np.zeros(6), dtype=torch.float32)

        # Create graph
        graph = Data(
            positions=torch.tensor(positions, dtype=torch.float32),
            species=torch.tensor(types, dtype=torch.long),
            forces=torch.tensor(forces, dtype=torch.float32),
            edges=torch.tensor(Sij, dtype=torch.float32),
            num_nodes=num_nodes,
            num_edges=num_edges,
            energy=torch.tensor(enr, dtype=torch.float32),
            cell=torch.tensor(original_cell, dtype=torch.float32).view(1, 3, 3),
            edge_index=torch.tensor(np.array([iatoms, jatoms]), dtype=torch.long),
            stress=stress_tensor,
            volume=torch.tensor(np.prod(np.diag(original_cell)) if original_cell is not None else 0.0)
        )

        # Bond flag for CG systems (0=non-bonded, 1=bonded), whole-system positional index
        if bond_topology is not None:
            n_beads_per_mol = bond_topology['n_beads_per_mol']
            bonds_local = bond_topology['bonds']
            edge_bond = np.zeros(num_edges, dtype=np.float32)
            n_mol = n_sites // n_beads_per_mol
            bonded_set = set()
            for m in range(n_mol):
                offset = m * n_beads_per_mol
                for bi, bj in bonds_local:
                    bonded_set.add((offset + bi, offset + bj))
                    bonded_set.add((offset + bj, offset + bi))
            for e in range(num_edges):
                if (int(iatoms[e]), int(jatoms[e])) in bonded_set:
                    edge_bond[e] = 1.0
            graph.edge_bond = torch.tensor(edge_bond, dtype=torch.float32)

        graph_list.append(graph)

    return graph_list


def get_graphset_cg_to_predict(cg_traj, cutoff, uniq_type,
                                regress_forces=True, max_neigh=None):
    """
    Convert CG trajectory data to PyTorch Geometric graph dataset for PREDICTION.

    Unlike get_graphset_cg (for training), this function does NOT subtract energy offset.
    This matches the behavior of get_graphset_to_predict for all-atom models.
    """
    from ase import Atoms

    graph_list = []
    for cg_data in cg_traj:
        positions = cg_data['positions']
        types = cg_data['types']
        cell = cg_data['cell']
        energy = cg_data['energy']
        forces = cg_data.get('forces', np.zeros_like(positions))

        # NOTE: Unlike get_graphset_cg, we do NOT subtract energy offset here.
        enr = energy

        # Create dummy ASE Atoms for neighbor list calculation
        n_sites = len(positions)

        # Detect non-periodic systems (zero cell matrix)
        use_pbc = cell is not None and np.abs(cell).sum() > 1e-6
        original_cell = cell.copy() if cell is not None else np.zeros((3, 3))
        if not use_pbc:
            pos_min = positions.min(axis=0)
            pos_max = positions.max(axis=0)
            box_size = pos_max - pos_min + 2 * cutoff + 10.0
            nlist_cell = np.diag(box_size)
            positions_shifted = positions - pos_min + cutoff + 5.0
        else:
            nlist_cell = cell
            positions_shifted = positions

        cg_atoms = Atoms(
            symbols=['X'] * n_sites,
            positions=positions_shifted,
            cell=nlist_cell,
            pbc=use_pbc
        )

        # Calculate neighbor list
        iatoms, jatoms, Sij = neighbour_list(
            quantities='ijS',
            atoms=cg_atoms,
            cutoff=cutoff
        )

        num_nodes = n_sites
        num_edges = len(iatoms)

        # Sort neighbors by distance, limit to max_neigh
        if max_neigh is not None and num_edges > 0:
            Rij_vec = positions[jatoms] - positions[iatoms]
            shift_v = np.einsum('ij,jk->ik', Sij, cell)
            Rij_vec = Rij_vec + shift_v
            dist = np.linalg.norm(Rij_vec, axis=1)

            nonmax_idx = []
            for i in range(n_sites):
                idx_i = np.where(iatoms == i)[0]
                if len(idx_i) > max_neigh:
                    idx_sorted = idx_i[np.argsort(dist[idx_i])[:max_neigh]]
                else:
                    idx_sorted = idx_i
                nonmax_idx.append(idx_sorted)
            nonmax_idx = np.concatenate(nonmax_idx) if nonmax_idx else np.array([], dtype=int)

            iatoms = iatoms[nonmax_idx]
            jatoms = jatoms[nonmax_idx]
            Sij = Sij[nonmax_idx]
            num_edges = len(iatoms)

        # Map types to species indices
        species = np.array([uniq_type[t] for t in types])

        # Create graph
        graph = Data(
            positions=torch.tensor(positions, dtype=torch.float32),
            species=torch.tensor(species, dtype=torch.long),
            forces=torch.tensor(forces, dtype=torch.float32),
            edges=torch.tensor(Sij, dtype=torch.float32),
            num_nodes=num_nodes,
            num_edges=num_edges,
            energy=torch.tensor(enr, dtype=torch.float32),
            cell=torch.tensor(original_cell, dtype=torch.float32).view(1, 3, 3),
            edge_index=torch.tensor(np.array([iatoms, jatoms]), dtype=torch.long),
            stress=torch.tensor(np.zeros(6), dtype=torch.float32),
            volume=torch.tensor(np.prod(np.diag(original_cell)) if original_cell is not None else 0.0)
        )
        # NOTE: requires_grad_ is set inside model.forward (positions always,
        # cell conditional on compute_stress). Setting it at the graph level
        # breaks PyG collate (torch.cat with out= rejects grad tensors).
        graph_list.append(graph)

    return graph_list


def get_dataloader_cg(fname, cg_mapping_config, ntrain, nvalid,
                      nbatch, cutoff, random_seed,
                      num_cg_types=1, regress_forces=True,
                      max_neigh=None, rank=0, world_size=1,
                      num_workers=4):
    """
    Create DataLoaders for CG training from atomistic trajectory.

    This function:
    1. Loads atomistic trajectory
    2. Converts to CG representation using the mapping
    3. Creates train/valid dataloaders

    Args:
        fname: Path to atomistic trajectory file
        cg_mapping_config: CG mapping configuration dictionary
        ntrain: Number of training samples (or path to train file)
        nvalid: Number of validation samples (or path to valid file)
        nbatch: Batch size
        cutoff: Cutoff distance for CG neighbor list
        random_seed: Random seed for data splitting
        num_cg_types: Number of unique CG bead types
        regress_forces: Whether to train on forces
        max_neigh: Maximum number of neighbors
        rank: Process rank for distributed training
        world_size: Number of processes

    Returns:
        train_loader, valid_loader, uniq_type, enr_avg_per_type
    """
    from .cg_mapping import CGMapping

    msg = ''

    # Load atomistic trajectory
    if isinstance(ntrain, str):
        # Separate train/valid files
        train_data = read(ntrain, index=slice(None))
        valid_data = read(nvalid, index=slice(None))
        msg += 'Number of atomistic frames:\n'
        msg += f'\033[33m -- training      {len(train_data)}\n'
        msg += f' -- validation    {len(valid_data)}\033[0m\n\n'
        traj = train_data + valid_data
    else:
        # Single file with ntrain/nvalid split
        nsamp = ntrain + nvalid
        traj = read(fname, index=slice(None))[-nsamp:]
        torch.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)
        idx = torch.arange(nsamp)
        idx = idx[torch.randperm(nsamp)]
        idx_train = idx[:ntrain]
        idx_valid = idx[ntrain:]
        train_data = [traj[i] for i in idx_train]
        valid_data = [traj[i] for i in idx_valid]
        msg += 'Number of atomistic frames:\n'
        msg += f'\033[33m -- training      {len(train_data)}\n'
        msg += f' -- validation    {len(valid_data)}\033[0m\n\n'

    # Create CG mapping
    # Handle auto-detection case
    if isinstance(cg_mapping_config, dict) and cg_mapping_config.get('auto', False):
        from .cg_mapping import auto_detect_cg_mapping
        if rank == 0:
            print("Auto-detecting CG mapping from trajectory...")
        cg_mapping_config = auto_detect_cg_mapping(train_data)

    mapping = CGMapping(cg_mapping_config)

    # Convert to CG
    if rank == 0:
        print(msg)
        print(f"Converting atomistic data to CG representation...")
        print(f"  - Mapping method: {mapping.method}")
        print(f"  - Atoms per molecule: {mapping.atoms_per_molecule}")
        print(f"  - CG beads per molecule: {mapping.num_cg_sites}")

    from .cg_mapping import convert_trajectory_to_cg
    cg_train = convert_trajectory_to_cg(train_data, mapping, show_progress=(rank == 0))
    cg_valid = convert_trajectory_to_cg(valid_data, mapping, show_progress=(rank == 0))
    cg_traj = cg_train + cg_valid

    if rank == 0:
        n_cg_sites = len(cg_train[0]['positions'])
        print(f"  - CG sites per frame: {n_cg_sites}")

    # Calculate energy averages per CG type
    enr_avg_per_type, uniq_type, enr_var = get_cg_enr_avg_per_type(cg_traj, num_cg_types)

    if rank == 0:
        print(f"\nMean energy per CG type:")
        for t, e in enr_avg_per_type.items():
            print(f"  Type {t}: {e:.4f}")

    # Create graph datasets
    loaders = []
    for dataset, name in [(cg_train, 'train'), (cg_valid, 'valid')]:
        graphset = get_graphset_cg(
            dataset, cutoff, uniq_type, enr_avg_per_type, enr_var,
            regress_forces, max_neigh,
            show_progress=(rank == 0), desc=f"Building {name} graphs"
        )

        # Padding
        pad_nodes_to = max(g.num_nodes for g in graphset)
        pad_edges_to = max(g.num_edges for g in graphset)
        graphset = get_graphset_with_pad(deepcopy(graphset), pad_nodes_to, pad_edges_to)

        # Sampler for distributed training
        data_sampler = None
        if world_size > 1:
            data_sampler = DistributedSampler(
                graphset, num_replicas=world_size, rank=rank
            )

        _nw = max(num_workers, 0)
        loader = DataLoader(
            graphset,
            nbatch,
            shuffle=False,
            drop_last=False,
            pin_memory=True,
            num_workers=_nw,
            persistent_workers=(_nw > 0),
            prefetch_factor=(2 if _nw > 0 else None),
            collate_fn=None,
            sampler=data_sampler
        )
        loaders.append(loader)

    if rank == 0:
        print(f"\nCG DataLoaders created successfully!")
        print(f"  - Train batches: {len(loaders[0])}")
        print(f"  - Valid batches: {len(loaders[1])}")

    return loaders[0], loaders[1], uniq_type, enr_avg_per_type


def get_dataloader_to_predict_cg(fname, ndata, nbatch, cutoff, model_ckpt,
                                  regress_forces=True, max_neigh=None,
                                  num_workers=4):
    """
    Create DataLoader for CG model prediction from NPZ file.

    This function loads a CG NPZ file (with positions, forces, energies, types, cells)
    and creates a DataLoader suitable for CG model evaluation.

    Args:
        fname: Path to CG NPZ file
        ndata: Number of data samples to use (or 'all' for all samples)
        nbatch: Batch size
        cutoff: Cutoff distance for neighbor list
        model_ckpt: Model checkpoint containing uniq_element and enr_avg_per_element
        regress_forces: Whether to include forces in the data
        max_neigh: Maximum number of neighbors per site

    Returns:
        data_loader, uniq_element, enr_avg_per_element
    """
    # Load NPZ file
    data = np.load(fname, allow_pickle=True)

    positions = data['positions']
    forces = data['forces']
    energies = data['energies']
    cells = data['cells']
    types = data['types']

    n_frames = positions.shape[0]

    # Determine number of samples to use
    if isinstance(ndata, str) or ndata is None or ndata == 'all':
        n_samples = n_frames
    else:
        n_samples = min(ndata, n_frames)

    print('number of data:')
    print(f'\033[33m -- test          {n_samples}\033[0m\n')

    # Get uniq_element and enr_avg_per_element from model checkpoint
    uniq_element = model_ckpt['uniq_element']
    enr_avg_per_element = model_ckpt['enr_avg_per_element']

    # Convert NPZ data to cg_traj format
    cg_traj = []
    for i in range(n_samples):
        cg_data = {
            'positions': positions[i],
            'types': types,
            'forces': forces[i] if regress_forces else np.zeros_like(positions[i]),
            'energy': energies[i],
            'cell': cells[i]
        }
        cg_traj.append(cg_data)

    # Create graphset using prediction-specific function (no offset subtraction)
    graphset = get_graphset_cg_to_predict(
        cg_traj, cutoff, uniq_element, regress_forces, max_neigh
    )

    # Create DataLoader
    _nw = max(num_workers, 0)
    loader = DataLoader(
        graphset,
        nbatch,
        shuffle=False,
        drop_last=True,
        pin_memory=True,
        num_workers=_nw,
        persistent_workers=(_nw > 0),
        prefetch_factor=(2 if _nw > 0 else None),
        collate_fn=None
    )

    return loader, uniq_element, enr_avg_per_element

