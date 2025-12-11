import numpy as np
from ase.io import read
from scipy.optimize import minimize
from matscipy.neighbours import neighbour_list
from tqdm import tqdm

import torch
from torch import vmap
from torch_geometric.data import Data
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
            volume = atoms.get_volume()
            stress = np.zeros(6)
        else:
            frc = np.zeros((len(atoms), 3))

        cell = atoms.get_cell()
        if np.all(cell == [0.0, 0.0, 0.0]):
            cell = np.diag([30., 30., 30.])
            atoms.set_cell(cell)
        
        if 'stress' in atoms._calc.results.keys():
            stress = atoms.get_stress()
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
            energy=torch.tensor(enr, dtype=torch.float32),
            cell=torch.tensor(np.array(cell), dtype=torch.float32).view(1, 3, 3),
            edge_index=torch.tensor(np.array([iatoms, jatoms]), dtype=torch.long),  # senders, recerivers
            stress=torch.tensor(stress, dtype=torch.float32),
            volume=torch.tensor(volume)
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
        pad_nodes_to = 0 # nbatch * max_nodes 
        pad_edges_to = 0 # nbatch * max_edges
        for graph in graphset:
            pad_nodes_to = max(graph.num_nodes, pad_nodes_to)
            pad_edges_to = max(graph.num_edges, pad_edges_to)
        #graphset = get_graphset_with_pad(deepcopy(graphset), pad_nodes_to, pad_edges_to)
        #padded_graphset = graphset
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
            # No foundation → compute E0s using original method
            enr_avg_per_element, uniq_element, enr_var = get_enr_avg_per_element(traj, element)
            if rank == 0:
                print(f"  - Calculated new enr_avg_per_element (no foundation)")
        
        if rank == 0:
            print(f"  - Elements: {element}")
        
        # Save the first head's uniq_element / enr_avg_per_element as global
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
    
    # Shuffle train graphs and create DataLoaders
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
        print(f"  - Batch size: {nbatch}")
    
    return train_loader, valid_loader, global_uniq_element, global_enr_avg_per_element


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
        print(f'N_test: {len(traj)}\n')
    else: 
        traj = read(fname, index=slice(None))[:ndata]
        print(f'N_test: {len(traj)}\n')

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

    # ntrain/nvalid가 없을 수 있음 (multihead 등)
    ntrain_val = json_data.get("ntrain")
    nvalid_val = json_data.get("nvalid")
    if ntrain_val is not None and nvalid_val is not None:
        if type(ntrain_val) == str:
            from ase.io import read
            train = read(ntrain_val, index=slice(None))
            ntrain = len(train)
            valid = read(nvalid_val, index=slice(None))
            nvalid = len(valid)
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


