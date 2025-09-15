import numpy as np
from scipy.optimize import minimize
from matscipy.neighbours import neighbour_list
from ase.io import read, Trajectory

import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from datetime import datetime
import os
import json



def batched_gram_schmidt_3d(bvv):
    assert bvv.ndim == 3
    assert bvv.shape[1] == bvv.shape[2] == 3

    def projection(bu, bv):
        return (bv * bu).sum(-1, keepdim=True) / (bu * bu).sum(-1, keepdim=True) * bu

    buu = torch.zeros_like(bvv)
    buu[:, :, 0] = bvv[:, :, 0].clone()

    # k = 1 start
    bv1 = bvv[:, :, 1].clone()
    bu1 = torch.zeros_like(bv1)
    # j = 0
    bu0 = buu[:, :, 0].clone()
    bu1 = bu1 + projection(bu0, bv1)
    # k = 1 end
    buu[:, :, 1] = bv1 - bu1

    # k = 2 start
    bv2 = bvv[:, :, 2].clone()
    bu2 = torch.zeros_like(bv2)
    # j = 0
    bu0 = buu[:, :, 0].clone()
    bu2 = bu2 + projection(bu0, bv2)
    # j = 1
    bu1 = buu[:, :, 1].clone()
    bu2 = bu2 + projection(bu1, bv2)
    # k = 2 end
    buu[:, :, 2] = bv2 - bu2

    # normalize
    buu = torch.nn.functional.normalize(buu, dim=1)
    return buu


def get_enr_avg_per_element (traj, element):

    tgt_enr = np.array([atoms.get_potential_energy()
                    for atoms in traj])
    uniq_element = {int(e): i for i, e in enumerate(element)}
    element_counts = {i: np.array([ (atoms.numbers == e).sum()
                                   for atoms in traj])
                                for e, i in uniq_element.items()}
    c0 = np.array ([element_counts[i] for i in element_counts.keys()])
    m0 = tgt_enr.sum()/c0.sum()
    w0 = np.array ([m0 for _ in element])

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


def get_relative_vector(atoms, iatoms, jatoms, Sij):
    R = torch.tensor(atoms.get_positions())
    cell = torch.tensor(np.array(atoms.get_cell()))
    Sij = torch.tensor(Sij, dtype=torch.float32)
    shift_v = torch.einsum('ij,kj->ij', Sij, cell)   
    Rij = R[jatoms] - R[iatoms] + shift_v
    dist = torch.norm(Rij, dim=1)

    return Rij, dist

def get_graphset(data, cutoff, uniq_element, enr_avg_per_element, enr_var):
    graph_list = []
    for atoms in data:
        crds = atoms.get_positions()
        node_enr_avg = np.array([enr_avg_per_element[uniq_element[iz]]
                                  for iz in atoms.numbers])
        #enr = (atoms.get_potential_energy() - node_enr_avg.sum()) / enr_var
        enr = atoms.get_potential_energy() - node_enr_avg.sum()
        frc = atoms.get_forces()
        cell = atoms.get_cell()
        volume = atoms.get_volume()
        stress = np.zeros(6)
        if 'stress' in atoms._calc.results.keys():
            stress = atoms.get_stress()
    
        iatoms, jatoms, Sij = neighbour_list(quantities='ijS',
                                             atoms=atoms,
                                             cutoff=cutoff)
 
        species = np.array([uniq_element[iz] for iz in atoms.numbers])
        num_nodes = crds.shape[0]
        num_edges = iatoms.shape[0]

        Rij, dist = get_relative_vector(atoms, iatoms, jatoms, Sij)

        # Generate Graph data set
        graph = Data(
                    x=torch.tensor(crds, dtype=torch.float32),   # node features
                    x_species=torch.tensor(species, dtype=torch.float32),
                    x_forces=torch.tensor(frc, dtype=torch.float32),
                    edge_idx=torch.tensor(np.array([iatoms, jatoms]), dtype=torch.long), # To avoid automatic index adjustment 
                                             # senders and receivers
                    edge_attr=torch.tensor(Sij, dtype=torch.float32),# edge features
                    num_nodes=num_nodes,             
                    num_edges=num_edges,
                    y=torch.tensor(enr, dtype=torch.float32),
                    cell=torch.tensor(np.array(cell), dtype=torch.float32),
                    senders=torch.tensor(iatoms, dtype=torch.long),
                    receivers=torch.tensor(jatoms, dtype=torch.long),
                    Rij=Rij,
                    distance=dist
                )
        graph_list.append(graph)

    return graph_list


def get_graphset_with_pad(graphset, pad_nodes_to, pad_edges_to):
    graph_list = []
    for data in graphset:
        num_nodes = data.num_nodes
        original_num_nodes = num_nodes
        if num_nodes < pad_nodes_to:
            padding = torch.zeros((pad_nodes_to - num_nodes, data.x.size(1)))
            data.x = torch.cat([data.x, padding], dim=0)
            data.num_nodes = pad_nodes_to
            node_mask = torch.cat([torch.ones(original_num_nodes), 
                                   torch.zeros(pad_nodes_to - original_num_nodes)])
        else:
            node_mask = torch.ones(num_nodes)   
        # pad edges (attr)
        num_edges = data.num_edges
        original_num_edges = num_edges
        if num_edges < pad_edges_to:
            edge_attr_padding = torch.zeros((pad_edges_to - num_edges, 3))
            send_reci_padding = torch.zeros((pad_edges_to - num_edges))
            edge_index_padding = torch.zeros((2, pad_edges_to - num_edges))

            data.edge_attr = torch.cat([data.edge_attr, edge_attr_padding], dim=0)
            data.edge_idx = torch.cat([data.edge_idx, edge_index_padding], dim=1)
            data.receivers = torch.cat([data.receivers, send_reci_padding.long()], dim=0)
            data.senders = torch.cat([data.senders, send_reci_padding.long()], dim=0)
            data.Rij = torch.cat([data.Rij, edge_attr_padding], dim=0)
            data.distance = torch.cat([data.distance, send_reci_padding], dim=0)

            data.num_edges = pad_edges_to
            edge_mask = torch.cat([torch.ones(original_num_edges), 
                                   torch.zeros(pad_edges_to - original_num_edges)])
        else:
            edge_mask = torch.ones(num_edges)
        data.node_mask = node_mask
        data.edge_mask = edge_mask
        
        graph_list.append(data)
    return graph_list


def get_dataloader(fname, ntrain, ntest, 
                   nbatch, cutoff, random_seed, element=None):
    if type(ntrain) == str: 
        train_data = read(ntrain, index=slice(None))
        test_data = read(ntest, index=slice(None))
        print(f'\nntrain: {len(train_data)} | ntest: {len(test_data)}\n')
        traj = train_data + test_data
    else:
        nsamp = ntrain + ntest
        traj = read(fname, index=slice(None))[-nsamp:]
        torch.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)
        idx = torch.arange(nsamp)
        idx = idx[torch.randperm(nsamp)] 
        idx_train = idx[:ntrain]
        idx_test = idx[ntrain:]   
        train_data = [traj[i] for i in idx_train]
        test_data = [traj[i] for i in idx_test]

    if element == None or element == 'auto':
        element = sorted(
            list(set(atom.number for atoms in traj 
                                  for atom in atoms))
        )  # traj: ase.Atoms
    enr_avg_per_element, uniq_element, enr_var = get_enr_avg_per_element (traj, element)    
    loaders = []
    for dataset in [train_data, test_data]:
        graphset = get_graphset(dataset, cutoff, uniq_element, 
                                enr_avg_per_element, enr_var)
        pad_nodes_to = 0 # nbatch * max_nodes 
        pad_edges_to = 0 # nbatch * max_edges
        for graph in graphset:
            pad_nodes_to = max(graph.num_nodes, pad_nodes_to)
            pad_edges_to = max(graph.num_edges, pad_edges_to)
        padded_graphset = get_graphset_with_pad(graphset, pad_nodes_to, pad_edges_to)
# padded_graphset = [ensure_tensor_attributes(data) for data in padded_graphset]
        loader = DataLoader(padded_graphset,
                            nbatch,
                            shuffle=False,
                            drop_last=True,
                            pin_memory=True,
                            num_workers=0,
                collate_fn=None)
        loaders.append(loader)
        # train_loader, test_loader
    return loaders[0], loaders[1], uniq_element, enr_avg_per_element  


def get_edge_relative_vectors(data):
    pos = torch.tensor(data.x, dtype=torch.float32)
    #R = pos / cutoff
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


def get_graphset_to_predict(data, cutoff, uniq_element):
    graph_list = []
    for atoms in data:
        enr = atoms.get_potential_energy()
        frc = atoms.get_forces()
        crds = atoms.get_positions()
        cell = atoms.get_cell()
        volume = atoms.get_volume()
        stress = np.zeros(6)
        if 'stress' in atoms._calc.results.keys():
            stress = atoms.get_stress()
    
        iatoms, jatoms, Sij = neighbour_list(quantities='ijS',
                                             atoms=atoms,
                                             cutoff=cutoff)
 
        species = np.array([uniq_element[iz] for iz in atoms.numbers])
        num_nodes = crds.shape[0]
        num_edges = iatoms.shape[0]

        Rij, dist = get_relative_vector(atoms, iatoms, jatoms, Sij)
        # Generate Graph data set
        graph = Data(
                    x=torch.tensor(crds, dtype=torch.float32),   # node features
                    x_species=torch.tensor(species, dtype=torch.float32),
                    x_forces=torch.tensor(frc, dtype=torch.float32),
                    edge_idx=torch.tensor(np.array([iatoms, jatoms]), dtype=torch.long), # To avoid automatic index adjustment 
                                             # senders and receivers
                    edge_attr=torch.tensor(Sij, dtype=torch.float32),# edge features
                    num_nodes=num_nodes,             
                    num_edges=num_edges,
                    y=torch.tensor(enr, dtype=torch.float32),
                    cell=torch.tensor(np.array(cell), dtype=torch.float32),
                    senders=torch.tensor(iatoms, dtype=torch.long),
                    receivers=torch.tensor(jatoms, dtype=torch.long),
                    Rij=Rij,
                    distance=dist
                )
        graph_list.append(graph)

    return graph_list


def get_dataloader_to_predict(fname, ndata, nbatch, 
                              cutoff, model_ckpt):
    if type(ndata) == str:
        traj = read(fname, index=slice(None))
        print(f'\nN_predict: {len(traj)}\n')
    else: 
        traj = read(fname, index=slice(None))[:ndata]
        print(f'\nN_predict: {len(traj)}\n')

    uniq_element = model_ckpt['uniq_element']
    enr_avg_per_element = model_ckpt['enr_avg_per_element']

    graphset = get_graphset_to_predict(traj, cutoff, uniq_element)
    pad_nodes_to = 0 # nbatch * max_nodes 
    pad_edges_to = 0 # nbatch * max_edges
    for graph in graphset:
        pad_nodes_to = max(graph.num_nodes, pad_nodes_to)
        pad_edges_to = max(graph.num_edges, pad_edges_to)
    padded_graphset = get_graphset_with_pad(graphset, pad_nodes_to, pad_edges_to)
# padded_graphset = [ensure_tensor_attributes(data) for data in padded_graphset]
    loader = DataLoader(padded_graphset,
                        nbatch,
                        shuffle=False,
                        drop_last=True,
                        pin_memory=True,
                        num_workers=0,
            collate_fn=None)
    return loader, uniq_element, enr_avg_per_element


def date(fmt="%m/%d/%Y %H:%M:%S"):
    return datetime.now().strftime(fmt)

def find_input_json():
    current_dir = os.getcwd()
    input_json_path = os.path.join(current_dir, 'input.json')
    if os.path.exists(input_json_path):
        return input_json_path
    else:
        return None

def ensure_tensor_attributes(data):
    for key, value in data:
        if isinstance(value, list): 
            print(" ### list ", key, value)
            setattr(data, key, torch.tensor(value))
    return data

def on_exit(fout, separator_bottom, n_params, 
            backbone_n_params, interface_n_params, json_data,
            date1):
    print(separator_bottom, file=fout)
    print(f'\n* NUMBER OF PARAMETERS: ', file=fout)
    print(f' - {"MODEL(TOTAL)":14} {n_params}', file=fout)
    print(f' -- {"EQUIV. MODEL":13} {interface_n_params}', file=fout)
    print(f' -- {"BACKBONE":13} {backbone_n_params}', file=fout)
    print(f' --- {"HIDDEN.":12} {json_data["hidden_channels"]}', file=fout)
    print(f' --- {"FEATS. DIM.":12} {json_data["features_dim"]}', file=fout)
    print(f' --- {"RADI. BASIS.":12} {json_data["num_radial_basis"]}', file=fout)
    print(f'\n* NUMBER OF "g" PER DATA:\n   {" ":14} {json_data["nsamples"]}', file=fout)
    print(f'\n* SEED NUMBER:', file=fout)
    print(f' - {"DATA_SEED":14} {json_data["NN"]["data_seed"]}', file=fout)
    print(f' - {"INIT_SEED":14} {json_data["NN"]["init_seed"]}', file=fout)

    if type(json_data["ntrain"]) == str:
        from ase.io import read
        train = read(json_data["ntrain"], index=slice(None))
        ntrain = len(train)
        test = read(json_data["ntest"], index=slice(None))
        ntest = len(test)
    else:
        ntrain = json_data["ntrain"]
        ntest = json_data["ntest"]
    print(f'\n* DATA INFO:\n - {"N(TRAIN)":14} {ntrain}\n - {"N(TEST)":14} {ntest}', file=fout)
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
