import torch

import numpy as np
import random

from bam_torch.ga.utils.pa_utils import *
from bam_torch.ga.model.equivariant_layer import EquivariantInterface


@torch.no_grad()
def parse_batch(data, device):
    """
    pos = torch.tensor(data.x, dtype=torch.float32).to(device)
    forces = torch.tensor(data.x_forces, dtype=torch.float32).to(device)
    num_edges = torch.tensor(data.num_edges, dtype=int).to(device)
    b = num_edges.shape[0]
            
    edges = torch.tensor(data.edge_idx, dtype=torch.float32).to(device)
    """
    pos = data.x
    num_edges = data.num_edges
    b = num_edges.shape[0]
    num_edges = num_edges[0]
    b_n, _ = pos.shape
    n = int(b_n / b)
    pos = pos.view(b, n, 3)
    
    Rij = data.Rij 
    loc_dist = data.distance 
    Rij = Rij.view(b, num_edges, 3)
    loc_dist = loc_dist.view(b, 1, num_edges).float()
    iatoms = data.senders
    jatoms = data.receivers
    iatoms = iatoms.view(b, 1, num_edges).long()
    jatoms = jatoms.view(b, 1, num_edges).long()

    edges = torch.cat([iatoms, jatoms], dim=1).long()
    edge_attr = torch.cat([edges, loc_dist], dim=1)
    edge_attr = edge_attr.transpose(1, 2)

    species = data.x_species
    species = species.view(b, n, 1)
    species = species[:, :, :, None].expand(-1, -1, -1, 3).view(b, n, 3)
    #node_features = torch.cat([pos, forces], dim=-1)
    node_features = torch.cat([pos, species], dim=-1)
    # node_features = torch.cat([pos, forces, species.unsqueeze(-1)], dim=-1)
    node_features = node_features.view(b, n, 2, 3)    
    node_features = node_features.transpose(-1, -2)
    assert (pos - node_features[:, :, :, 0]).abs().sum().item() == 0

    idx = torch.tensor([i for i in range(b)], device=device)
    edge_features = torch.zeros(b, n, n, edge_attr.size(-1), device=device) 
    batch_idxs = torch.arange(b, device=device).repeat_interleave(num_edges).long()
    edge_features[batch_idxs, edges[:, 0, :].flatten().long(), edges[:, 1, :].flatten().long(), :] \
        = edge_attr.reshape(-1, edge_attr.size(-1))

    return node_features, edge_features, edges, idx

def parse_translation(node_features):
    pos, frc = node_features.unbind(-1)
    _, pos_center = torch.std_mean(pos, dim=-1, keepdim=True)
    node_features = torch.stack([pos - pos_center, frc], dim=-1)

    return pos, pos_center, node_features

def get_representations(equiv_model, data_loader, n_samples, device):
    gs_list = []
    for data in data_loader:
        data = data.to(device)
        #data.x.requires_grad_(True)
        ## 1) Parse batched-data
        node_features, edge_features, edge_idx, idx = parse_batch(data, device=device)
        b, n, _, d_node = node_features.shape
        b, n, n, d_edge = edge_features.shape
        _, _, n_edges = edge_idx.shape
        assert node_features.shape == (b, n, 3, d_node)
        assert edge_features.shape == (b, n, n, d_edge)
        assert edge_idx.shape == (b, 2, n_edges)
        assert idx.shape == (b,)
        ## Handle residual component: Translational transform
        loc_input, loc_center, node_features = parse_translation(node_features)
        ## 2) Sample from p(g|x)
        gs = equiv_model(node_features, edge_features, idx, n_samples).detach().to('cpu').numpy()
        gs_list.append(gs)
    #gs_list = torch.stack(gs_list)
    return gs_list


def get_gs_list(json_data):
    # reproducibility
    rng_seed = json_data['NN']['data_seed']
    random.seed(rng_seed)
    np.random.seed(rng_seed)
    torch.manual_seed(rng_seed)
    torch.cuda.manual_seed_all(rng_seed)
    torch.use_deterministic_algorithms(True)
    os.environ["CUBLAS_WORKSPACE_CONFIG"]=":16:8"
    
    # Configure device
    if json_data['device'] == 'cpu':
        device = 'cpu'
        print(f'\ndevice:\n\033[33m -- {device}\033[0m')
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f'\ndevice:\n\033[33m -- {device}\033[0m')


    # Configure data
    train_loader, test_loader, _, _ = get_dataloader(
                                            json_data['fname_traj'],
                                            json_data['ntrain'],
                                            json_data['nvalid'],
                                            json_data['nbatch'],
                                            json_data['cutoff'],
                                            rng_seed,
                                            json_data['element']
                                        )
    

    # Configure model
    avg_num_neighbors = json_data['avg_num_neighbors']
    nsamples = json_data['nsamples']

    equiv_model = EquivariantInterface(
                        symmetry='O3',
                        interface='prob',
                        fixed_noise=False,
                        noise_scale=1,
                        tau=0.01,
                        hard=True,
                        vnn_dropout=0.1,
                        vnn_hidden_dim =96,
                        vnn_k_nearest_neighbors=avg_num_neighbors
                    ).to(device)

    interface_n_params = sum(p.numel() for p in equiv_model.parameters() if p.requires_grad)
    print(f'\nnumber of parameters:\n\033[36m -- interface {interface_n_params}\033[0m\n')

    train_gs = get_representations(equiv_model, train_loader, nsamples, device)
    valid_gs = get_representations(equiv_model, test_loader, nsamples, device)

    train_np = np.array(train_gs)
    valid_np = np.array(valid_gs)
    interface_n_params = np.array(interface_n_params)

    np.save('train_gs.npy', train_np)
    np.save('valid_gs.npy', valid_np)
    np.save('interface_n_params.npy', interface_n_params)

def get_test_gs_list(json_data):
    # reproducibility
    rng_seed = json_data['NN']['data_seed']
    random.seed(rng_seed)
    np.random.seed(rng_seed)
    torch.manual_seed(rng_seed)
    torch.cuda.manual_seed_all(rng_seed)
    torch.use_deterministic_algorithms(True)
    os.environ["CUBLAS_WORKSPACE_CONFIG"]=":16:8"
    
    # Configure device
    if json_data['device'] == 'cpu':
        device = 'cpu'
        print(f'\ndevice:\n\033[33m -- {device}\033[0m')
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f'\ndevice:\n\033[33m -- {device}\033[0m')


    # Configure data
    evaluate_config = json_data['predict']
    #evaluate = evaluate_config.get('evaluate_tag')
    model_ckpt = torch.load(evaluate_config["model"], map_location=device)
    data_loader, uniq_element, enr_avg_per_element = \
                                get_dataloader_to_predict(
                                    json_data["predict"]['fname_traj'],
                                    json_data["predict"]['ndata'],
                                    1,  # nbatch
                                    json_data['cutoff'],
                                    model_ckpt,
                                )
    

    # Configure model
    avg_num_neighbors = json_data['avg_num_neighbors']
    nsamples = json_data['nsamples']

    equiv_model = EquivariantInterface(
                        symmetry='O3',
                        interface='prob',
                        fixed_noise=False,
                        noise_scale=1,
                        tau=0.01,
                        hard=True,
                        vnn_dropout=0.1,
                        vnn_hidden_dim =96,
                        vnn_k_nearest_neighbors=avg_num_neighbors
                    ).to(device)

    interface_n_params = sum(p.numel() for p in equiv_model.parameters() if p.requires_grad)
    print(f'\nnumber of parameters:\n\033[36m -- interface {interface_n_params}\033[0m\n')

    test_gs = get_representations(equiv_model, data_loader, nsamples, device)

    train_np = np.array(test_gs)
    interface_n_params = np.array(interface_n_params)

    np.save('test_gs.npy', test_gs)
    np.save('interface_n_params_test.npy', interface_n_params)

if __name__ == '__main__':
    print(date()) 
    print("\nGetting the matrix list of group representation from data by Equivariant Model ... ")
    input_json_path = find_input_json()

    with open(input_json_path) as f:
        json_data = json.load(f)

        if json_data['trainer'] in ['base']:
            get_gs_list(json_data)
        else:
            print('we are making')

    print(date())
