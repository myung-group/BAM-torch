import random
from copy import deepcopy
from itertools import product
from bam_torch.ga.utils.fa_utils import RandomRotate

import torch
import numpy as np


def compute_frames(
    eigenvec, pos, cell, fa_method="stochastic", pos_3D=None, det_index=0
):
    """Compute all `frames` for a given graph, i.e. all possible
    canonical representations of the 3D graph (of all euclidean transformations).

    Args:
        eigenvec (tensor): eigenvectors matrix # (3, 3)
        pos (tensor): centered position vector
        cell (tensor): cell direction (dxd)
        fa_method (str): the Frame Averaging (FA) inspired technique
            chosen to select frames: stochastic-FA (stochastic), deterministic-FA (det),
            Full-FA (all) or SE(3)-FA (se3).
        pos_3D: for 2D FA, pass atoms' 3rd position coordinate.

    Returns:
        (list): 3D position tensors of projected representation
    """
    # pos.shape == nbatch*num_nodes, 3
    # eigenvec.shape == 3, 3
    dim = pos.shape[1]  # to differentiate between 2D or 3D case
    plus_minus_list = list(product([1, -1], repeat=dim))
    plus_minus_list = [torch.tensor(x) for x in plus_minus_list]
    if fa_method == "prob":
        plus_minus_list = plus_minus_list[0]
    all_fa_pos = []
    all_cell = []
    all_rots = []
    assert fa_method in {
        "all",
        "stochastic",
        "det",
        "se3-all",
        "se3-stochastic",
        "se3-det",
        "prob"  ### 
    }
    se3 = fa_method in {
        "se3-all",
        "se3-stochastic",
        "se3-det",
    }
    #fa_cell = deepcopy(cell)
    fa_cell = cell

    if fa_method == "det" or fa_method == "se3-det":
        sum_eigenvec = torch.sum(eigenvec, axis=0)
        plus_minus_list = [torch.where(sum_eigenvec >= 0, 1.0, -1.0)]

    for pm in plus_minus_list:
        # Append new graph positions to list
        pm = pm.to(eigenvec.device)
        new_eigenvec = pm * eigenvec
        # Consider frame if it passes above check
        fa_pos = pos @ new_eigenvec

        if pos_3D is not None:
            full_eigenvec = torch.eye(3)
            fa_pos = torch.cat((fa_pos, pos_3D.unsqueeze(1)), dim=1)
            full_eigenvec[:2, :2] = new_eigenvec
            new_eigenvec = full_eigenvec

        if cell is not None:
            fa_cell = cell @ new_eigenvec

        # Check if determinant is 1 for SE(3) case
        if se3 and not torch.allclose(
            torch.linalg.det(new_eigenvec), torch.tensor(1.0), atol=1e-03
        ):
            continue

        all_fa_pos.append(fa_pos)
        all_cell.append(fa_cell)
        all_rots.append(new_eigenvec.unsqueeze(0))

    # Handle rare case where no R is positive orthogonal
    if all_fa_pos == []:
        all_fa_pos.append(fa_pos)
        all_cell.append(fa_cell)
        all_rots.append(new_eigenvec.unsqueeze(0))

    # Return frame(s) depending on method fa_method
    if fa_method == "all" or fa_method == "se3-all":
        return all_fa_pos, all_cell, all_rots

    elif fa_method == "det" or fa_method == "se3-det":
        return [all_fa_pos[det_index]], [all_cell[det_index]], [all_rots[det_index]]

    index = random.randint(0, len(all_fa_pos) - 1)
    return [all_fa_pos[index]], [all_cell[index]], [all_rots[index]]


def probablistic_averaging_3D_(data, gs=None, fa_method="prob", check=False):
    
    b, k, _, _ = gs.shape
    b_n, _ = data.positions.shape
    n = int(b_n/b)
    device = data.positions.device

    pos = data.positions.view(b, n, 3)
    pos = pos[:, None, :, :].expand(b, k, n, 3)
    cell = data.cell[:, None, :, :].expand(b, k, 3, 3)
    _, center = torch.std_mean(pos, dim=2, keepdim=True)
    pos = pos - center

    # if symmetry == 'O3' or 'SO3'
    print(gs.shape)
    print(gs)
    gs = torch.from_numpy(gs).to(device)
    gs_inv = gs.transpose(2, 3).to(device)
    pa_pos = torch.einsum('bkij,bkjt->bkit', pos, gs_inv)
    pa_pos = pa_pos.view(b*k, n, 3)
    pa_pos = pa_pos.view(b*k*n, 3)
    pa_cell = torch.einsum('bkij,bkjt->bkit', cell, gs_inv)

    cell_offsets = data.edges # cell_offsets (Sij)
    cell_offsets = cell_offsets.view(b, -1, 3)
    b, e, _ = cell_offsets.shape
    cell_offsets = cell_offsets[:, None, :, :].expand(b, k, e, 3)
    cell_offsets = cell_offsets.reshape(b*k, e, 3)
    data.cell_offsets = cell_offsets.view(b*k*e, 3)

    data.fa_rot = [gs_inv.view(b*k, 3, 3)]
    data.fa_pos = [pa_pos]
    data.fa_cell = [pa_cell.view(b*k, 3, 3)]

    #iatoms = data.senders
    #jatoms = data.receivers
    #iatoms = iatoms.reshape(b, 1, e).long()
    #jatoms = jatoms.reshape(b, 1, e).long()
    #edge_idx = torch.cat([jatoms, iatoms], dim=1).long()
    edge_idx = data.edge_index
    edge_idx = edge_idx[:, None, :, :].expand(b, k, 2, e)
    edge_idx = edge_idx.reshape(b*k, 2, e)

    incr = n * torch.arange(b*k, device=device)
    incr_edge_idx = edge_idx + incr[:, None, None]
    pyg_edge_idx = incr_edge_idx.transpose(0, 1).reshape(2, b*k*e).long()
    data.pa_edge_index = pyg_edge_idx

    return data

def probablistic_averaging_3D(data, equiv_model, n_samples, fa_method="prob", check=False):
    pos = data.positions


    node_features, edge_features, edge_idx, idx = parse_batch(data, device=pos.device)
    b, n, _, d_node = node_features.shape
    b, n, n, d_edge = edge_features.shape
    _, _, n_edges = edge_idx.shape
    assert node_features.shape == (b, n, 3, d_node)
    assert edge_features.shape == (b, n, n, d_edge)
    assert edge_idx.shape == (b, 2, n_edges)
    assert idx.shape == (b,)
    ## Handle residual component: Translational transform
    #loc_input, loc_center, node_features = parse_translation(node_features)

    ## 2) Sample from p(g|x)
    gs_list = equiv_model(node_features, edge_features, idx, n_samples)
    #gs_list = equiv_model(data, n_samples)
    # Shape: b, k, 3, 3
    #pos = data.positions
    pos = pos - pos.mean(dim=0, keepdim=True)
    # Compute fa_pos
    #gs_list = torch.from_numpy(gs).to(pos.device)
    
    all_fa_pos = []
    all_cell = []
    all_rots = []

    fa_cell = data.cell
    cell = data.cell
    pos_3D = None
    nbatch, _, _ = cell.shape
    b_n, _ = pos.shape
    n = int(b_n/nbatch)
    fa_pos = pos.view(nbatch, n, 3)
    #print("fa_pos.shape", fa_pos.shape)
    #print("gs_list", gs_list) 
    #print("cell", cell.shape)
    """
    for b in range(nbatch):

        for gs in gs_list[b]:
            # Append new graph positions to list
            rot_pos = fa_pos[b] @ gs.T

            if cell is not None:
                rot_cell = fa_cell[b] @ gs

            all_fa_pos.append(rot_pos)
            all_cell.append(rot_cell)
            all_rots.append(gs.unsqueeze(0))
    all_fa_pos = torch.cat(all_fa_pos, dim=0)
    all_cell = torch.cat(all_cell, dim=0)
    """
    fa_pos_exp = fa_pos[:, None, :, :]  
    fa_cell_exp = fa_cell[:, None, :, :]
    fa_pos_exp = torch.matmul(fa_pos_exp, gs_list.transpose(-1, -2))
    fa_cell_exp = torch.matmul(fa_cell_exp, gs_list.transpose(-1, -2))
    all_fa_pos = fa_pos_exp.permute(1, 0, 2, 3).contiguous().view(n_samples, b_n, 3)
    all_cell = fa_cell_exp.permute(1, 0, 2, 3).contiguous()
    all_rots = gs_list.permute(1, 0, 2, 3) #.contiguous().view(n_samples, 3, 3)
    # No need to update distances, they are preserved.
    #print(all_fa_pos.shape)
    return all_fa_pos, all_cell, all_rots


def check_constraints(eigenval, eigenvec, dim=3):
    """Check that the requirements for frame averaging are satisfied

    Args:
        eigenval (tensor): eigenvalues
        eigenvec (tensor): eigenvectors
        dim (int): 2D or 3D frame averaging
    """
    # Check eigenvalues are different
    if dim == 3:
        if (eigenval[1] / eigenval[0] > 0.90) or (eigenval[2] / eigenval[1] > 0.90):
            print("Eigenvalues are quite similar")
    else:
        if eigenval[1] / eigenval[0] > 0.90:
            print("Eigenvalues are quite similar")

    # Check eigenvectors are orthonormal
    if not torch.allclose(eigenvec @ eigenvec.T, torch.eye(dim), atol=1e-03):
        print("Matrix not orthogonal")

    # Check determinant of eigenvectors is 1
    if not torch.allclose(torch.linalg.det(eigenvec), torch.tensor(1.0), atol=1e-03):
        print("Determinant is not 1")


def frame_averaging_3D(pos, cell=None, fa_method="stochastic", check=False):
    """Computes new positions for the graph atoms using
    frame averaging, which itself builds on the PCA of atom positions.
    Base case for 3D inputs.

    Args:
        pos (tensor): positions of atoms in the graph
        cell (tensor): unit cell of the graph. None if no pbc.
        fa_method (str): FA method used
            (stochastic, det, all, se3-all, se3-det, se3-stochastic)
        check (bool): check if constraints are satisfied. Default: False.

    Returns:
        (tensor): updated atom positions
        (tensor): updated unit cell
        (tensor): the rotation matrix used (PCA)
    """

    # Compute centroid and covariance
    pos = pos - pos.mean(dim=0, keepdim=True)
    C = torch.matmul(pos.t(), pos)

    # Eigendecomposition
    eigenval, eigenvec = torch.linalg.eigh(C)

    # Sort, if necessary
    idx = eigenval.argsort(descending=True)
    eigenvec = eigenvec[:, idx]
    eigenval = eigenval[idx]

    # Check if constraints are satisfied
    if check:
        check_constraints(eigenval, eigenvec, 3)

    # Compute fa_pos
    fa_pos, fa_cell, fa_rot = compute_frames(eigenvec, pos, cell, fa_method)
    # No need to update distances, they are preserved.

    return fa_pos, fa_cell, fa_rot


def frame_averaging_2D(pos, cell=None, fa_method="stochastic", check=False):
    """Computes new positions for the graph atoms using
    frame averaging, which itself builds on the PCA of atom positions.
    2D case: we project the atoms on the plane orthogonal to the z-axis.
    Motivation: sometimes, the z-axis is not the most relevant one (e.g. fixed).

    Args:
        pos (tensor): positions of atoms in the graph
        cell (tensor): unit cell of the graph. None if no pbc.
        fa_method (str): FA method used (stochastic, det, all, se3)
        check (bool): check if constraints are satisfied. Default: False.

    Returns:
        (tensor): updated atom positions
        (tensor): updated unit cell
        (tensor): the rotation matrix used (PCA)
    """

    # Compute centroid and covariance
    pos_2D = pos[:, :2] - pos[:, :2].mean(dim=0, keepdim=True)
    C = torch.matmul(pos_2D.t(), pos_2D)

    # Eigendecomposition
    eigenval, eigenvec = torch.linalg.eigh(C)
    # Sort eigenvalues
    idx = eigenval.argsort(descending=True)
    eigenval = eigenval[idx]
    eigenvec = eigenvec[:, idx]

    # Check if constraints are satisfied
    if check:
        check_constraints(eigenval, eigenvec, 3)

    # Compute all frames
    fa_pos, fa_cell, fa_rot = compute_frames(
        eigenvec, pos_2D, cell, fa_method, pos[:, 2]
    )
    # No need to update distances, they are preserved.

    return fa_pos, fa_cell, fa_rot


def data_augmentation(g, d=3, *args):
    """Data augmentation: randomly rotated graphs are added
    in the dataloader transform.

    Args:
        g (data.Data): single graph
        d (int): dimension of the DA rotation (2D around z-axis or 3D)
        rotation (str, optional): around which axis do we rotate it.
            Defaults to 'z'.

    Returns:
        (data.Data): rotated graph
    """

    # Sampling a random rotation within [-180, 180] for all axes.
    if d == 3:
        transform = RandomRotate([-180, 180], [0, 1, 2])  # 3D
    else:
        transform = RandomRotate([-180, 180], [2])  # 2D around z-axis

    # Rotate graph
    graph_rotated, _, _ = transform(g)

    return graph_rotated


@torch.no_grad()
def parse_batch(data, device):
    """
    pos = torch.tensor(data.x, dtype=torch.float32).to(device)
    forces = torch.tensor(data.x_forces, dtype=torch.float32).to(device)
    num_edges = torch.tensor(data.num_edges, dtype=int).to(device)
    b = num_edges.shape[0]
            
    edges = torch.tensor(data.edge_idx, dtype=torch.float32).to(device)
    """
    #torch.autograd.set_detect_anomaly(True)
    pos = data.positions
    num_edges = data.num_edges
    b = num_edges.shape[0]
    #num_edges = num_edges[0]
    b_n, _ = pos.shape
    n = int(b_n / b)
    
    #Rij = data.Rij 
    #loc_dist = data.distance 
    #Rij = Rij.view(b, num_edges, 3)
    #loc_dist = loc_dist.view(b, 1, num_edges).float()
    #iatoms = data.senders
    #jatoms = data.receivers
    #iatoms = iatoms.view(b, 1, num_edges).long()
    #jatoms = jatoms.view(b, 1, num_edges).long()

    #edges = torch.cat([iatoms, jatoms], dim=1).long()
    edges = data.edge_index
    cell = data.cell
    iatoms = edges[0]
    jatoms = edges[1]
    Sij = data.edges
    Sij = torch.split(Sij, num_edges.tolist(), dim=0)
    shift_v = torch.cat(
        [torch.einsum('ni,ij->nj', s, c)
         for s, c in zip(Sij, cell)], dim=0
    )
    _R = pos[jatoms] - pos[iatoms]
    Rij = _R + shift_v
    loc_dist = torch.norm(Rij, dim=1)
    loc_dist = loc_dist.view(b, 1, num_edges[0])

    iatoms = iatoms.view(b, 1, num_edges[0]).long()
    jatoms = jatoms.view(b, 1, num_edges[0]).long()
    offsets = torch.arange(b, device=iatoms.device).view(b, 1, 1) * n
    iatoms = iatoms - offsets
    jatoms = jatoms - offsets

    edges = torch.cat([iatoms, jatoms], dim=1).long()
    edge_attr = torch.cat([edges, loc_dist], dim=1)
    edge_attr = edge_attr.transpose(1, 2)


    species = data.species
    species = species.view(b, n, 1)
    species = species[:, :, :, None].expand(-1, -1, -1, 3).view(b, n, 3)
    #node_features = torch.cat([pos, forces], dim=-1)
    pos = pos.view(b, n, 3)
    node_features = torch.cat([pos, species], dim=-1) # pos, Rij..?
    # node_features = torch.cat([pos, forces, species.unsqueeze(-1)], dim=-1)
    node_features = node_features.view(b, n, 2, 3)    
    node_features = node_features.transpose(-1, -2)
    assert (pos - node_features[:, :, :, 0]).abs().sum().item() == 0

    idx = torch.tensor([i for i in range(b)], device=device)
    edge_features = torch.zeros(b, n, n, edge_attr.size(-1), device=device) 
    batch_idxs = torch.arange(b, device=device).repeat_interleave(num_edges[0]).long()
    edge_features[batch_idxs, edges[:, 0, :].flatten().long(), edges[:, 1, :].flatten().long(), :] \
        = edge_attr.reshape(-1, edge_attr.size(-1))

    return node_features, edge_features, edges, idx

@torch.no_grad()
def parse_batch_(data, device):
    """
    pos = torch.tensor(data.x, dtype=torch.float32).to(device)
    forces = torch.tensor(data.x_forces, dtype=torch.float32).to(device)
    num_edges = torch.tensor(data.num_edges, dtype=int).to(device)
    b = num_edges.shape[0]
            
    edges = torch.tensor(data.edge_idx, dtype=torch.float32).to(device)
    """
    pos = data.positions
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

    species = data.species
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