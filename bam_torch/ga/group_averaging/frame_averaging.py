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


def probablistic_averaging_3D(data, gs=None, fa_method="prob", check=False):
    
    b, k, _, _ = gs.shape
    b_n, _ = data.pos.shape
    n = int(b_n/b)
    device = data.pos.device

    pos = data.pos.view(b, n, 3)
    pos = pos[:, None, :, :].expand(b, k, n, 3)
    cell = data.cell[:, None, :, :].expand(b, k, 3, 3)
    _, center = torch.std_mean(pos, dim=2, keepdim=True)
    pos = pos - center

    # if symmetry == 'O3' or 'SO3'
    gs_inv = gs.transpose(2, 3).to(device)
    pa_pos = torch.einsum('bkij,bkjt->bkit', pos, gs_inv)
    pa_pos = pa_pos.view(b*k, n, 3)
    pa_pos = pa_pos.view(b*k*n, 3)
    pa_cell = torch.einsum('bkij,bkjt->bkit', cell, gs_inv)

    cell_offsets = data.cell_offsets
    cell_offsets = cell_offsets.view(b, -1, 3)
    b, e, _ = cell_offsets.shape
    cell_offsets = cell_offsets[:, None, :, :].expand(b, k, e, 3)
    cell_offsets = cell_offsets.reshape(b*k, e, 3)
    data.cell_offsets = cell_offsets.view(b*k*e, 3)

    data.fa_rot = [gs_inv.view(b*k, 3, 3)]
    data.fa_pos = [pa_pos]
    data.fa_cell = [pa_cell.view(b*k, 3, 3)]

    iatoms = data.senders
    jatoms = data.receivers
    iatoms = iatoms.reshape(b, 1, e).long()
    jatoms = jatoms.reshape(b, 1, e).long()
    edge_idx = torch.cat([jatoms, iatoms], dim=1).long()
    edge_idx = edge_idx[:, None, :, :].expand(b, k, 2, e)
    edge_idx = edge_idx.reshape(b*k, 2, e)

    incr = n * torch.arange(b*k, device=device)
    incr_edge_idx = edge_idx + incr[:, None, None]
    pyg_edge_idx = incr_edge_idx.transpose(0, 1).reshape(2, b*k*e).long()
    data.pa_edge_index = pyg_edge_idx


    return data


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
