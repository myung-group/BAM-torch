###########################################################################################
# Utilities
# Authors: Ilyes Batatia, Gregor Simm and David Kovacs
# This program is distributed under the MIT License (see MIT.md)
###########################################################################################

from typing import List, Optional, Tuple
from torch.jit import annotate
import torch
import torch.nn
import torch.utils.data
from bam_torch.utils.scatter import scatter_sum, scatter_mean


def compute_forces(
    energy: torch.Tensor, positions: torch.Tensor, training: bool = True
) -> torch.Tensor:
    grad_outputs: List[Optional[torch.Tensor]] = [torch.ones_like(energy)]
    gradient = torch.autograd.grad(
        outputs=[energy],  # [n_graphs, ]
        inputs=[positions],  # [n_nodes, 3]
        grad_outputs=grad_outputs,
        retain_graph=training,  # Make sure the graph is not destroyed during training
        create_graph=training,  # Create graph for second derivative
        allow_unused=True,  # For complete dissociation turn to true
    )[0]  # [n_nodes, 3]
    if gradient is None:
        return torch.zeros_like(positions)
    return -1 * gradient


def compute_forces_virials(
    energy: torch.Tensor,
    positions: torch.Tensor,
    displacement: torch.Tensor,
    cell: torch.Tensor,
    training: bool = True,
    compute_stress: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    grad_outputs: List[Optional[torch.Tensor]] = [torch.ones_like(energy)]
    forces, virials = torch.autograd.grad(
        outputs=[energy],  # [n_graphs, ]
        inputs=[positions, displacement],  # [n_nodes, 3]
        grad_outputs=grad_outputs,
        retain_graph=training,  # Make sure the graph is not destroyed during training
        create_graph=training,  # Create graph for second derivative
        allow_unused=True,
    )
    stress = torch.zeros_like(displacement)
    if compute_stress and virials is not None:
        cell = cell.view(-1, 3, 3)
        volume = torch.linalg.det(cell).abs().unsqueeze(-1)
        stress = virials / volume.view(-1, 1, 1)
        stress = torch.where(torch.abs(stress) < 1e10, stress, torch.zeros_like(stress))
    if forces is None:
        forces = torch.zeros_like(positions)
    if virials is None:
        virials = torch.zeros((1, 3, 3))

    return -1 * forces, -1 * virials, stress

def compute_forces_stress(
    energy: torch.Tensor,
    positions: torch.Tensor,
    cell: torch.Tensor,
    batch_idx: torch.Tensor,
    num_graphs: int,
    training: bool = True,
    compute_stress: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    grad_outputs = annotate(List[Optional[torch.Tensor]], [torch.ones_like(energy)])
    #cellgrad_outputs: List[Optional[torch.Tensor]] = [torch.ones_like(energy)]
   # grad : torch.Tensor = torch.tensor([])
    virials = torch.zeros((num_graphs, 3, 3), dtype=positions.dtype, device=positions.device)
    stress = torch.zeros((num_graphs, 6), dtype=positions.dtype, device=positions.device)
    #n_edges: torch.Tensor = 
    grad, cellgrad = torch.autograd.grad(
        outputs=[energy],  # [n_graphs, ]
        inputs=[positions, cell],  # [n_nodes, 3]
        grad_outputs=grad_outputs,
        retain_graph=True,  # Make sure the graph is not destroyed during training
        create_graph=True,  # Create graph for second derivative
        allow_unused=True,
    )
    #print(grad)
    if compute_stress and cellgrad is not None:
        cell = cell.view(-1, 3, 3)
        volume = torch.linalg.det(cell).abs().unsqueeze(-1)
        stress_cell = (
            torch.transpose(cellgrad, 1, 2) @ cell
        )
        assert grad is not None
        assert positions is not None
        stress_grad = torch.einsum("iu,iv->iuv", grad, positions)
        stress_grad = scatter_sum(
                src=stress_grad,
                index=batch_idx,
                dim=0,
                dim_size=num_graphs,
            ) 

        virials = stress_cell + stress_grad
        stress = virials / volume.view(-1, 1, 1)
        #stress = torch.where(torch.abs(stress) < 1e10, stress, torch.zeros_like(stress))
        #stress = stress[:-1]
        stress = stress.view(-1, 9)[:, [0,4,8,5,2,1]]

    if grad is None:
        forces = torch.zeros_like(positions)
    #if cellgrad is None:
    #    virials = torch.zeros((1, 3, 3))

    #del cellgrad
    #torch.cuda.empty_cache()
    assert grad is not None
    assert isinstance(virials, torch.Tensor)
    assert stress is not None
    return -1 * grad, -1 * virials, stress


def get_symmetric_displacement(
    positions: torch.Tensor,
    cell: Optional[torch.Tensor],
    edge_index: torch.Tensor,
    num_graphs: int,
    batch: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if cell is None:
        cell = torch.zeros(
            num_graphs * 3,
            3,
            dtype=positions.dtype,
            device=positions.device,
        )
    sender = edge_index[0]
    displacement = torch.zeros(
        (num_graphs, 3, 3),
        dtype=positions.dtype,
        device=positions.device,
    )
    displacement.requires_grad_(True)
    symmetric_displacement = 0.5 * (
        displacement + displacement.transpose(-1, -2)
    )  # From https://github.com/mir-group/nequip
    positions = positions + torch.einsum(
        "be,bec->bc", positions, symmetric_displacement[batch]
    )
    cell = cell.view(-1, 3, 3)
    cell = cell + torch.matmul(cell, symmetric_displacement)

    return positions, displacement


@torch.jit.unused
def compute_hessians_vmap(
    forces: torch.Tensor,
    positions: torch.Tensor,
) -> torch.Tensor:
    forces_flatten = forces.view(-1)
    num_elements = forces_flatten.shape[0]

    def get_vjp(v):
        return torch.autograd.grad(
            -1 * forces_flatten,
            positions,
            v,
            retain_graph=True,
            create_graph=False,
            allow_unused=False,
        )

    I_N = torch.eye(num_elements).to(forces.device)
    try:
        chunk_size = 1 if num_elements < 64 else 16
        gradient = torch.vmap(get_vjp, in_dims=0, out_dims=0, chunk_size=chunk_size)(
            I_N
        )[0]
    except RuntimeError:
        gradient = compute_hessians_loop(forces, positions)
    if gradient is None:
        return torch.zeros((positions.shape[0], forces.shape[0], 3, 3))
    return gradient


@torch.jit.unused
def compute_hessians_loop(
    forces: torch.Tensor,
    positions: torch.Tensor,
) -> torch.Tensor:
    hessian = []
    for grad_elem in forces.view(-1):
        hess_row = torch.autograd.grad(
            outputs=[-1 * grad_elem],
            inputs=[positions],
            grad_outputs=torch.ones_like(grad_elem),
            retain_graph=True,
            create_graph=False,
            allow_unused=False,
        )[0]
        hess_row = hess_row.detach()  # this makes it very slow? but needs less memory
        if hess_row is None:
            hessian.append(torch.zeros_like(positions))
        else:
            hessian.append(hess_row)
    hessian = torch.stack(hessian)
    return hessian


def get_outputs(
    energy: torch.Tensor,
    positions: torch.Tensor,
    displacement: Optional[torch.Tensor],
    cell: torch.Tensor,
    batch_idx: torch.Tensor,
    num_graphs: int,
    training: bool = False,
    compute_force: bool = True,
    compute_virials: bool = True,
    compute_stress: bool = True,
    compute_hessian: bool = False,
) -> Tuple[
    Optional[torch.Tensor],
    Optional[torch.Tensor],
    Optional[torch.Tensor],
    Optional[torch.Tensor],
]:
    if (compute_virials or compute_stress) and displacement is not None:
        """
        forces, virials, stress = compute_forces_virials(
            energy=energy,
            positions=positions,
            displacement=displacement,
            cell=cell,
            compute_stress=compute_stress,
            training=(training or compute_hessian),
        )
        """
        forces, virials, stress = compute_forces_stress(
            energy=energy,
            positions=positions,
            cell=cell,
            batch_idx=batch_idx,
            num_graphs=num_graphs,
            training=(training or compute_hessian),
            compute_stress=compute_stress,
        )
    elif compute_force:
        forces, virials, stress = (
            compute_forces(
                energy=energy,
                positions=positions,
                training=(training or compute_hessian),
            ),
            None,
            None,
        )
    else:
        forces, virials, stress = (None, None, None)

    if compute_hessian:
        assert forces is not None, "Forces must be computed to get the hessian"
        hessian = compute_hessians_vmap(forces, positions)
    else:
        hessian = None

    return forces, virials, stress, hessian


def _compute_net_torque(
    positions: torch.Tensor,
    forces: torch.Tensor,
    n_nodes: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute the net torque on a system of particles."""
    com = scatter_mean(positions, n_nodes, dim=0)
    com_repeat = com[n_nodes]  # Shape: (N, 3)
    com_relative_positions = positions - com_repeat  # Shape: (N, 3)
    torques = torch.linalg.cross(com_relative_positions, forces)  # Shape: (N, 3)
    net_torque = scatter_sum(torques, n_nodes, dim=0)
    return net_torque, com_relative_positions

def remove_net_torque(
    positions: torch.Tensor,
    forces: torch.Tensor,
    n_nodes: torch.Tensor,
) -> torch.Tensor:
    """Adjust the predicted forces to eliminate net torque for each graph in the batch.

    The mathematical derivation of this function is given here:
    https://www.notion.so/orbitalmaterials/Net-torque-removal-11f56117b79780dfbbb9ce78e245be38?pvs=4

    The naming conventions here match those in the derivation.

    Args:
        positions : torch.Tensor of shape (N, 3)
            Positions of atoms (concatenated for all graphs in the batch).
        forces : torch.Tensor of shape (N, 3)
            Predicted forces on atoms.
        n_nodes : torch.Tensor of shape (B,)
            Number of nodes in each graph, where B is the number of graphs in the batch.

    Returns:
        adjusted_forces : torch.Tensor of shape (N, 3)
            Adjusted forces with zero net torque and net force for each graph.
    """
    # B = n_nodes.shape[0]
    uniq_n_nodes = torch.unique(n_nodes)
    B = uniq_n_nodes.shape[0]
    tau_total, r = _compute_net_torque(positions, forces, n_nodes)

    # Compute scalar s per graph: sum_i ||r_i||^2
    r_squared = torch.sum(r**2, dim=1)  # Shape: (N,)
    s = scatter_sum(r_squared, n_nodes, dim=0)  # Shape: (B,)

    # Compute matrix S per graph: sum_i outer(r_i, r_i)
    r_unsqueezed = r.unsqueeze(2)  # Shape: (N, 3, 1)
    r_T_unsqueezed = r.unsqueeze(1)  # Shape: (N, 1, 3)
    outer_products = r_unsqueezed @ r_T_unsqueezed  # Shape: (N, 3, 3)
    S = scatter_sum(outer_products, n_nodes, dim=0)  # Shape: (B, 3, 3)

    # Compute M = S - sI
    I = (  # noqa: E741
        torch.eye(3, device=positions.device).unsqueeze(0).expand(B, -1, -1)
    )  # Shape: (B, 3, 3)
    M = S - (s.view(-1, 1, 1)) * I  # Shape: (B, 3, 3)

    # Right-hand side vector b per graph
    b = -tau_total  # Shape: (B, 3)

    # Solve M * mu = b for mu per graph
    try:
        mu = torch.linalg.solve(M, b.unsqueeze(2)).squeeze(2)  # Shape: (B, 3)
    except RuntimeError:
        # Handle singular matrix M by using the pseudo-inverse
        M_pinv = torch.linalg.pinv(M)  # Shape: (B, 3, 3)
        mu = torch.bmm(M_pinv, b.unsqueeze(2)).squeeze(2)  # Shape: (B, 3)

    # Compute adjustments to forces
    mu_batch = mu[n_nodes]  # Shape: (N, 3)
    forces_delta = torch.linalg.cross(r, mu_batch)  # Shape: (N, 3)

    # Adjusted forces
    adjusted_forces = forces + forces_delta  # Shape: (N, 3)

    return adjusted_forces
