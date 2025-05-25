import torch
from tqdm import tqdm
from copy import deepcopy

from bam_torch.ga.utils.fa_utils import RandomRotate, RandomReflect
from bam_torch.ga.group_averaging.transforms import FrameAveraging
from torch_geometric.data import Batch


def transform_batch(batch, frame_averaging, fa_method, neighbors=None):
    r"""Apply a transformation to a batch of graphs

    Args:
        batch (data.Batch): batch of data.Data objects.
        frame_averaging (str): Transform method used.
        fa_method (str): FA method used.
        neighbors (list, optional): list containing the number of edges
            in each graph of the batch. (default: :obj:`None`)

    Returns:
        (data.Batch): transformed batch sample
    """
    delattr(batch, "fa_pos")  # delete it otherwise can't iterate
    delattr(batch, "fa_cell")  # delete it otherwise can't iterate
    delattr(batch, "fa_rot")  # delete it otherwise can't iterate

    # Convert batch to list of graphs
    g_list = batch.to_data_list()

    # Apply transform to individual graphs of batch
    fa_transform = FrameAveraging(frame_averaging, fa_method)
    for g in g_list:
        g = fa_transform(g)
    batch = Batch.from_data_list(g_list)
    if neighbors is not None:
        batch.neighbors = neighbors
    return batch


def rotate_graph(batch, frame_averaging, fa_method, rotation=None):
    r"""Rotate all graphs in a batch

    Args:
        batch (data.Batch): batch of graphs.
        frame_averaging (str): Transform method used.
            ("2D", "3D", "DA", "")
        fa_method (str): FA method used.
            ("", "stochastic", "all", "det", "se3-stochastic", "se3-all", "se3-det")
        rotation (str, optional): type of rotation applied. (default: :obj:`None`)
            ("z", "x", "y", None)

    Returns:
        (dict): rotated batch sample and rotation matrix used to rotate it
    """
    if isinstance(batch, list):
        batch = batch[0]

    # Sampling a random rotation within [-180, 180] for all axes.
    if rotation == "z":
        transform = RandomRotate([-180, 180], [2])
    elif rotation == "x":
        transform = RandomRotate([-180, 180], [0])
    elif rotation == "y":
        transform = RandomRotate([-180, 180], [1])
    else:
        transform = RandomRotate([-180, 180], [0, 1, 2])

    # Rotate graph
    batch_rotated, rot, inv_rot = transform(deepcopy(batch))
    assert not torch.allclose(batch.pos, batch_rotated.pos, atol=1e-05)

    # Recompute fa-pos for batch_rotated
    if hasattr(batch, "fa_pos"):
        batch_rotated = transform_batch(
            batch_rotated,
            frame_averaging,
            fa_method,
            batch.neighbors if hasattr(batch, "neighbors") else None,
        )

    return {"batch_list": [batch_rotated], "rot": rot}


def reflect_graph(batch, frame_averaging, fa_method, reflection=None):
    r"""Rotate all graphs in a batch

    Args:
        batch (data.Batch): batch of graphs
        frame_averaging (str): Transform method used
            ("2D", "3D", "DA", "")
        fa_method (str): FA method used
            ("", "stochastic", "all", "det", "se3-stochastic", "se3-all", "se3-det")
        reflection (str, optional): type of reflection applied. (default: :obj:`None`)

    Returns:
        (dict): reflected batch sample and rotation matrix used to reflect it
    """
    if isinstance(batch, list):
        batch = batch[0]

    # Sampling a random rotation within [-180, 180] for all axes.
    transform = RandomReflect()

    # Reflect batch
    batch_reflected, rot, inv_rot = transform(deepcopy(batch))
    assert not torch.allclose(batch.pos, batch_reflected.pos, atol=1e-05)

    if hasattr(batch, "fa_pos"):
        batch_reflected = transform_batch(
            batch_reflected,
            frame_averaging,
            fa_method,
            batch.neighbors if hasattr(batch, "neighbors") else None,
        )
    return {"batch_list": [batch_reflected], "rot": rot}


@torch.no_grad()
def eval_model_symmetries(data_loader, model, model_forward, 
                          frame_averaging, fa_method, device):
    """Test rotation and reflection invariance & equivariance of GNNs

    Args:
        loader (data): dataloader
        model: model instance
        frame_averaging (str): frame averaging ("2D", "3D"), data augmentation ("DA")
            or none ("")
        fa_method (str): _description_
        task_name (str): the targeted task
            ("energy", "forces")
        crystal_task (bool): whether we have a crystal (i.e. a unit cell)
            or a molecule

    Returns:
        (dict): metrics measuring invariance/equivariance
            of energy/force predictions
    """
    model.eval()

    energy_diff = torch.zeros(1, device=device)
    energy_diff_z = torch.zeros(1, device=device)
    energy_diff_z_percentage = torch.zeros(1, device=device)
    energy_diff_refl = torch.zeros(1, device=device)
    pos_diff_total = torch.zeros(1, device=device)
    forces_diff = torch.zeros(1, device=device)
    forces_diff_z = torch.zeros(1, device=device)
    forces_diff_z_graph = torch.zeros(1, device=device)
    forces_diff_refl = torch.zeros(1, device=device)
    n_batches = 0
    n_atoms = 0

    for batch in tqdm(data_loader, total=len(data_loader), position=0, desc="Evaluating Symmetry... "):
        batch = batch.to(device)
        try:
            n_batches += len(batch[0].natoms) 
        except:
            n_batches += 1
        
        try:
            n_atoms += batch[0].natoms.sum()
        except:
            n_atoms += batch[0].num_nodes
        batch.pos = batch.positions
        print(batch)
        t_batch = transform_batch(batch, frame_averaging, fa_method, neighbors=None)
        # Computes model prediction
        preds1 = model_forward(deepcopy(t_batch), 
                               model = model,
                               mode='eval',
                               frame_averaging="3D",
                               crystal_task=True)

        # Compute prediction on rotated graph
        print(t_batch)
        rotated = rotate_graph(t_batch, frame_averaging, fa_method, rotation="z")
        preds2 = model_forward(deepcopy(rotated["batch_list"]), 
                               model = model,
                               mode='eval',
                               frame_averaging="3D",
                               crystal_task=True)
        
        # Difference in predictions, for energy and forces
        energy_diff_z += torch.abs(preds1["energy"] - preds2["energy"]).sum()
        energy_diff_z_percentage += (torch.abs(preds1["energy"] - preds2["energy"])
                                     / torch.abs(batch[0].y).to(preds1["energy"].device)
                                     ).sum()
        
        # Difference in positions
        pos_diff = 0
        # Compute total difference across frames
        for pos1, pos2 in zip(t_batch.fa_pos, rotated["batch_list"][0].fa_pos):
            pos_diff += pos1 - pos2
        # Manhattan distance of pos matrix wrt 0 matrix
        pos_diff_total += torch.abs(pos_diff).sum()
    

        # Reflect graph and compute diff in prediction
        reflected = reflect_graph(t_batch, frame_averaging, fa_method)
        preds3 = model_forward(reflected["batch_list"], 
                               model = model,
                               mode='eval',
                               frame_averaging="3D",
                               crystal_task=True)
        energy_diff_refl += torch.abs(preds1["energy"] - preds3["energy"]).sum()

        # 3D rotation and compute difference in prediction
        rotated = rotate_graph(t_batch, frame_averaging, fa_method)
        preds4 = model_forward(rotated["batch_list"], 
                               model = model,
                               mode='eval',
                               frame_averaging="3D",
                               crystal_task=True)
        energy_diff += torch.abs(preds1["energy"] - preds4["energy"]).sum()
        #forces_diff += torch.abs(preds1["forces"] - preds4["forces"]).sum()
        forces_diff += torch.abs(preds1["forces"] @ rotated["rot"].to(preds1["forces"].device)
                                  - preds4["forces"]).sum()

    # Aggregate the results
    energy_diff_z = energy_diff_z / n_batches
    energy_diff_z_percentage = energy_diff_z_percentage / n_batches
    energy_diff = energy_diff / n_batches
    energy_diff_refl = energy_diff_refl / n_batches
    pos_diff_total = pos_diff_total / n_batches

    symmetry = {
        "2D_E_rot": float(energy_diff_z),
        "2D_E_rot_percentage": float(energy_diff_z_percentage),
        "3D_E_rot": float(energy_diff),
        "2D_pos_rot": float(pos_diff_total),
        "2D_E_refl": float(energy_diff_refl),
    }

    # Test equivariance of forces
    forces_diff_z = forces_diff_z / n_atoms
    forces_diff_z_graph = forces_diff_z / n_batches
    forces_diff = forces_diff / n_atoms
    forces_diff_refl = forces_diff_refl / n_atoms
    #symmetry.update(
    #    {
    #        "2D_F_ri_graph": float(forces_diff_z_graph),
    #        "2D_F_ri": float(forces_diff_z),
    #        "3D_F_ri": float(forces_diff),
    #        "2D_F_refl_i": float(forces_diff_refl),
    #    }
    #)
    symmetry.update({"3D_F_rot": float(forces_diff)})

    return symmetry