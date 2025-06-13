from typing import Dict, List, Optional, Tuple

import torch
from e3nn.util.jit import compile_mode

from bam_torch.utils.scatter import scatter_sum


@compile_mode("script")
class LAMMPS_BAM(torch.nn.Module):
    def __init__(
            self, 
            model, 
            enr_avg_per_element: Optional[Dict[int, float]] = None, 
            e_corr: float = 0.0, 
            **kwargs
        ):
        super().__init__()
        self.model = model
        self.register_buffer("atomic_numbers", model.atomic_numbers)
        self.register_buffer("r_max", model.r_max)
        self.register_buffer("num_interactions", model.num_interactions)
        
        #self.register_buffer("lammps_shifts", torch.empty(0))
        #self.register_buffer("lammps_displacement", torch.empty(0))

        if enr_avg_per_element is not None:
            max_mapped_val = max(enr_avg_per_element.keys())
            enr_avg_tensor = torch.zeros(max_mapped_val + 1, dtype=torch.float32)
            for mapped_val, avg_energy in enr_avg_per_element.items():
                enr_avg_tensor[mapped_val] = avg_energy
            self.register_buffer("enr_avg_per_element", enr_avg_tensor)
        else:
            self.register_buffer("enr_avg_per_element", torch.empty(0))

        self.register_buffer("e_corr", torch.tensor(e_corr, dtype=torch.float32))

        if not hasattr(model, "heads"):
            model.heads = [None]
        self.register_buffer(
            "head",
            torch.tensor(
                self.model.heads.index(kwargs.get("head", self.model.heads[-1])),
                dtype=torch.long,
            ).unsqueeze(0),
        )

        for param in self.model.parameters():
            param.requires_grad = False
            

    def forward(self, data: Dict[str, torch.Tensor], local_or_ghost: torch.Tensor, compute_virials: bool = False) -> Dict[str, Optional[torch.Tensor]]:
        num_graphs = data["ptr"].numel() - 1
        data["head"] = self.head
        data["num_nodes"] = torch.tensor(data["positions"].shape[0], dtype=torch.long, device=data["positions"].device)

        out = self.model(data, backprop=True)

        if "energy" not in out or out["energy"] is None:
            return {
                "total_energy_local": None,
                "node_energy": None,
                "forces": None,
                "virials": None,
            }

        energy = out["energy"]
        assert energy is not None, "Energy should not be None"
        
        if (self.enr_avg_per_element.numel() > 0 and "species" in data):
            species = data["species"]
            node_avg_energies = self.enr_avg_per_element[species]
            
            node_enr_avg = scatter_sum(
                src=node_avg_energies,
                index=data["batch"],
                dim=0,
                dim_size=num_graphs,
            )
            
            energy = energy + node_enr_avg
        
        energy = energy + self.e_corr
        
        positions = data["positions"]

        forces = torch.autograd.grad(
            outputs=[energy.sum()],
            inputs=[positions],
            create_graph=True,
            retain_graph=True,
            allow_unused=True,
        )[0]

        if forces is not None:
            forces = -forces
        else:
            forces = torch.zeros_like(positions)

        num_nodes = positions.shape[0]
        node_energy = torch.zeros(num_nodes, dtype=energy.dtype, device=energy.device)
        
        for i in range(num_graphs):
            batch_mask = (data["batch"] == i)
            nodes_in_graph = batch_mask.sum().item()
            
            if nodes_in_graph > 0:
                node_energy[batch_mask] = energy[i] / nodes_in_graph

        """
        if hasattr(self.model, 'training_mode_for_lammps') and self.model.training_mode_for_lammps:
            if "shifts" in data:
                self.lammps_shifts = data["shifts"]
            if "cell" in data:
                self.lammps_displacement = torch.zeros(
                    (data["cell"].shape[0], 3, 3), 
                    dtype=energy.dtype, 
                    device=energy.device
                )
        """

        node_energy_local = node_energy * local_or_ghost
        total_energy_local = scatter_sum(
            src=node_energy_local,
            index=data["batch"],
            dim=0,
            dim_size=num_graphs,
        )

        virials = torch.zeros((1, 3, 3), dtype=energy.dtype, device=energy.device)

        return {
            "total_energy_local": total_energy_local,
            "node_energy": node_energy,
            "forces": forces,
            "virials": virials,
        }