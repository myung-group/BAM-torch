from typing import Dict, List, Optional, Tuple

import torch
from e3nn.util.jit import compile_mode

from bam_torch.utils.scatter import scatter_sum


@compile_mode("script")
class LAMMPS_BAM(torch.nn.Module):
    def __init__(
        self, 
        model, 
        enr_avg_per_element: Dict[int, float], 
        e_corr: float = 0.0, 
        **kwargs
    ):
        super().__init__()
        self.model = model
        self.register_buffer("atomic_numbers", model.atomic_numbers)
        self.register_buffer("r_max", model.r_max)
        self.register_buffer("num_interactions", model.num_interactions)

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
            

    def forward(
        self, 
        data: Dict[str, torch.Tensor], 
        local_or_ghost: torch.Tensor, 
        compute_virials: bool = False
    ) -> Dict[str, Optional[torch.Tensor]]:
        
        num_graphs = data["ptr"].numel() - 1
        data["head"] = self.head
        data["num_nodes"] = torch.tensor(data["positions"].shape[0], 
                                         dtype=torch.long, 
                                         device=data["positions"].device)

        out = self.model(data, backprop=False)
        """
        if "energy" not in out or out["energy"] is None:
            return {
                "total_energy_local": None,
                "node_energy": None,
                "forces": None,
                "virials": None,
            }
        """
        #assert energy is not None, "Energy should not be None"
        """
        node_enr_avg = scatter_sum(
            src=node_avg_energies,
            index=data["batch"],
            dim=0,
            dim_size=num_graphs,
        )
        energy = energy + node_enr_avg + self.e_corr
        """

        node_energy = out["node_energy"]
        assert node_energy is not None
        forces = out["forces"]
        
        species = data["species"]
        local_species = species[local_or_ghost]
        local_node_avg_energies = self.enr_avg_per_element[local_species]

        node_energy[local_or_ghost] = node_energy[local_or_ghost] + local_node_avg_energies
        energy = node_energy.sum()

        if self.e_corr != 0.0:
            local_count = local_or_ghost.sum().item()
            if "total_local_atoms" in data:  # if multi-GPU
                total_system_atoms = data["total_local_atoms"].item()
            else:  # if single-GPU
                total_system_atoms = local_count
            e_corr_per_local = self.e_corr / total_system_atoms
            node_energy[local_or_ghost] = node_energy[local_or_ghost] + e_corr_per_local
            energy = energy + (e_corr_per_local * local_count)

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