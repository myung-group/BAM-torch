from typing import Dict, List, Optional
import torch
from e3nn.util.jit import compile_mode
from bam_torch.utils.scatter import scatter_sum
from bam_torch.model.models import to_one_hot


@compile_mode("script")
class LAMMPS_BAM(torch.nn.Module):
    def __init__(self, model, **kwargs):
        super().__init__()
        self.model = model
        self.register_buffer("atomic_numbers", model.atomic_numbers.clone().detach())
        self.register_buffer("r_max", model.r_max.clone().detach())
        self.register_buffer("num_interactions", model.num_interactions.clone().detach())
        self.num_species = len(model.atomic_numbers)
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
        compute_virials: bool = False,
    ) -> Dict[str, Optional[torch.Tensor]]:
        num_graphs = data["ptr"].numel() - 1
        compute_displacement = compute_virials  # virials 계산 시 displacement 필요

        if "species" not in data:
            # LAMMPS의 atom types를 사용하거나, 단일 원소 가정
            num_atoms = data["positions"].shape[0]
            data["species"] = self.atomic_numbers.repeat(num_atoms)  # [num_atoms]
        
        # RACE 모델에 맞게 data와 backprop만 전달
        data["head"] = self.head
        out = self.model(data, backprop=False)  # RACE는 backprop 인자를 사용

        node_energy = out["energy"]  # RACE는 "energy"를 반환
        if node_energy is None:
            return {
                "total_energy_local": None,
                "node_energy": None,
                "forces": None,
                "virials": None,
            }

        positions = data["positions"]
        displacement = torch.zeros_like(data["cell"]) if compute_virials else None
        forces: Optional[torch.Tensor] = out.get("forces", torch.zeros_like(positions))
        virials: Optional[torch.Tensor] = torch.zeros_like(data["cell"])

        # 로컬 원자의 에너지 누적
        node_energy_local = node_energy * local_or_ghost
        total_energy_local = scatter_sum(
            src=node_energy_local, index=data["batch"], dim=-1, dim_size=num_graphs
        )

        # virials 계산이 필요할 경우 추가 처리
        if compute_virials and displacement is not None:
            grad_outputs: List[Optional[torch.Tensor]] = [
                torch.ones_like(total_energy_local)
            ]
            forces, virials = torch.autograd.grad(
                outputs=[total_energy_local],
                inputs=[positions, displacement],
                grad_outputs=grad_outputs,
                retain_graph=False,
                create_graph=False,
                allow_unused=True,
            )
            forces = -forces if forces is not None else torch.zeros_like(positions)
            virials = -virials if virials is not None else torch.zeros_like(displacement)

        return {
            "total_energy_local": total_energy_local,
            "node_energy": node_energy,
            "forces": forces,
            "virials": virials,
        }