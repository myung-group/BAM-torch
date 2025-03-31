from typing import Dict, List, Optional
import torch
from e3nn.util.jit import compile_mode
from bam_torch.utils.scatter import scatter_sum

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

        # --- Debug 출력 ---
        print(f"Input positions dtype: {data['positions'].dtype}")
        if "node_attrs" in data:
            print(f"Input node_attrs dtype: {data['node_attrs'].dtype}")
        print(f"Input batch dtype: {data['batch'].dtype}")
        print(f"Input ptr dtype: {data['ptr'].dtype}")
        print(f"batch: {data['batch'].shape}, max={data['batch'].max()}")
        print(f"ptr: {data['ptr']}, num_graphs={num_graphs}")
        print(f"local_or_ghost shape: {local_or_ghost.shape}, dtype={local_or_ghost.dtype}")

        # --- species 자동 생성 ---
        if "species" not in data:
            num_atoms = data["positions"].shape[0]
            atomic_number = int(self.atomic_numbers[0].item())
            data["species"] = torch.full((num_atoms,), atomic_number, dtype=torch.long, device=data["positions"].device)

        data["head"] = self.head
        data["batch"] = torch.zeros(data["positions"].shape[0], dtype=torch.long, device=data["positions"].device)
        data["ptr"] = torch.tensor([0, data["positions"].shape[0]], dtype=torch.long, device=data["positions"].device)

        # --- 모델 실행 ---
        out = self.model(data, backprop=True)
        node_energy = out["node_energy"]
        print(f"node_energy shape: {node_energy.shape}, dtype={node_energy.dtype}")

        if "forces" in out:
            print(f"Output forces dtype: {out['forces'].dtype}")

        if node_energy is None:
            return {
                "total_energy_local": None,
                "node_energy": None,
                "forces": None,
                "virials": None,
            }

        # --- autograd 연결 ---
        positions = data["positions"]
        if not positions.requires_grad:
            positions.requires_grad_(True)

        if compute_virials:
            displacement = torch.zeros_like(data["cell"])
            displacement.requires_grad_(True)
            virials: Optional[torch.Tensor] = torch.zeros_like(data["cell"]).unsqueeze(0)
        else:
            displacement = torch.zeros_like(data["cell"])  # dummy for TorchScript
            virials: Optional[torch.Tensor] = torch.zeros_like(data["cell"]).unsqueeze(0)

        forces: Optional[torch.Tensor] = out.get("forces", torch.zeros_like(positions))

        # --- energy sum ---
        node_energy_local = node_energy * local_or_ghost
        print(f"node_energy_local shape: {node_energy_local.shape}, dtype={node_energy_local.dtype}")

        if data["batch"].max() >= num_graphs:
            raise ValueError(f"batch max ({data['batch'].max()}) exceeds num_graphs ({num_graphs})")

        total_energy_local = scatter_sum(
            src=node_energy_local, index=data["batch"], dim=0, dim_size=num_graphs
        )
        print(f"total_energy_local shape: {total_energy_local.shape}, dtype={total_energy_local.dtype}")

        # --- autograd 기반 force/virial 계산 ---
        if compute_virials:
            grad_outputs: Optional[List[Optional[torch.Tensor]]] = [torch.ones_like(total_energy_local)]
            grads = torch.autograd.grad(
                outputs=[total_energy_local],
                inputs=[positions, displacement],
                grad_outputs=grad_outputs,
                retain_graph=True,
                create_graph=False,
                allow_unused=True,
            )
            forces_grad, virials_grad = grads
            if forces_grad is not None:
                forces = -forces_grad
                print(f"Forces dtype after grad: {forces.dtype}")
            if virials_grad is not None:
                virials = -virials_grad.unsqueeze(0)  # make [1, 3, 3]
                print(f"Virials dtype after grad: {virials.dtype}")

        # --- 최종 반환 ---
        return {
            "total_energy_local": total_energy_local,
            "node_energy": node_energy,
            "forces": forces,
            "virials": virials,
        }
