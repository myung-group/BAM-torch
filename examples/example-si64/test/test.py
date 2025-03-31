import torch
model = torch.jit.load("model.pt-lammps.pt")
for name, param in model.named_parameters():
    print(f"{name}: {param.dtype}")
