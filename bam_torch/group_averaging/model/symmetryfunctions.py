"""
All the symmetry functions used in BPNN (PyTorch version)
"""

import torch

class G1:
    def __repr__(self):
        return "G1"
    def __call__(self, **var):
        return torch.sum(var["fc"], dim=-1)  # (..., j) -> (...,)

class G2:
    def __init__(self, eta, Rs):
        self.eta = float(eta)
        self.Rs = float(Rs)
    def __repr__(self):
        return f"G2({self.eta:.2f}, {self.Rs:.2f})"
    def __call__(self, **var):
        sf = torch.exp(-self.eta * (var["Rij"] - self.Rs) ** 2) * var["fc"]
        return torch.sum(sf, dim=-1)

class G4:
    def __init__(self, eta=1.0, zeta=1.0, lambd=1.0):
        self.eta = float(eta)
        self.zeta = float(zeta)
        self.lambd = float(lambd)
    def __repr__(self):
        return f"G4({self.eta:.2f}, {self.zeta:.2f}, {self.lambd:.2f})"
    def __call__(self, **var):
        Rij_j = var["Rij"].unsqueeze(-1)   # (..., j, 1)
        Rij_k = var["Rij"].unsqueeze(-2)   # (..., 1, j)

        expo = torch.exp(-self.eta * (Rij_j + Rij_k))
        cosin = (var["cos"] * self.lambd + 1.0) ** self.zeta
        cutoff = var["fc"].unsqueeze(-1) * var["fc"].unsqueeze(-2)

        sf = (2.0 ** (1.0 - self.zeta)) * expo * cutoff * cosin
        return torch.sum(torch.sum(sf, dim=-1), dim=-1)  # (..., j, k)->(...)

