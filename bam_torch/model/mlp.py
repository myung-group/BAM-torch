import torch
from e3nn.o3 import Irreps

import math
import torch
import torch.nn as nn
from typing import Any, Callable, Dict, List, Optional, Type, Union, Tuple, Sequence


class SeparatedLayerNorm(nn.Module):
    """
    Equivariant Layer Normalization that treats scalar and non-scalar irreps separately.

    - Scalar (l=0): mean-centering + division by std (standard LayerNorm)
    - Non-scalar (l>=1): division by shared RMS (no centering, to preserve equivariance)

    Args:
        irreps (Irreps): Input irreducible representations
        eps (float): Small constant for numerical stability
        affine (bool): If True, learnable scale (gamma) per multiplicity channel
    """

    def __init__(self, irreps: Irreps, eps: float = 1e-6, affine: bool = True):
        super().__init__()
        self.irreps = Irreps(irreps)
        self.eps = eps
        self.affine = affine

        # Affine parameters: one scale per (mul, ir) group
        if affine:
            num_params = sum(mul for mul, _ in self.irreps)
            self.gamma = nn.Parameter(torch.ones(num_params))
        else:
            self.register_parameter('gamma', None)

        self._slices = []
        idx = 0
        for mul, ir in self.irreps:
            dim = mul * ir.dim
            self._slices.append((idx, idx + dim, mul, ir))
            idx += dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, irreps_dim) or (batch, N, irreps_dim) 
        """
        orig_shape = x.shape
        leading_shape = orig_shape[:-1]  # batch dims
        x = x.view(-1, orig_shape[-1])   # (B, D)
        B = x.shape[0]

        xi_s = torch.zeros(B, 1, device=x.device, dtype=x.dtype)
        cnt = 0

        for start, end, mul, ir in self._slices:
            if ir.l > 0:  # non-scalar
                xi = x[:, start:end]             # (B, mul * (2l+1))
                xi_s += torch.sum(xi ** 2, dim=-1, keepdim=True)
                cnt += mul * ir.dim

        if cnt > 0:
            nonscalar_rms = torch.sqrt(xi_s / cnt + self.eps)  # (B, 1)
        else:
            nonscalar_rms = torch.ones(B, 1, device=x.device, dtype=x.dtype)

        out = torch.zeros_like(x)
        gamma_idx = 0

        for start, end, mul, ir in self._slices:
            xi = x[:, start:end]  # (B, mul * ir.dim)

            if ir.l == 0:
                # --- Scalar: standard LayerNorm ---
                xi = xi.view(B, mul)  # (B, mul), dim=1 since l=0
                xi_mean = xi.mean(dim=-1, keepdim=True)           # (B, 1)
                xi_centered = xi - xi_mean
                xi_std = torch.sqrt(
                    (xi_centered ** 2).mean(dim=-1, keepdim=True) + self.eps
                )  # (B, 1)
                xi_normed = xi_centered / xi_std  # (B, mul)

            else:
                # --- Non-scalar: shared RMS norm (no centering) ---
                xi_normed = xi / nonscalar_rms    # (B, mul * ir.dim)

            # Affine transform (per-channel scale)
            if self.affine:
                gamma = self.gamma[gamma_idx:gamma_idx + mul]  # (mul,)
                if ir.l == 0:
                    xi_normed = xi_normed * gamma.unsqueeze(0)  # (B, mul)
                else:
                    gamma_expanded = gamma.repeat_interleave(ir.dim)  # (mul * ir.dim,)
                    xi_normed = xi_normed * gamma_expanded.unsqueeze(0)

            out[:, start:end] = xi_normed.view(B, -1)
            gamma_idx += mul

        return out.view(orig_shape)

    def __repr__(self):
        return f"SeparatedLayerNorm({self.irreps}, eps={self.eps}, affine={self.affine})"


def separated_layer_norm(irreps: Irreps, chunks: list[torch.Tensor]):
    """
    Args:
        irreps: e3nn.o3.Irreps
        chunks: list of tensors, each shape [B, mul, ir.dim]
    Returns:
        new_chunks: list of tensors with same shapes
    """

    device = chunks[0].device
    B = chunks[0].shape[0]

    xi_s = torch.zeros((B, 1), device=device)
    cnt = 0

    # --- non-scalar RMS ---
    print(irreps)
    print(chunks.shape)
    for (mul, ir), chunk in zip(irreps, chunks):
        if not ir.is_scalar():
            xi = chunk.reshape(B, mul * ir.dim)
            xi_s += torch.sum(xi ** 2, dim=-1, keepdim=True)
            cnt += mul * ir.dim

    xi_rms = torch.sqrt(xi_s / cnt + 1e-6)

    new_chunks = []

    for (mul, ir), chunk in zip(irreps, chunks):
        if ir.is_scalar():
            xi = chunk.reshape(B, mul)
            xi_centered = xi - xi.mean(dim=-1, keepdim=True)
            xi_rms_s = torch.sqrt(
                torch.mean(xi_centered ** 2, dim=-1, keepdim=True) + 1e-6
            )
            new_chunk = (xi_centered / xi_rms_s).reshape(B, mul, ir.dim)
        else:
            xi = chunk.reshape(B, mul * ir.dim)
            new_chunk = (xi / xi_rms).reshape(B, mul, ir.dim)

        new_chunks.append(new_chunk)

    return new_chunks

def separated_layer_norm(x: torch.Tensor):
    """
    x: (B, C)  where C = number of scalar channels
    """
    mean = x.mean(dim=-1, keepdim=True)
    var = ((x - mean) ** 2).mean(dim=-1, keepdim=True)
    return (x - mean) / torch.sqrt(var + 1e-6)

"""
class Linear(nn.Module):

    def __init__(
        self,
        in_size: int,
        out_size: int,
        use_bias: bool = True,
        init_scale: float = 1.0,
    ):
        super().__init__()

        scale = math.sqrt(init_scale / in_size)

        self.weight = nn.Parameter(
            torch.randn(in_size, out_size) * scale
        )

        self.use_bias = use_bias
        if use_bias:
            self.bias = nn.Parameter(torch.zeros(out_size))
        else:
            self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x @ self.weight
        if self.use_bias:
            x = x + self.bias
        return x

class MLP(nn.Module):

    def __init__(
        self,
        sizes,
        activation=torch.nn.SiLU(),
        init_scale: float = 1.0,
        use_bias: bool = False,
    ):
        super().__init__()

        self.activation = activation
        layers = []

        for i in range(len(sizes) - 1):
            layer = Linear(
                sizes[i],
                sizes[i + 1],
                use_bias=use_bias,
                init_scale=init_scale if i < len(sizes) - 2 else 1.0,
            )
            layers.append(layer)

        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = self.activation(x)
        return x
"""

class Linear(nn.Module):
    """Simple linear layer with optional bias."""
    def __init__(
        self,
        in_size: int,
        out_size: int,
        use_bias: bool = True,
        init_scale: float = 1.0,
    ):
        super().__init__()

        scale = math.sqrt(init_scale / in_size)

        # Xavier/He initialization
        self.weight = nn.Parameter(
            torch.randn(in_size, out_size) * scale
        )

        self.use_bias = use_bias
        if use_bias:
            self.bias = nn.Parameter(torch.zeros(out_size))
        else:
            self.register_parameter('bias', None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x @ self.weight
        if self.use_bias:
            x = x + self.bias
        return x


class MLP(nn.Module):
    """Multi-layer perceptron."""
    def __init__(
        self,
        sizes: Sequence[int],
        activation: Callable = torch.nn.functional.silu, 
        init_scale: float = 1.0,
        use_bias: bool = False,
    ):
        super().__init__()

        if not callable(activation):
            raise ValueError(f"activation must be callable, got {type(activation)}")
        
        self.activation = activation
        layers = []

        for i in range(len(sizes) - 1):
            current_scale = init_scale if i < len(sizes) - 2 else 1.0
            
            layer = Linear(
                sizes[i],
                sizes[i + 1],
                use_bias=use_bias,
                init_scale=current_scale,
            )
            layers.append(layer)

        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = self.activation(x)
        return x