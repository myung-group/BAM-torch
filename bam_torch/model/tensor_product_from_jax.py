import torch
import math
from fractions import Fraction
from typing import List, Optional, Tuple
from e3nn.o3 import Irreps, Irrep
import e3nn.o3
import functools


def from_chunks(
    irreps: Irreps,
    chunks: List[Optional[torch.Tensor]],
    leading_shape: Tuple[int, ...],
    dtype=None,
) -> torch.Tensor:
    """
    Create a concatenated tensor from a list of tensors based on irreps.

    Args:
        irreps (Irreps): Irreducible representations.
        chunks (list of optional torch.Tensor): List of tensors.
        leading_shape (tuple of int): Leading shape of the tensors (excluding irreps).
        dtype (torch.dtype, optional): Data type of the output tensor.

    Returns:
        torch.Tensor: Concatenated tensor.
    """
    irreps = Irreps(irreps)
    
    if len(irreps) != len(chunks):
        raise ValueError(f"from_chunks_torch: len(irreps) != len(chunks), {len(irreps)} != {len(chunks)}")

    if not all(x is None or isinstance(x, torch.Tensor) for x in chunks):
        raise ValueError(f"from_chunks_torch: chunks contains non-tensor elements {[type(x) for x in chunks]}")

    if not all(
        x is None or x.shape == leading_shape + (mul, ir.dim)
        for x, (mul, ir) in zip(chunks, irreps)
    ):
        raise ValueError(
            f"from_chunks_torch: chunks shapes {[None if x is None else x.shape for x in chunks]} "
            f"incompatible with leading shape {leading_shape} and irreps {irreps}. "
            f"Expecting {[leading_shape + (mul, ir.dim) for (mul, ir) in irreps]}."
        )

    # Infer dtype from first non-None tensor
    if dtype is None:
        for x in chunks:
            if x is not None:
                dtype = x.dtype
                break

    if dtype is None:
        raise ValueError("from_chunks_torch: Need to specify dtype if chunks is empty or contains only None.")

    # Concatenate non-None tensors, filling None entries with zeros
    if irreps.dim > 0:
        array = torch.cat(
            [
                (torch.zeros(leading_shape + (mul_ir.dim,), dtype=dtype) if x is None else x.reshape(leading_shape + (mul_ir.dim,)))
                for mul_ir, x in zip(irreps, chunks)
            ],
            dim=-1,
        )
    else:
        array = torch.zeros(leading_shape + (0,), dtype=dtype)

    return array

def su2_generators(j: float, device=None, dtype=torch.float32) -> torch.Tensor:
    """Return the generators of the SU(2) group of dimension `2j+1` (Torch version)."""
    m = torch.arange(-j, j, dtype=dtype, device=device)
    raising = torch.diag(-torch.sqrt(j * (j + 1) - m * (m + 1)), diagonal=-1)

    m = torch.arange(-j + 1, j + 1, dtype=dtype, device=device)
    lowering = torch.diag(torch.sqrt(j * (j + 1) - m * (m - 1)), diagonal=1)

    m = torch.arange(-j, j + 1, dtype=dtype, device=device)
    return torch.stack([
        0.5 * (raising + lowering),  # x (usually)
        torch.diag(1j * m),  # z (usually)
        -0.5j * (raising - lowering),  # -y (usually)
    ], dim=0)

def su2_clebsch_gordan(j1: float, j2: float, j3: float, device=None, dtype=torch.float32) -> torch.Tensor:
    """Calculates the Clebsch-Gordan matrix (Torch version)."""
    assert isinstance(j1, (int, float))
    assert isinstance(j2, (int, float))
    assert isinstance(j3, (int, float))
    
    shape = (int(2 * j1 + 1), int(2 * j2 + 1), int(2 * j3 + 1))
    mat = torch.zeros(shape, dtype=dtype, device=device)
    
    if int(2 * j3) in range(int(2 * abs(j1 - j2)), int(2 * (j1 + j2)) + 1, 2):
        for m1 in (x / 2 for x in range(-int(2 * j1), int(2 * j1) + 1, 2)):
            for m2 in (x / 2 for x in range(-int(2 * j2), int(2 * j2) + 1, 2)):
                if abs(m1 + m2) <= j3:
                    mat[int(j1 + m1), int(j2 + m2), int(j3 + m1 + m2)] = _su2_cg(
                        (j1, m1), (j2, m2), (j3, m1 + m2), device, dtype
                    )
    return mat / math.sqrt(2 * j3 + 1)

def _su2_cg(idx1, idx2, idx3, device=None, dtype=torch.float32) -> torch.Tensor:
    """Calculates the Clebsch-Gordan coefficient (Torch version)."""
    j1, m1 = idx1
    j2, m2 = idx2
    j3, m3 = idx3

    if m3 != m1 + m2:
        return torch.tensor(0.0, dtype=dtype, device=device)
    
    vmin = int(max([-j1 + j2 + m3, -j1 + m1, 0]))
    vmax = int(min([j2 + j3 + m1, j3 - j1 + j2, j3 + m3]))
    
    def f(n):
        assert n == round(n)
        return math.factorial(round(n))
    
    C = (
        (2.0 * j3 + 1.0)
        * Fraction(
            f(j3 + j1 - j2)
            * f(j3 - j1 + j2)
            * f(j1 + j2 - j3)
            * f(j3 + m3)
            * f(j3 - m3),
            f(j1 + j2 + j3 + 1) * f(j1 - m1) * f(j1 + m1) * f(j2 - m2) * f(j2 + m2),
        )
    ) ** 0.5
    
    S = 0
    for v in range(vmin, vmax + 1):
        S += (-1.0) ** (v + j2 + m2) * Fraction(
            f(j2 + j3 + m1 - v) * f(j1 - m1 + v),
            f(v) * f(j3 - j1 + j2 - v) * f(j3 + m3 - v) * f(v + j1 - j2 - m3),
        )
    C = C * S
    return torch.tensor(float(C), dtype=dtype, device=device)

def change_basis_real_to_complex(l: int, device="cpu") -> torch.Tensor:
    """
    Constructs the change-of-basis matrix from real to complex spherical harmonics.

    Args:
        l (int): Degree of the spherical harmonics.
        device (str): Device to place the tensor on.

    Returns:
        torch.Tensor: Change-of-basis matrix (complex).
    """
    q = torch.zeros((2 * l + 1, 2 * l + 1), dtype=torch.complex128, device=device)
    for m in range(-l, 0):
        q[l + m, l + abs(m)] = 1 / torch.sqrt(torch.tensor(2.0, dtype=torch.float64, device=device))
        q[l + m, l - abs(m)] = -1j / torch.sqrt(torch.tensor(2.0, dtype=torch.float64, device=device))
    q[l, l] = 1.0

    for m in range(1, l + 1):
        q[l + m, l + abs(m)] = (-1) ** m / torch.sqrt(torch.tensor(2.0, dtype=torch.float64, device=device))
        q[l + m, l - abs(m)] = 1j * (-1) ** m / torch.sqrt(torch.tensor(2.0, dtype=torch.float64, device=device))
    # Factor to make Clebsch-Gordan coefficients real
    return (-1j) ** l * q

@functools.lru_cache(maxsize=None)
def clebsch_gordan(l1: int, l2: int, l3: int, device="cpu") -> torch.Tensor:
    """
    Computes the Clebsch-Gordan coefficients for the real irreps of SO(3).

    Args:
        l1 (int): Degree of first irrep.
        l2 (int): Degree of second irrep.
        l3 (int): Degree of third irrep.
        device (str): Device to place the tensor on.

    Returns:
        torch.Tensor: Clebsch-Gordan coefficients.
    """
    # Clebsch-Gordan coefficients from SU(2)
    C = su2_clebsch_gordan(l1, l2, l3).to(device).to(torch.complex128)

    # Change-of-basis matrices
    Q1 = change_basis_real_to_complex(l1, device)
    Q2 = change_basis_real_to_complex(l2, device)
    Q3 = change_basis_real_to_complex(l3, device)

    # Apply change-of-basis transformation
    C = torch.einsum("ij,kl,mn,ikn->jlm", Q1, Q2, Q3.conj().T, C)

    # Ensure that the result is real
    assert torch.all(torch.abs(torch.imag(C)) < 1e-5)

    return torch.real(C)  # Only return the real part

def generators(l: int, device='cpu') -> torch.Tensor:
    """Generators of the real irreducible representations of SO(3)."""
    from Su2_Clebsch_Gordan_Torch import su2_generators
    
    X = su2_generators(l).to(device)
    Q = change_basis_real_to_complex(l, device)
    X = Q.conj().T @ X @ Q
    
    assert torch.all(torch.abs(torch.imag(X)) < 1e-5)
    return torch.real(X)

def split_tensor_by_irreps(tensor: torch.Tensor, irreps: Irreps):
    """   
    Args:
        tensor (torch.Tensor): (batch, feature_dim)
        irreps (Irreps): e3nn.o3.Irreps, "4x0e + 4x1o + 3x2e"

    Returns:
        List[torch.Tensor]: (batch, mul, ir.dim)
    """
    batch_size, feature_dim = tensor.shape
    chunks = []
    idx = 0 

    for mul, ir in irreps:
        dim = mul * ir.dim 
        chunk = tensor[:, idx:idx + dim].reshape(batch_size, mul, ir.dim)
        chunks.append(chunk)
        idx += dim 

    return chunks

def _validate_filter_ir_out(filter_ir_out):
    unique_irreps = sorted({ir for _, ir in filter_ir_out}) 
    return unique_irreps

def sort_irreps_chunks(irreps, chunks, leading_shape):
    irreps, p, inv = irreps.sort()
    return from_chunks(
        irreps,
        [chunks[i] for i in inv],
        leading_shape,
    )

class TensorProductTorch(torch.nn.Module):
    """Tensor product reduced into irreps (Torch version)."""
    
    def __init__(self, irreps_in1: Irreps, irreps_in2: Irreps, filter_ir_out: Irreps,
                 irrep_normalization: Optional[str] = "component", regroup_output: bool = True):
        super().__init__()
        self.irreps_in1 = Irreps(irreps_in1)
        self.irreps_in2 = Irreps(irreps_in2)
        self.filter_ir_out = _validate_filter_ir_out(filter_ir_out)
        self.irrep_normalization = irrep_normalization
        self.regroup_output = regroup_output

    def forward(self, input1: torch.Tensor, input2: torch.Tensor) -> torch.Tensor:
        irreps_out = []
        chunks = []
        input1_chunks = split_tensor_by_irreps(input1, self.irreps_in1)
        input2_chunks = split_tensor_by_irreps(input2, self.irreps_in2)
        
        for (mul_1, ir_1), x1 in zip(self.irreps_in1, input1_chunks):
            for (mul_2, ir_2), x2 in zip(self.irreps_in2, input2_chunks):
                for ir_out in ir_1 * ir_2:
                    if self.filter_ir_out is not None and ir_out not in self.filter_ir_out:
                        continue
    
                    irreps_out.append((mul_1 * mul_2, ir_out))
                    
                    if x1 is not None and x2 is not None:
                        cg = clebsch_gordan(ir_1.l, ir_2.l, ir_out.l)

                        if self.irrep_normalization == "component":
                            cg = cg * torch.sqrt(torch.tensor(ir_out.dim, dtype=x1.dtype))
                        elif self.irrep_normalization == "norm":
                            cg = cg * torch.sqrt(torch.tensor(ir_1.dim * ir_2.dim, dtype=x1.dtype))
                        elif self.irrep_normalization == "none":
                            pass
                        else:
                            raise ValueError(f"irrep_normalization={self.irrep_normalization} not supported")

                        cg = cg.to(x1.dtype)
                        cg = cg.to(x1.device)
                        chunk = torch.einsum("...ui , ...vj , ijk -> ...uvk", x1, x2, cg)
                        chunk = chunk.reshape(chunk.shape[:-3] + (mul_1 * mul_2, ir_out.dim))
                    else:
                        chunk = None
                    
                    chunks.append(chunk)
        
        output = from_chunks(irreps_out, chunks, leading_shape=(input1.shape[0], ))
        output = sort_irreps_chunks(Irreps(irreps_out), chunks, leading_shape=(input1.shape[0], ))
        return output

def tensor_irreps_array_product(tensor, irreps_array, irreps):
    x1 = irreps_array # mji
    x1_chunks = split_tensor_by_irreps(x1, irreps)
    x2 = tensor # mix
    """
    num = 0
    chunks = []
    for (mul, ir), x in zip(irreps, x1_chunks):
        mix = x2[:, num:num+mul]
        chunks.append(x*mix.unsqueeze(-1))
        num += mul
    """
    offsets = torch.cumsum(torch.tensor([0] + [mul for mul, _ in irreps[:-1]]), dim=0)
    chunks = [
        x * x2[:, offset:offset+mul].unsqueeze(-1)
        for (mul, _), x, offset in zip(irreps, x1_chunks, offsets)
    ]
    return from_chunks(irreps, chunks, leading_shape=(x1.shape[0], ))

