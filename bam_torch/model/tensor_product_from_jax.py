import torch
import math
from fractions import Fraction
from typing import List, Optional, Tuple
from e3nn.o3 import Irreps, Irrep
from e3nn import o3
import functools
from e3nn.util.jit import compile_mode


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

@compile_mode("script")
class FromChunks(torch.nn.Module):
    def __init__(
        self,
        irreps: Irreps,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        self.irreps = Irreps(irreps)
        self.irreps_dims = [(mul, ir.dim) for mul, ir in self.irreps]
        self.dtype = dtype

    def forward(self, chunks: List[torch.Tensor], leading_shape: Tuple[int]) -> torch.Tensor:
        if len(self.irreps_dims) != len(chunks):
            raise ValueError(f"from_chunks_torch: len(irreps) != len(chunks), {len(self.irreps_dims)} != {len(chunks)}")

        for x in chunks:
            if x is not None and not isinstance(x, torch.Tensor):
                raise ValueError(f"from_chunks_torch: chunks contains non-tensor elements")

        if not all(
            x is None or x.shape == leading_shape + (mul, dim)
            for x, (mul, dim) in zip(chunks, self.irreps_dims)
        ):
            raise ValueError(
                f"from_chunks_torch: chunks shapes {[None if x is None else x.shape for x in chunks]} "
                f"incompatible with leading shape {leading_shape} and irreps_dims {self.irreps_dims}. "
                f"Expecting {[leading_shape + (mul, dim) for (mul, dim) in self.irreps_dims]}."
            )

        non_none_chunks = []
        for x in chunks:
            if x is not None:
                non_none_chunks.append(x)
        if len(non_none_chunks) == 0:
            raise ValueError("no tensor found in chunks to infer dtype")

        dtype = non_none_chunks[0].dtype

        if dtype is None:
            raise ValueError("from_chunks_torch: Need to specify dtype if chunks is empty or contains only None.")

        if sum(mul * dim for mul, dim in self.irreps_dims) > 0:
            array = torch.cat(
                [
                    (torch.zeros(leading_shape + (mul * dim,), dtype=dtype, device=chunks[0].device)
                     if x is None else x.reshape(leading_shape + (mul * dim,)))
                    for (mul, dim), x in zip(self.irreps_dims, chunks)
                ],
                dim=-1,
            )
        else:
            array = torch.zeros(leading_shape + (0,), dtype=dtype, device=chunks[0].device)

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


def split_tensor_by_irreps(
        tensor: torch.Tensor, 
        irreps: List[Tuple[int, int]],
) -> List[torch.Tensor]:
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
    for mul, ir_dim in irreps:
        dim = mul * ir_dim 
        chunk = tensor[:, idx:idx + dim].reshape(batch_size, mul, ir_dim)
        chunks.append(chunk)
        idx += dim 

    return chunks

def get_irreps_dim(irreps: str) -> int:
    if '0' in irreps:
        return 1
    elif '1' in irreps:
        return 3
    elif '2' in irreps:
        return 5
    elif '3' in irreps:
        return 7
    elif '4' in irreps:
        return 9


@compile_mode("script")
class SplitTensorByIrreps(torch.nn.Module):
    def __init__(self, irreps: o3.Irreps):
        self.irreps = irreps
        self.irreps_dims = [(mul, ir.dim) for mul, ir in self.irreps]
        super().__init__()

    
    def forward(self, tensor: torch.Tensor):
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

        for mul, ir_dim in self.irreps_dims:
            #ir_dim = get_irreps_dim(f"{ir}")
            dim = mul * ir_dim
            chunk = tensor[:, idx:idx + dim].reshape(batch_size, mul, ir_dim)
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

@compile_mode('script')
class SortIrrepsChunks(torch.nn.Module):
    def __init__(self, irreps: o3.Irreps):
        super().__init__()
        self.irreps, p, self.inv = o3.Irreps(irreps).sort()
        self.from_chunks = FromChunks(self.irreps)
    
    def forward(self, 
        chunks: List[torch.Tensor],
        leading_shape: Tuple[int]
    ) -> torch.Tensor:
        return self.from_chunks([chunks[i] for i in self.inv], leading_shape)


@compile_mode('script')
class TensorProductTorch(torch.nn.Module):
    def __init__(self, irreps_in1: Irreps, irreps_in2: Irreps, filter_ir_out: Irreps,
                 irrep_normalization: Optional[str] = "component"):
        super().__init__()
        self.irrep_normalization = irrep_normalization

        #self.split_tensor_irreps_in1 = SplitTensorByIrreps(irreps_in1)
        #self.split_tensor_irreps_in2 = SplitTensorByIrreps(irreps_in2)

        self.filter_ir_out = _validate_filter_ir_out(filter_ir_out)

        self.irrep_pairs = []  # List of (mul1, mul2, dim1, dim2, l1, l2, l_out, dim_out)
        self.irreps_out = []
        for i, (mul_1, ir_1) in enumerate(irreps_in1):
            for j, (mul_2, ir_2) in enumerate(irreps_in2):
                for ir_out in ir_1 * ir_2:
                    if self.filter_ir_out is not None and ir_out not in self.filter_ir_out:
                        tag = 'false'
                        self.irrep_pairs.append((tag, i, j, mul_1, mul_2, ir_1.dim, ir_2.dim, ir_1.l, ir_2.l, ir_out.dim, ir_out.l))
                    else:
                        tag = 'true'
                        self.irrep_pairs.append((tag, i, j, mul_1, mul_2, ir_1.dim, ir_2.dim, ir_1.l, ir_2.l, ir_out.dim, ir_out.l))
                        self.irreps_out.append((mul_1 * mul_2, str(ir_out)))
        self.irreps_in1_dims = [(x.mul, x.ir.dim) for x in irreps_in1]
        self.irreps_in2_dims = [(x.mul, x.ir.dim) for x in irreps_in2]
        self.from_chunks = FromChunks(self.irreps_out)
        self.sort_irreps_chunks = SortIrrepsChunks(self.irreps_out)

    def forward(self, input1: torch.Tensor, input2: torch.Tensor) -> torch.Tensor:

        x1 = split_tensor_by_irreps(input1, self.irreps_in1_dims)
        x2 = split_tensor_by_irreps(input2, self.irreps_in2_dims)
        chunks = []

        for tag, x1_idx, x2_idx, mul_1, mul_2, ir_1_dim, ir_2_dim, ir_1_l, ir_2_l, ir_out_dim, ir_out_l in self.irrep_pairs:
            if tag == 'true':
                if x1[x1_idx] is not None and x2[x2_idx] is not None:
                    cg = clebsch_gordan(ir_1_l, ir_2_l, ir_out_l)
                    if self.irrep_normalization == "component":
                        cg = cg * torch.sqrt(torch.tensor(ir_out_dim, dtype=x1[x1_idx].dtype))
                    elif self.irrep_normalization == "norm":
                        cg = cg * torch.sqrt(torch.tensor(ir_1_dim * ir_2_dim, dtype=x1[x1_idx].dtype))
                    elif self.irrep_normalization == "none":
                        pass
                    else:
                        raise ValueError(f"irrep_normalization={self.irrep_normalization} not supported")
                    cg = cg.to(x1[x1_idx].dtype).to(x1[x1_idx].device)

                    chunk = torch.einsum("...ui , ...vj , ijk -> ...uvk", x1[x1_idx], x2[x2_idx], cg)
                    chunk = chunk.reshape(chunk.shape[:-3] + (mul_1 * mul_2, ir_out_dim))
                else:
                    chunk = None
                
                chunks.append(chunk)
                
        output = self.from_chunks(chunks, leading_shape=(input1.shape[0], ))
        output = self.sort_irreps_chunks(chunks, leading_shape=(input1.shape[0], ))
        return output


@compile_mode('script')
class TensorProductTorch_(torch.nn.Module):
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
        print(len(input1_chunks), len(input2_chunks))
        
        num = 1
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
                    print(num)
                    num+=1
                    
                    chunks.append(chunk)
        print(len(chunks), chunks[0].shape)
        print("irreps_out", irreps_out)
        output = from_chunks(irreps_out, chunks, leading_shape=(input1.shape[0], ))
        output = sort_irreps_chunks(Irreps(irreps_out), chunks, leading_shape=(input1.shape[0], ))
        return output
    

def tensor_irreps_array_product(tensor, irreps_array, irreps):
    x1 = irreps_array # mji
    irreps_dims = [(x.mul, x.ir.dim) for x in irreps]
    x1_chunks = split_tensor_by_irreps(x1, irreps_dims)
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

@compile_mode("script")
class TensorIrrepsArrayProduct(torch.nn.Module):
    def __init__(self, irreps: o3.Irreps):
        super().__init__()
        self.irreps_dims = [(mul, ir.dim) for mul, ir in irreps]
        self.from_chunks = FromChunks(irreps)
    
    def forward(self,
        tensor: torch.Tensor,
        irreps_array: torch.Tensor
    ) -> torch.Tensor:
        x1 = irreps_array
        x1_chunks = split_tensor_by_irreps(x1, self.irreps_dims)
        x2 = tensor
        offsets = torch.cumsum(torch.tensor([0] + [mul for mul, _ in self.irreps_dims]), dim=0)
        chunks = [
            x * x2[:, offset:offset+mul].unsqueeze(-1)
            for (mul, _), x, offset in zip(self.irreps_dims, x1_chunks, offsets)
        ]
        return self.from_chunks(chunks, leading_shape=(x1.shape[0], ))

