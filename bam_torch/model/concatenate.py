import torch

from e3nn import o3
from e3nn.o3 import Irrep, Irreps
from e3nn.util.jit import compile_mode

from typing import Callable, List, NamedTuple, Optional, Tuple, Union


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
        

@compile_mode("script")
class TensorRegroupByIrreps(torch.nn.Module):
    def __init__(self, irreps: o3.Irreps):
        super().__init__()
        
        self.irreps_dims = [(mul, ir.dim) for mul, ir in irreps]
        self.irreps, _, self.inv = irreps.sort()
        self.sorted_irreps_dims = [(mul.dim) for mul in self.irreps]
        #self.simplified_irreps = self.irreps.simplify()
    
    def forward(self, tensor: torch.Tensor, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        leading_shape = tensor.shape[:-1]
        chunks = split_tensor_by_irreps(tensor, self.irreps_dims)
        r_chunks = [chunks[i] for i in self.inv]
        if dtype is None:
            #dtype = next((x.dtype for x in chunks if x is not None), None)
            for x in chunks:
                if x is not None:
                    dtype = x.dtype
                    break
            
        array = torch.cat([
                    torch.zeros(leading_shape + (mul_ir_dim,), dtype=dtype) if x is None
                    else x.reshape(leading_shape + (mul_ir_dim,))
                    for mul_ir_dim, x in zip(self.sorted_irreps_dims, r_chunks)
                ], dim=-1
        )
        return array #, self.simplified_irreps


@compile_mode("script")
class ConcatenateIrrepsTensor(torch.nn.Module):
    def __init__(self, irreps_list= List[o3.Irreps]):
        super().__init__()
        self.concat_irreps = sum(irreps_list[1:], start=irreps_list[0])
    
    def forward(
        self,
        tensors: List[torch.Tensor],
        axis: int = -1
    ) -> torch.Tensor:
        if axis < 0:
            axis += tensors[0].ndim
        concat_tensor = torch.cat(tensors, dim=axis)
        return concat_tensor
    
    def get_concatenated_irreps(self):
        return self.concat_irreps


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