from typing import Callable, List, NamedTuple, Optional, Tuple, Union
from e3nn.o3 import Irrep, Irreps
import torch
from e3nn import o3
from bam_torch.model.tensor_product_from_jax import split_tensor_by_irreps
from e3nn.util.jit import compile_mode


def irreps_filter(
    irreps: Union["Irreps", List[Irrep]] = None,
    keep: Union["Irreps", List[Irrep]] = None,
    drop: Union["Irreps", List[Irrep]] = None,
    lmax: int = None,
) -> "Irreps":
    r"""Filter the irreps.

    Args:
        keep (`Irreps` or list of `Irrep` or function): list of irrep to keep
        drop (`Irreps` or list of `Irrep` or function): list of irrep to drop
        lmax (int): maximum :math:`l` value

    Returns:
        `Irreps`: filtered irreps

    Examples:
        >>> Irreps("1e + 2e + 0e").filter(keep=["0e", "1e"])
        1x1e+1x0e

        >>> Irreps("1e + 2e + 0e").filter(keep="2e + 2x1e")
        1x1e+1x2e

        >>> Irreps("1e + 2e + 0e").filter(drop="2e + 2x1e")
        1x0e

        >>> Irreps("1e + 2e + 0e").filter(lmax=1)
        1x1e+1x0e
    """
    if keep is None and drop is None and lmax is None:
        return irreps
    if keep is not None and drop is not None:
        raise ValueError("Cannot specify both keep and drop")
    if keep is not None and lmax is not None:
        raise ValueError("Cannot specify both keep and lmax")
    if drop is not None and lmax is not None:
        raise ValueError("Cannot specify both drop and lmax")

    if keep is not None:
        if isinstance(keep, str):
            keep = Irreps(keep)
        if isinstance(keep, Irrep):
            keep = [keep]
        if callable(keep):
            return Irreps([mul_ir for mul_ir in irreps if keep(mul_ir)])
        keep = {Irrep(ir) for ir in keep}
        return Irreps([(mul, ir) for mul, ir in irreps if ir in keep])

    if drop is not None:
        if isinstance(drop, str):
            drop = Irreps(drop)
        if isinstance(drop, Irrep):
            drop = [drop]
        if callable(drop):
            return Irreps([mul_ir for mul_ir in irreps if not drop(mul_ir)])
        drop = {Irrep(ir) for ir in drop}
        return Irreps([(mul, ir) for mul, ir in irreps if ir not in drop])

    if lmax is not None:
        return Irreps([(mul, ir) for mul, ir in irreps if ir.l <= lmax])


def concatenate_irreps_tensor(
    tensors: List[torch.Tensor],
    irreps_list: List[o3.Irreps],
    axis: int = -1
) -> tuple[o3.Irreps, torch.Tensor]:
    if axis < 0:
        axis += tensors[0].ndim
    concat_irreps = sum(irreps_list[1:], start=irreps_list[0])
    concat_tensor = torch.cat(tensors, dim=axis)
    return concat_tensor, concat_irreps

def tensor_regroup_by_irreps(
    tensor: torch.tensor, 
    irreps: Irreps, 
    dtype=None
):
    leading_shape = tensor.shape[:-1]
    chunks = split_tensor_by_irreps(tensor, irreps)
    irreps, p, inv = irreps.sort()
    r_chunks = [chunks[i] for i in inv]
    if dtype is None:
        dtype = next((x.dtype for x in chunks if x is not None), None)
        
    array = torch.cat([
                torch.zeros(leading_shape + (mul_ir.dim,), dtype=dtype) if x is None
                else x.reshape(leading_shape + (mul_ir.dim,))
                for mul_ir, x in zip(irreps, r_chunks)
            ], dim=-1
    )
    return array, irreps.simplify()

