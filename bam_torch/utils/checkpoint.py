"""Load checkpoints across equivariant backends (e3nn / OEQ / cuEquivariance).

e3nn and OpenEquivariance store a Linear or FullyConnectedTensorProduct weight
as [N]; cuEquivariance stores the same weights as [1, N]. e3nn also keeps empty
biases, `output_mask` and Wigner-3j (`_w3j_*`) buffers where cuEquivariance
keeps constant graph buffers. None of these hold trained values, so reshaping
the weights is all it takes to load a RACE checkpoint trained with one backend
into a model built with another. (MACE's symmetric contraction stores its
weights differently per backend and is not converted here.)
"""
from typing import Any, Dict, List

import torch


def _holds_no_weights(name: str, tensor: torch.Tensor) -> bool:
    return (
        tensor.numel() == 0
        or name.endswith("output_mask")
        or "._w3j_" in name
        or ".graphs." in name
    )


def _fit(tensor: torch.Tensor, shape: torch.Size) -> torch.Tensor:
    if tensor.shape != shape and tensor.numel() == shape.numel():
        # detach: stored EMA shadows may carry requires_grad, and torch_ema
        # deep-copies its state, which only works on graph leaves.
        return tensor.detach().reshape(shape)
    return tensor


def match_state_dict(
    state_dict: Dict[str, torch.Tensor], model: torch.nn.Module
) -> Dict[str, torch.Tensor]:
    """Return `state_dict` laid out for `model`'s backend.

    Anything that is not a backend difference is left alone, so a real
    architecture mismatch still fails in `load_state_dict`.
    """
    target = model.state_dict()
    out = {}
    for name, tensor in state_dict.items():
        if name in target:
            out[name] = _fit(tensor, target[name].shape)
        elif not _holds_no_weights(name, tensor):
            out[name] = tensor
    for name, tensor in target.items():
        if name not in out and _holds_no_weights(name, tensor):
            out[name] = tensor
    return out


def match_parameter_list(
    tensors: List[torch.Tensor], model: torch.nn.Module
) -> List[torch.Tensor]:
    """Reshape tensors stored in `model.parameters()` order (EMA shadows)."""
    shapes = [p.shape for p in model.parameters()]
    if len(tensors) != len(shapes):
        return tensors
    return [_fit(t, s) for t, s in zip(tensors, shapes)]


def match_ema_state(
    ema_state: Dict[str, Any], model: torch.nn.Module
) -> Dict[str, Any]:
    ema_state = dict(ema_state)
    for key in ("shadow_params", "collected_params"):
        if ema_state.get(key):
            ema_state[key] = match_parameter_list(ema_state[key], model)
    return ema_state


def match_optimizer_state(
    opt_state: Dict[str, Any], model: torch.nn.Module
) -> Dict[str, Any]:
    """Reshape per-parameter optimizer state (e.g. Adam moments)."""
    shapes = [p.shape for p in model.parameters()]
    state = {}
    for index, param_state in opt_state.get("state", {}).items():
        if isinstance(index, int) and index < len(shapes):
            param_state = {
                k: _fit(v, shapes[index])
                if torch.is_tensor(v) and v.dim() > 0 else v
                for k, v in param_state.items()
            }
        state[index] = param_state
    return dict(opt_state, state=state)
