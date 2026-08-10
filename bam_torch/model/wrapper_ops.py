"""
Wrappers around o3 / cuequivariance / openequivariance modules.

Adapted from MACE's mace/modules/wrapper_ops.py, with BAM-specific additions:
- BAM's `FullTensorProduct` wrapper (no weights) and its cuet counterpart.
- `SymmetricContractionWrapper` keeps the cuet adapter that converts one-hot
  attrs to indices and handles the `mul_ir` layout transpose.
"""

import dataclasses
import itertools
import types
from typing import Iterator, List, Optional

import numpy as np
import torch
from e3nn import o3

from .symmetric_contraction import SymmetricContraction
from ._sub import FullTensorProduct as o3_FullTensorProduct

try:
    import cuequivariance as cue
    import cuequivariance_torch as cuet

    CUET_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    CUET_AVAILABLE = False

try:
    import openequivariance as oeq

    OEQ_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    OEQ_AVAILABLE = False


if CUET_AVAILABLE:

    class O3_e3nn(cue.O3):
        def __mul__(  # pylint: disable=no-self-argument
            rep1: "O3_e3nn", rep2: "O3_e3nn"
        ) -> Iterator["O3_e3nn"]:
            return [O3_e3nn(l=ir.l, p=ir.p) for ir in cue.O3.__mul__(rep1, rep2)]

        @classmethod
        def clebsch_gordan(
            cls, rep1: "O3_e3nn", rep2: "O3_e3nn", rep3: "O3_e3nn"
        ) -> np.ndarray:
            rep1, rep2, rep3 = cls._from(rep1), cls._from(rep2), cls._from(rep3)

            if rep1.p * rep2.p == rep3.p:
                return o3.wigner_3j(rep1.l, rep2.l, rep3.l).numpy()[None] * np.sqrt(
                    rep3.dim
                )
            return np.zeros((0, rep1.dim, rep2.dim, rep3.dim))

        def __lt__(  # pylint: disable=no-self-argument
            rep1: "O3_e3nn", rep2: "O3_e3nn"
        ) -> bool:
            rep2 = rep1._from(rep2)
            return (rep1.l, rep1.p) < (rep2.l, rep2.p)

        @classmethod
        def iterator(cls) -> Iterator["O3_e3nn"]:
            for l in itertools.count(0):
                yield O3_e3nn(l=l, p=1 * (-1) ** l)
                yield O3_e3nn(l=l, p=-1 * (-1) ** l)

    class cuetFullTensorProduct(torch.nn.Module):
        """cuet equivalent of o3.FullTensorProduct (no weights)."""

        def __init__(
            self,
            irreps_in1: "cue.Irreps",
            irreps_in2: "cue.Irreps",
            filter_irreps_out: "cue.Irreps" = None,
            *,
            layout: "cue.IrrepsLayout" = None,
            device: torch.device = None,
        ):
            super().__init__()
            e = cue.descriptors.full_tensor_product(
                irreps_in1, irreps_in2, filter_irreps_out
            )
            self.irreps_in1 = irreps_in1
            self.irreps_in2 = irreps_in2
            self.irreps_out = e.operands[-1].irreps
            self.f = cuet.EquivariantTensorProduct(e, layout=layout, device=device)

        def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
            return self.f(x1, x2)

else:
    print(
        "cuequivariance / cuequivariance_torch not available. "
        "Cuequivariance acceleration will be disabled."
    )

if not OEQ_AVAILABLE:
    print(
        "openequivariance not available. OEQ acceleration will be disabled."
    )


@dataclasses.dataclass
class CuEquivarianceConfig:
    """Configuration for cuequivariance acceleration."""

    enabled: bool = False
    layout: str = "mul_ir"  # One of: mul_ir, ir_mul
    layout_str: str = "mul_ir"
    group: str = "O3"
    optimize_all: bool = False  # Set to True to enable all optimizations
    optimize_linear: bool = False
    optimize_channelwise: bool = False
    optimize_symmetric: bool = False
    optimize_fctp: bool = False
    conv_fusion: bool = False  # Set to True to enable cuet conv fusion

    def __post_init__(self):
        if self.enabled and CUET_AVAILABLE:
            self.layout_str = self.layout
            self.layout = getattr(cue, self.layout)
            self.group = (
                O3_e3nn if self.group == "O3_e3nn" else getattr(cue, self.group)
            )
        if not CUET_AVAILABLE:
            self.enabled = False


@dataclasses.dataclass
class OEQConfig:
    """Configuration for openequivariance acceleration."""

    enabled: bool = False
    optimize_all: bool = False
    optimize_channelwise: bool = False
    conv_fusion: Optional[str] = None  # None | "atomic"

    def __post_init__(self):
        if not OEQ_AVAILABLE:
            self.enabled = False


class Linear:
    """Returns either a cuet.Linear or o3.Linear based on config."""

    def __new__(
        cls,
        irreps_in: o3.Irreps,
        irreps_out: o3.Irreps,
        shared_weights: bool = True,
        internal_weights: bool = True,
        cueq_config: Optional[CuEquivarianceConfig] = None,
    ):
        if (
            CUET_AVAILABLE
            and cueq_config is not None
            and cueq_config.enabled
            and (cueq_config.optimize_all or cueq_config.optimize_linear)
        ):
            return cuet.Linear(
                cue.Irreps(cueq_config.group, irreps_in),
                cue.Irreps(cueq_config.group, irreps_out),
                layout=cueq_config.layout,
                shared_weights=shared_weights,
                method="naive",
            )

        return o3.Linear(
            irreps_in,
            irreps_out,
            shared_weights=shared_weights,
            internal_weights=internal_weights,
        )


def with_cueq_conv_fusion(conv_tp: torch.nn.Module) -> torch.nn.Module:
    """Wrap a cuet `SegmentedPolynomial` to expose a (node_feats, edge_attrs,
    weights, edge_index) signature, computing gather + TP + scatter in one shot.
    """
    conv_tp.original_forward = conv_tp.forward
    num_segment = conv_tp.m.buffer_num_segments[0]
    num_operands = conv_tp.m.operand_extent
    conv_tp.weight_numel = num_segment * num_operands

    def forward(
        self,
        node_feats: torch.Tensor,
        edge_attrs: torch.Tensor,
        tp_weights: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        sender = edge_index[0]
        receiver = edge_index[1]
        return self.original_forward(
            [tp_weights, node_feats, edge_attrs],
            {1: sender},
            {0: node_feats},
            {0: receiver},
        )[0]

    conv_tp.forward = types.MethodType(forward, conv_tp)
    return conv_tp


class TensorProduct:
    """Wrapper around o3.TensorProduct / cuet.ChannelWiseTensorProduct /
    oeq.TensorProduct (or oeq.TensorProductConv when conv fusion is on).

    Default forward signature is `(x1, x2, weights) -> y`. When cuet conv
    fusion or oeq's atomic conv fusion is enabled, the forward signature
    becomes `(node_feats, edge_attrs, weights, edge_index) -> y`.
    """

    def __new__(
        cls,
        irreps_in1: o3.Irreps,
        irreps_in2: o3.Irreps,
        irreps_out: o3.Irreps,
        instructions: Optional[List] = None,
        shared_weights: bool = False,
        internal_weights: bool = False,
        use_conv_fusion: bool = False,
        cueq_config: Optional[CuEquivarianceConfig] = None,
        oeq_config: Optional[OEQConfig] = None,
    ):
        if (
            CUET_AVAILABLE
            and cueq_config is not None
            and cueq_config.enabled
            and (cueq_config.optimize_all or cueq_config.optimize_channelwise)
        ):
            if cueq_config.conv_fusion and use_conv_fusion:
                return with_cueq_conv_fusion(
                    cuet.SegmentedPolynomial(
                        cue.descriptors.channelwise_tensor_product(
                            cue.Irreps(cueq_config.group, irreps_in1),
                            cue.Irreps(cueq_config.group, irreps_in2),
                            cue.Irreps(cueq_config.group, irreps_out),
                        )
                        .flatten_coefficient_modes()
                        .squeeze_modes()
                        .polynomial,
                        math_dtype=torch.get_default_dtype(),
                        method="uniform_1d",
                    )
                )
            return cuet.ChannelWiseTensorProduct(
                cue.Irreps(cueq_config.group, irreps_in1),
                cue.Irreps(cueq_config.group, irreps_in2),
                cue.Irreps(cueq_config.group, irreps_out),
                layout=cueq_config.layout,
                shared_weights=shared_weights,
                internal_weights=internal_weights,
                dtype=torch.get_default_dtype(),
                math_dtype=torch.get_default_dtype(),
            )

        if (
            OEQ_AVAILABLE
            and oeq_config is not None
            and oeq_config.enabled
            and (oeq_config.optimize_all or oeq_config.optimize_channelwise)
        ):
            dtype = oeq.torch_to_oeq_dtype(torch.get_default_dtype())
            tpp = oeq.TPProblem(
                irreps_in1,
                irreps_in2,
                irreps_out,
                instructions,
                shared_weights=shared_weights,
                internal_weights=internal_weights,
                irrep_dtype=dtype,
                weight_dtype=dtype,
            )
            if oeq_config.conv_fusion is None:
                return oeq.TensorProduct(tpp)
            if oeq_config.conv_fusion == "atomic":
                return oeq.TensorProductConv(tpp, deterministic=False)
            raise ValueError(
                f"Unknown oeq conv_fusion option: {oeq_config.conv_fusion!r}"
            )

        return o3.TensorProduct(
            irreps_in1,
            irreps_in2,
            irreps_out,
            instructions=instructions,
            shared_weights=shared_weights,
            internal_weights=internal_weights,
        )


class FullTensorProduct:
    """Wrapper around o3.FullTensorProduct / cuet equivalent (no weights).

    z_{uv} = x_u (x) y_v
    """

    def __new__(
        cls,
        irreps_in1: o3.Irreps,
        irreps_in2: o3.Irreps,
        filter_ir_out: o3.Irreps,
        cueq_config: Optional[CuEquivarianceConfig] = None,
    ):
        if (
            CUET_AVAILABLE
            and cueq_config is not None
            and cueq_config.enabled
            and (cueq_config.optimize_all or cueq_config.optimize_fctp)
        ):
            return cuetFullTensorProduct(
                cue.Irreps(cueq_config.group, irreps_in1),
                cue.Irreps(cueq_config.group, irreps_in2),
                layout=cueq_config.layout,
            )

        return o3_FullTensorProduct(
            irreps_in1,
            irreps_in2,
            filter_ir_out,
        )


class FullyConnectedTensorProduct:
    """Wrapper around o3.FullyConnectedTensorProduct / cuet equivalent."""

    def __new__(
        cls,
        irreps_in1: o3.Irreps,
        irreps_in2: o3.Irreps,
        irreps_out: o3.Irreps,
        shared_weights: bool = True,
        internal_weights: bool = True,
        cueq_config: Optional[CuEquivarianceConfig] = None,
    ):
        if (
            CUET_AVAILABLE
            and cueq_config is not None
            and cueq_config.enabled
            and (cueq_config.optimize_all or cueq_config.optimize_fctp)
        ):
            return cuet.FullyConnectedTensorProduct(
                cue.Irreps(cueq_config.group, irreps_in1),
                cue.Irreps(cueq_config.group, irreps_in2),
                cue.Irreps(cueq_config.group, irreps_out),
                layout=cueq_config.layout,
                shared_weights=shared_weights,
                internal_weights=internal_weights,
                method="naive",
            )

        return o3.FullyConnectedTensorProduct(
            irreps_in1,
            irreps_in2,
            irreps_out,
            shared_weights=shared_weights,
            internal_weights=internal_weights,
        )


class SymmetricContractionWrapper:
    """Wrapper around BAM SymmetricContraction / cuet.SymmetricContraction.

    The cuet branch keeps a small forward adapter that:
      * converts one-hot `node_attrs` to integer indices (cuet API),
      * transposes from `mul_ir` to `ir_mul` when needed.
    """

    def __new__(
        cls,
        irreps_in: o3.Irreps,
        irreps_out: o3.Irreps,
        correlation: int,
        num_elements: Optional[int] = None,
        cueq_config: Optional[CuEquivarianceConfig] = None,
        oeq_config: Optional[OEQConfig] = None,  # pylint: disable=unused-argument
    ):
        if (
            CUET_AVAILABLE
            and cueq_config is not None
            and cueq_config.enabled
            and (cueq_config.optimize_all or cueq_config.optimize_symmetric)
        ):
            instance = cuet.SymmetricContraction(
                cue.Irreps(cueq_config.group, irreps_in),
                cue.Irreps(cueq_config.group, irreps_out),
                layout_in=cue.ir_mul,
                layout_out=cueq_config.layout,
                contraction_degree=correlation,
                num_elements=num_elements,
                original_mace=True,
                dtype=torch.get_default_dtype(),
                math_dtype=torch.get_default_dtype(),
            )
            instance.original_forward = instance.forward
            instance.layout = cueq_config.layout

            def cuet_forward(
                self, x: torch.Tensor, attrs: torch.Tensor
            ) -> torch.Tensor:
                if self.layout == cue.mul_ir:
                    x = torch.transpose(x, 1, 2)
                index_attrs = torch.nonzero(attrs)[:, 1].int()
                return self.original_forward(x.flatten(1), index_attrs)

            instance.forward = types.MethodType(cuet_forward, instance)
            return instance

        return SymmetricContraction(
            irreps_in=irreps_in,
            irreps_out=irreps_out,
            correlation=correlation,
            num_elements=num_elements,
        )


class TransposeIrrepsLayoutWrapper:
    """Wrapper around cuet.TransposeIrrepsLayout (no-op on plain e3nn)."""

    def __new__(
        cls,
        irreps: o3.Irreps,
        source: str,
        target: str,
        cueq_config: Optional[CuEquivarianceConfig] = None,
    ):
        if CUET_AVAILABLE and cueq_config is not None and cueq_config.enabled:
            if source == target:
                return None
            return cuet.TransposeIrrepsLayout(
                cue.Irreps(cueq_config.group, irreps),
                source=getattr(cue, source),
                target=getattr(cue, target),
                use_fallback=True,
            )
        return None
