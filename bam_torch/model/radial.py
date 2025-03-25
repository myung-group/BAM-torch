###########################################################################################
# Radial basis and cutoff
# Authors: Ilyes Batatia, Gregor Simm
# This program is distributed under the MIT License (see MIT.md)
###########################################################################################

import ase
import torch
import numpy as np
from e3nn.util.jit import compile_mode

import logging

from bam_torch.utils.scatter import scatter_sum


@compile_mode("script")
class BesselBasis(torch.nn.Module):
    """
    Equation (7)
    """

    def __init__(self, r_max: float, num_basis=8, trainable=False):
        super().__init__()

        bessel_weights = (
            np.pi
            / r_max
            * torch.linspace(
                start=1.0,
                end=num_basis,
                steps=num_basis,
                dtype=torch.get_default_dtype(),
            )
        )
        if trainable:
            self.bessel_weights = torch.nn.Parameter(bessel_weights)
        else:
            self.register_buffer("bessel_weights", bessel_weights)

        self.register_buffer(
            "r_max", torch.tensor(r_max, dtype=torch.get_default_dtype())
        )
        self.register_buffer(
            "prefactor",
            torch.tensor(np.sqrt(2.0 / r_max), dtype=torch.get_default_dtype()),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # [..., 1]
        numerator = torch.sin(self.bessel_weights * x)  # [..., num_basis]
        return self.prefactor * (numerator / x)

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(r_max={self.r_max}, num_basis={len(self.bessel_weights)}, "
            f"trainable={self.bessel_weights.requires_grad})"
        )


@compile_mode("script")
class ChebychevBasis(torch.nn.Module):
    """
    Equation (7)
    """

    def __init__(self, r_max: float, num_basis=8):
        super().__init__()
        self.register_buffer(
            "n",
            torch.arange(1, num_basis + 1, dtype=torch.get_default_dtype()).unsqueeze(
                0
            ),
        )
        self.num_basis = num_basis
        self.r_max = r_max

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # [..., 1]
        x = x.repeat(1, self.num_basis)
        n = self.n.repeat(len(x), 1)
        return torch.special.chebyshev_polynomial_t(x, n)

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(r_max={self.r_max}, num_basis={self.num_basis},"
        )


@compile_mode("script")
class GaussianBasis(torch.nn.Module):
    """
    Gaussian basis functions
    """

    def __init__(self, r_max: float, num_basis=128, trainable=False):
        super().__init__()
        gaussian_weights = torch.linspace(
            start=0.0, end=r_max, steps=num_basis, dtype=torch.get_default_dtype()
        )
        if trainable:
            self.gaussian_weights = torch.nn.Parameter(
                gaussian_weights, requires_grad=True
            )
        else:
            self.register_buffer("gaussian_weights", gaussian_weights)
        self.coeff = -0.5 / (r_max / (num_basis - 1)) ** 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # [..., 1]
        x = x - self.gaussian_weights
        return torch.exp(self.coeff * torch.pow(x, 2))


@compile_mode("script")
class PolynomialCutoff(torch.nn.Module):
    """Polynomial cutoff function that goes from 1 to 0 as x goes from 0 to r_max.
    Equation (8) -- TODO: from where?
    """

    p: torch.Tensor
    r_max: torch.Tensor

    def __init__(self, r_max: float, p=6):
        super().__init__()
        self.register_buffer("p", torch.tensor(p, dtype=torch.int))
        self.register_buffer(
            "r_max", torch.tensor(r_max, dtype=torch.get_default_dtype())
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.calculate_envelope(x, self.r_max, self.p.to(torch.int))

    @staticmethod
    def calculate_envelope(
        x: torch.Tensor, r_max: torch.Tensor, p: torch.Tensor
    ) -> torch.Tensor:
        r_over_r_max = x / r_max
        envelope = (
            1.0
            - ((p + 1.0) * (p + 2.0) / 2.0) * torch.pow(r_over_r_max, p)
            + p * (p + 2.0) * torch.pow(r_over_r_max, p + 1)
            - (p * (p + 1.0) / 2) * torch.pow(r_over_r_max, p + 2)
        )
        return envelope * (x < r_max)

    def __repr__(self):
        return f"{self.__class__.__name__}(p={self.p}, r_max={self.r_max})"


@compile_mode("script")
class ZBLBasis(torch.nn.Module):
    """Implementation of the Ziegler-Biersack-Littmark (ZBL) potential
    with a polynomial cutoff envelope.
    """

    p: torch.Tensor

    def __init__(self, p=6, trainable=False, **kwargs):
        super().__init__()
        if "r_max" in kwargs:
            logging.warning(
                "r_max is deprecated. r_max is determined from the covalent radii."
            )

        # Pre-calculate the p coefficients for the ZBL potential
        self.register_buffer(
            "c",
            torch.tensor(
                [0.1818, 0.5099, 0.2802, 0.02817], dtype=torch.get_default_dtype()
            ),
        )
        self.register_buffer("p", torch.tensor(p, dtype=torch.int))
        self.register_buffer(
            "covalent_radii",
            torch.tensor(
                ase.data.covalent_radii,
                dtype=torch.get_default_dtype(),
            ),
        )
        if trainable:
            self.a_exp = torch.nn.Parameter(torch.tensor(0.300, requires_grad=True))
            self.a_prefactor = torch.nn.Parameter(
                torch.tensor(0.4543, requires_grad=True)
            )
        else:
            self.register_buffer("a_exp", torch.tensor(0.300))
            self.register_buffer("a_prefactor", torch.tensor(0.4543))

    def forward(
        self,
        x: torch.Tensor,
        node_attrs: torch.Tensor,
        edge_index: torch.Tensor,
        atomic_numbers: torch.Tensor,
    ) -> torch.Tensor:
        sender = edge_index[0]
        receiver = edge_index[1]
        node_atomic_numbers = atomic_numbers[torch.argmax(node_attrs, dim=1)].unsqueeze(
            -1
        )
        Z_u = node_atomic_numbers[sender]
        Z_v = node_atomic_numbers[receiver]
        a = (
            self.a_prefactor
            * 0.529
            / (torch.pow(Z_u, self.a_exp) + torch.pow(Z_v, self.a_exp))
        )
        r_over_a = x / a
        phi = (
            self.c[0] * torch.exp(-3.2 * r_over_a)
            + self.c[1] * torch.exp(-0.9423 * r_over_a)
            + self.c[2] * torch.exp(-0.4028 * r_over_a)
            + self.c[3] * torch.exp(-0.2016 * r_over_a)
        )
        v_edges = (14.3996 * Z_u * Z_v) / x * phi
        r_max = self.covalent_radii[Z_u] + self.covalent_radii[Z_v]
        envelope = PolynomialCutoff.calculate_envelope(x, r_max, self.p)
        v_edges = 0.5 * v_edges * envelope
        V_ZBL = scatter_sum(v_edges, receiver, dim=0, dim_size=node_attrs.size(0))
        return V_ZBL.squeeze(-1)

    def __repr__(self):
        return f"{self.__class__.__name__}(c={self.c})"


@compile_mode("script")
class AgnesiTransform(torch.nn.Module):
    """Agnesi transform - see section on Radial transformations in
    ACEpotentials.jl, JCP 2023 (https://doi.org/10.1063/5.0158783).
    """

    def __init__(
        self,
        q: float = 0.9183,
        p: float = 4.5791,
        a: float = 1.0805,
        trainable=False,
    ):
        super().__init__()
        self.register_buffer("q", torch.tensor(q, dtype=torch.get_default_dtype()))
        self.register_buffer("p", torch.tensor(p, dtype=torch.get_default_dtype()))
        self.register_buffer("a", torch.tensor(a, dtype=torch.get_default_dtype()))
        self.register_buffer(
            "covalent_radii",
            torch.tensor(
                ase.data.covalent_radii,
                dtype=torch.get_default_dtype(),
            ),
        )
        if trainable:
            self.a = torch.nn.Parameter(torch.tensor(1.0805, requires_grad=True))
            self.q = torch.nn.Parameter(torch.tensor(0.9183, requires_grad=True))
            self.p = torch.nn.Parameter(torch.tensor(4.5791, requires_grad=True))

    def forward(
        self,
        x: torch.Tensor,
        node_attrs: torch.Tensor,
        edge_index: torch.Tensor,
        atomic_numbers: torch.Tensor,
    ) -> torch.Tensor:
        sender = edge_index[0]
        receiver = edge_index[1]
        node_atomic_numbers = atomic_numbers[torch.argmax(node_attrs, dim=1)].unsqueeze(
            -1
        )
        Z_u = node_atomic_numbers[sender]
        Z_v = node_atomic_numbers[receiver]
        r_0: torch.Tensor = 0.5 * (self.covalent_radii[Z_u] + self.covalent_radii[Z_v])
        r_over_r_0 = x / r_0
        return (
            1
            + (
                self.a
                * torch.pow(r_over_r_0, self.q)
                / (1 + torch.pow(r_over_r_0, self.q - self.p))
            )
        ).reciprocal_()

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(a={self.a:.4f}, q={self.q:.4f}, p={self.p:.4f})"
        )


@compile_mode("script")
class SoftTransform(torch.nn.Module):
    """
    Tanh-based smooth transformation:
        T(x) = p1 + (x - p1)*0.5*[1 + tanh(alpha*(x - m))],
    which smoothly transitions from ~p1 for x << p1 to ~x for x >> r0.
    """

    def __init__(self, alpha: float = 4.0, trainable=False):
        """
        Args:
            p1 (float): Lower "clamp" point.
            alpha (float): Steepness; if None, defaults to ~6/(r0-p1).
            trainable (bool): Whether to make parameters trainable.
        """
        super().__init__()
        # Initialize parameters
        self.register_buffer(
            "alpha", torch.tensor(alpha, dtype=torch.get_default_dtype())
        )
        if trainable:
            self.alpha = torch.nn.Parameter(self.alpha.clone())
        self.register_buffer(
            "covalent_radii",
            torch.tensor(
                ase.data.covalent_radii,
                dtype=torch.get_default_dtype(),
            ),
        )

    def compute_r_0(
        self,
        node_attrs: torch.Tensor,
        edge_index: torch.Tensor,
        atomic_numbers: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute r_0 based on atomic information.

        Args:
            node_attrs (torch.Tensor): Node attributes (one-hot encoding of atomic numbers).
            edge_index (torch.Tensor): Edge index indicating connections.
            atomic_numbers (torch.Tensor): Atomic numbers.

        Returns:
            torch.Tensor: r_0 values for each edge.
        """
        sender = edge_index[0]
        receiver = edge_index[1]
        node_atomic_numbers = atomic_numbers[torch.argmax(node_attrs, dim=1)].unsqueeze(
            -1
        )
        Z_u = node_atomic_numbers[sender]
        Z_v = node_atomic_numbers[receiver]
        r_0: torch.Tensor = self.covalent_radii[Z_u] + self.covalent_radii[Z_v]
        return r_0

    def forward(
        self,
        x: torch.Tensor,
        node_attrs: torch.Tensor,
        edge_index: torch.Tensor,
        atomic_numbers: torch.Tensor,
    ) -> torch.Tensor:

        r_0 = self.compute_r_0(node_attrs, edge_index, atomic_numbers)
        p_0 = (3 / 4) * r_0
        p_1 = (4 / 3) * r_0
        m = 0.5 * (p_0 + p_1)
        alpha = self.alpha / (p_1 - p_0)
        s_x = 0.5 * (1.0 + torch.tanh(alpha * (x - m)))
        return p_0 + (x - p_0) * s_x

    def __repr__(self):
        return f"{self.__class__.__name__}(alpha={self.alpha.item():.4f})"


# ============= Custom Radial ===============
import torch
from functools import lru_cache
from math import factorial
from typing import Callable


def sus(x: torch.Tensor) -> torch.Tensor:
    r"""Smooth Unit Step function.

    ``-inf->0, 0->0, 2->0.6, +inf->1``

    .. math::
        \text{sus}(x) = \begin{cases}
            0, & \text{if } x < 0 \\
            exp(-1/x), & \text{if } x \geq 0 \\
        \end{cases}
    """
    return torch.where(x > 0.0, 
                       torch.exp(-1.0 / torch.where(x > 0.0, x, torch.tensor(1.0, device=x.device))), 
                       torch.tensor(0.0, device=x.device))


def soft_envelope(
    x: torch.Tensor,
    x_max: float = 1.0,
    arg_multiplicator: float = 2.0,
    value_at_origin: float = 1.2,
) -> torch.Tensor:
    r"""Smooth envelope function.

    Args:
        x (torch.Tensor): input of shape ``[...]``
        x_max (float): cutoff value

    Returns:
        torch.Tensor: smooth (:math:`C^\infty`) envelope function of shape ``[...]``
    """
    cste = value_at_origin / sus(torch.tensor(arg_multiplicator, device=x.device))
    return cste * sus(arg_multiplicator * (1.0 - x / x_max))



def u(p: int, x: torch.Tensor) -> torch.Tensor:
    r"""Equivalent to :func:`poly_envelope` with ``n0 = p-1`` and ``n1 = 2``."""
    return (
        1
        - (p + 1) * (p + 2) / 2 * x**p
        + p * (p + 2) * x ** (p + 1)
        - p * (p + 1) / 2 * x ** (p + 2)
    )


def _constraint(x: float, derivative: int, degree: int):
    return [
        (
            0
            if derivative > N
            else factorial(N) // factorial(N - derivative) * x ** (N - derivative)
        )
        for N in range(degree)
    ]


#@lru_cache(maxsize=None)
def solve_polynomial_torch(constraints) -> Callable[[torch.Tensor], torch.Tensor]:
    degree = len(constraints)

    A = torch.tensor(
        [
            _constraint(x, derivative, degree)
            for x, derivative, _ in sorted(constraints)
        ],
        dtype=torch.float32,
    )

    B = torch.tensor([y for _, _, y in sorted(constraints)], dtype=torch.float32)

    if A.shape[0] != A.shape[1]:
        raise ValueError(f"Matrix A must be square, but got shape {A.shape}")
    if B.shape[0] != A.shape[0]:
        raise ValueError(f"Vector B must match A's size, but got {B.shape}")

    c = torch.linalg.solve(A, B)
    c = torch.flip(c, dims=[0])  

    def poly_fn(x: torch.Tensor) -> torch.Tensor:
        powers = torch.arange(len(c), dtype=x.dtype, device=x.device).flip(0)  
        poly_values = sum(c[i] * x ** powers[i] for i in range(len(c))) 
        return poly_values

    return poly_fn



def poly_envelope_torch(n0: int, n1: int, x_max: float = 1.0) -> Callable[[torch.Tensor], torch.Tensor]:
    r"""Polynomial envelope function with ``n0`` and ``n1`` derivatives equal to 0 at ``x=0`` and ``x=1`` respectively.

    Args:
        n0 (int): number of derivatives equal to 0 at ``x=0``
        n1 (int): number of derivatives equal to 0 at ``x=1``
        x_max (float): maximum value of the input, instead of 1

    Returns:
        callable: polynomial envelope function
    """
    poly = solve_polynomial_torch(
        frozenset(
            {(-0.5, 0, 1.0), (0.5, 0, 0.0)}
            | {(-0.5, derivative, 0.0) for derivative in range(1, n0 + 1)}
            | {(0.5, derivative, 0.0) for derivative in range(1, n1 + 1)}
        )
    )

    def f(x: torch.Tensor) -> torch.Tensor:
        x_small = torch.where(x < x_max, x, torch.tensor(x_max, dtype=x.dtype, device=x.device))
        return torch.where(
            x < x_max,
            poly(x_small / x_max - 0.5),
            torch.tensor(0.0, dtype=x.dtype, device=x.device),
        )

    return f

@compile_mode("script")
class BesselFunction(torch.nn.Module):
    def __init__(self, n: int, x_max: float = 1.0):
        """
        Bessel basis functions.

        Args:
            n (int): Number of basis functions.
            x_max (float): Maximum value of the input.
        """
        super().__init__()
        assert isinstance(n, int), "n must be an integer."
        self.n = n
        self.register_buffer("x_max", torch.tensor(x_max, dtype=torch.float32))
        self.register_buffer("factor", torch.sqrt(torch.tensor(2.0 / x_max, dtype=torch.float32)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the Bessel function for input `x`.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Basis function values.
        """
        x = x[..., None]  # Expand dimension
        n_values = torch.arange(1, self.n + 1, dtype=x.dtype, device=x.device)
        x_nonzero = torch.where(x == 0.0, torch.tensor(1.0, device=x.device), x)

        return self.factor * torch.where(
            x == 0,
            n_values * torch.pi / self.x_max,
            torch.sin(n_values * torch.pi / self.x_max * x_nonzero) / x_nonzero,
        )

@compile_mode("script")
class PolyEnvelope(torch.nn.Module):
    def __init__(self, n0: int, n1: int, x_max: float = 1.0):
        """
        Polynomial envelope function with `n0` and `n1` derivatives equal to 0 at `x=0` and `x=1`.

        Args:
            n0 (int): Number of derivatives equal to 0 at `x=0`.
            n1 (int): Number of derivatives equal to 0 at `x=1`.
            x_max (float): Maximum value of the input, instead of 1.
        """
        super().__init__()
        self.n0 = n0
        self.n1 = n1
        self.register_buffer("x_max", torch.tensor(x_max, dtype=torch.float32))
        self.poly = self._solve_polynomial()

    def _solve_polynomial(self) -> Callable[[torch.Tensor], torch.Tensor]:
        """
        Precompute the polynomial function using `solve_polynomial_torch`.
        """
        constraints = frozenset(
            {(-0.5, 0, 1.0), (0.5, 0, 0.0)}
            | {(-0.5, derivative, 0.0) for derivative in range(1, self.n0 + 1)}
            | {(0.5, derivative, 0.0) for derivative in range(1, self.n1 + 1)}
        )
        return solve_polynomial_torch(constraints)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the polynomial envelope for input `x`.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Transformed output.
        """
        x_small = torch.where(x < self.x_max, x, torch.tensor(self.x_max, dtype=x.dtype, device=x.device))
        return torch.where(
            x < self.x_max,
            self.poly(x_small / self.x_max - 0.5),
            torch.tensor(0.0, dtype=x.dtype, device=x.device),
        )[:, None]

