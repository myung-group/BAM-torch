"""
ZBL (Ziegler-Biersack-Littmark) repulsive pair potential.

Short-range nuclear repulsion prior for ML potentials.
Prevents unphysical atom overlap during MD simulations when the ML model
was trained only on equilibrium structures.

Usage:
    energy_zbl, forces_zbl = compute_zbl_correction(
        positions, atomic_numbers, r_inner=0.5, r_outer=1.0
    )

Reference:
    Ziegler, Biersack, Littmark, "The Stopping and Range of Ions in Solids"
    (Pergamon Press, 1985)
"""

import numpy as np

# ZBL universal screening function: φ(x) = Σ c_i exp(-d_i x)
_ZBL_C = np.array([0.1818, 0.5099, 0.2802, 0.02817])
_ZBL_D = np.array([3.2, 0.9423, 0.4029, 0.2016])

# Physical constants
_KE2 = 14.3996  # e² / (4πε₀) in eV·Å (Coulomb constant)
_A_BOHR = 0.52918  # Bohr radius in Å


def compute_zbl_correction(positions, atomic_numbers, r_inner=0.5, r_outer=1.0):
    """Compute ZBL repulsive energy and forces for all pairs.

    Adds a smooth repulsive wall below r_outer that prevents atom overlap.
    Uses a quintic switching function (C² continuous) to blend:
      - r < r_inner : full ZBL repulsion
      - r_inner < r < r_outer : smooth transition
      - r > r_outer : no ZBL (pure ML potential)

    The correction is additive: E_total = E_ML + E_ZBL.
    At equilibrium bond lengths (≥ r_outer), E_ZBL = 0.

    Args:
        positions: (N, 3) atomic positions in Å.
        atomic_numbers: (N,) integer atomic numbers.
        r_inner: Full ZBL below this distance (Å).
        r_outer: No ZBL above this distance (Å).
                 Should be ≤ shortest bond length in training data.

    Returns:
        energy: Total ZBL energy (eV), scalar.
        forces: (N, 3) ZBL forces (eV/Å).
    """
    n_atoms = len(positions)
    energy = 0.0
    forces = np.zeros((n_atoms, 3), dtype=np.float64)
    dr_inv = 1.0 / (r_outer - r_inner)

    for i in range(n_atoms):
        zi = atomic_numbers[i]
        for j in range(i + 1, n_atoms):
            rij = positions[j] - positions[i]
            r = np.linalg.norm(rij)

            if r >= r_outer or r < 1e-10:
                continue

            zj = atomic_numbers[j]
            r_hat = rij / r

            # Screening length
            a = 0.8854 * _A_BOHR / (zi ** 0.23 + zj ** 0.23)
            x = r / a  # reduced distance

            # ZBL screening function and its derivative
            exp_terms = np.exp(-_ZBL_D * x)
            phi = np.dot(_ZBL_C, exp_terms)
            dphi_dx = np.dot(-_ZBL_D * _ZBL_C, exp_terms)

            # ZBL pair energy: V = ke² Z₁Z₂/r · φ(r/a)
            zz = _KE2 * zi * zj
            v_zbl = zz / r * phi

            # dV/dr = ke² Z₁Z₂ [-φ/r² + φ'/(r·a)]
            dv_dr = zz * (-phi / (r * r) + dphi_dx / (r * a))

            # Switching function: quintic, C² continuous
            t = (r - r_inner) * dr_inv
            if t <= 0.0:
                sw = 1.0
                dsw_dr = 0.0
            else:
                # sw = 1 - t³(10 - 15t + 6t²)
                t2 = t * t
                t3 = t2 * t
                sw = 1.0 - t3 * (10.0 - 15.0 * t + 6.0 * t2)
                # dsw/dt = -30 t²(1-t)²
                dsw_dr = -30.0 * t2 * (1.0 - t) ** 2 * dr_inv

            # Corrected energy
            energy += v_zbl * sw

            # Force on atom j along r_hat:
            # F_j = -(dV/dr · sw + V · dsw/dr) · r_hat
            f_mag = -(dv_dr * sw + v_zbl * dsw_dr)
            forces[j] += f_mag * r_hat
            forces[i] -= f_mag * r_hat

    return energy, forces
