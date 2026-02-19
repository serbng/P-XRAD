from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np

from pxrad.utils.types import FloatArray, IntArray
from pxrad.utils.units import deg2rad

@dataclass(frozen=True, slots=True)
class Lattice:
    """
    Unit-cell lattice parameters.

    Conventions / Units
    -------------------
    - a, b, c are in Å (Angstrom)
    - alpha, beta, gamma are in degrees
    - Direct basis matrix A has columns [a_vec, b_vec, c_vec] in Å
    - Reciprocal basis matrix B = inv(A).T (NO 2π) in Å^-1
    - For Miller indices hkl (integers), reciprocal vector is:
        G = B @ hkl   in Å^-1
      and d-spacing is:
        d = 1 / ||G|| in Å

    Notes
    -----
    This definition of B (without 2π) is the most convenient for Bragg/Laue
    relations expressed with wavelength in Å.
    """

    a: float
    b: float
    c: float
    alpha: float  # degrees
    beta: float   # degrees
    gamma: float  # degrees
    
    # Will be computed in __post_init__
    _A: FloatArray = field(init=False, repr=False)
    _B: FloatArray = field(init=False, repr=False)
    _volume_A3: float = field(init=False, repr=False)

    def __post_init__(self) -> None:
        a, b, c = float(self.a), float(self.b), float(self.c)
        if not (np.isfinite(a) and np.isfinite(b) and np.isfinite(c)):
            raise ValueError("Lattice lengths a,b,c must be finite.")
        if a <= 0 or b <= 0 or c <= 0:
            raise ValueError("Lattice lengths a,b,c must be > 0 Å.")

        for name, ang in [("alpha", self.alpha), ("beta", self.beta), ("gamma", self.gamma)]:
            ang = float(ang)
            if not np.isfinite(ang):
                raise ValueError(f"Lattice angle {name} must be finite.")
            # exclude degenerate/invalid cells
            if ang <= 0.0 or ang >= 180.0:
                raise ValueError(f"Lattice angle {name} must be in (0, 180) degrees.")

        # cache matrices (since frozen dataclass)
        A = self._compute_A()
        B = np.linalg.inv(A).T

        # sanity: volume should be positive and not tiny
        vol = float(abs(np.linalg.det(A)))
        if not np.isfinite(vol) or vol <= 0:
            raise ValueError("Invalid lattice cell: zero/negative volume.")
        if vol < 1e-12:
            raise ValueError("Invalid lattice cell: extremely small volume (check units/angles).")

        object.__setattr__(self, "_A", A)
        object.__setattr__(self, "_B", B)
        object.__setattr__(self, "_volume_A3", vol)
        
    def _compute_A(self) -> FloatArray:
        """
        Build direct basis matrix A from cell parameters.
        Columns are a_vec, b_vec, c_vec in an orthonormal crystal frame.
        """
        a, b, c = float(self.a), float(self.b), float(self.c)
        alpha = deg2rad(float(self.alpha))
        beta  = deg2rad(float(self.beta))
        gamma = deg2rad(float(self.gamma))

        cosa, cosb, cosg = np.cos(alpha), np.cos(beta), np.cos(gamma)
        sing = np.sin(gamma)

        # a along x
        a_vec = np.array([a, 0.0, 0.0], dtype=float)
        # b in xy-plane
        b_vec = np.array([b * cosg, b * sing, 0.0], dtype=float)

        # c has components:
        # cx = c*cos(beta)
        # cy = c*(cos(alpha) - cos(beta)*cos(gamma)) / sin(gamma)
        # cz from normalization to match cos(alpha) with b, etc.
        if abs(sing) < 1e-15:
            raise ValueError("Invalid lattice: gamma too close to 0 or 180 degrees.")

        cx = c * cosb
        cy = c * (cosa - cosb * cosg) / sing

        # cz^2 = c^2 - cx^2 - cy^2 (should be positive)
        cz2 = c * c - cx * cx - cy * cy
        if cz2 <= 0.0:
            # numerical slack: allow tiny negative due to rounding
            if cz2 > -1e-12:
                cz2 = 0.0
            else:
                raise ValueError("Invalid lattice: angles/lengths give non-real c_z component.")
        cz = np.sqrt(cz2)

        c_vec = np.array([cx, cy, cz], dtype=float)

        # A columns are the basis vectors
        A = np.column_stack([a_vec, b_vec, c_vec]).astype(float)
        return A
    
    @property
    def A(self) -> FloatArray:
        """Direct basis matrix A (3x3), columns are a,b,c vectors in Å."""
        return self._A.copy()
    
    @property
    def B(self) -> FloatArray:
        """Reciprocal basis matrix B (3x3) without 2π, in Å^-1."""
        return self._B.copy()

    @property
    def volume_A3(self) -> float:
        """Unit cell volume in Å^3."""
        return float(self._volume_A3)
    
    def G(self, hkl: IntArray) -> FloatArray:
        """
        Reciprocal lattice vectors G for given hkl.

        Parameters
        ----------
        hkl : (..., 3) integer array

        Returns
        -------
        G : (..., 3) float array in Å^-1
        """
        hkl = np.asarray(hkl)
        if hkl.shape[-1] != 3:
            raise ValueError(f"Expected hkl with last dimension 3, got shape {hkl.shape}")
        # promote to float for matmul
        hklt = hkl.astype(float)
        # (...,3) = (...,3) @ (3,3)^T  -> use einsum for clarity
        # G_i = sum_j hklt_j * B_{ij}
        G = np.einsum("...j,ij->...i", hklt, self._B)
        return G
    
    def d_spacing(self, hkl: IntArray, *, eps: float = 1e-15) -> FloatArray:
        """
        d-spacing for given hkl.

        d = 1 / ||G||, where G = B @ hkl (Å^-1)

        Parameters
        ----------
        hkl : (..., 3) integer array
        eps : small number to avoid division by zero

        Returns
        -------
        d : (...,) float array in Å
        """
        G = self.G(hkl)
        g2 = np.einsum("...i,...i->...", G, G)
        # hkl = (0,0,0) => g2 = 0 => d infinite; we return inf
        d = np.where(g2 > eps, 1.0 / np.sqrt(g2), np.inf)
        return d
