# pxrad/materials/material.py
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pxrad.lattice.lattice import Lattice
from pxrad.materials.rules import ExtinctionRule, AllowAll
from pxrad.utils.types import FloatArray, IntArray, BoolArray


@dataclass(frozen=True, slots=True)
class Material:
    """
    Crystal material description (minimal version).

    This class intentionally focuses on what is needed for Laue forward
    simulations (spot positions), i.e. lattice geometry + extinction rules.

    Parameters
    ----------
    name:
        Human-readable material name (e.g. "Si", "Al", "4H-SiC").
    lattice:
        Unit-cell lattice parameters (Å, degrees) and reciprocal basis B (Å^-1).
    rule:
        Extinction rule describing systematic absences (centering / basis).
        Default: AllowAll() (primitive lattice, no systematic absences).

    Notes
    -----
    - This does NOT yet include atomic positions / structure factors.
      Those will live here later (for intensities).
    """
    name: str
    lattice: Lattice
    rule: ExtinctionRule = AllowAll()

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("Material.name must be a non-empty string.")

    @property
    def B(self) -> FloatArray:
        """Reciprocal basis matrix (3x3) without 2π, in Å^-1."""
        return self.lattice.B

    @property
    def A(self) -> FloatArray:
        """Direct basis matrix (3x3) in Å."""
        return self.lattice.A

    @property
    def volume_A3(self) -> float:
        """Unit-cell volume in Å^3."""
        return self.lattice.volume_A3

    def G(self, hkl: IntArray) -> FloatArray:
        """Reciprocal lattice vectors G(hkl) in Å^-1."""
        return self.lattice.G(hkl)

    def d_spacing(self, hkl: IntArray, *, eps: float = 1e-15) -> FloatArray:
        """Interplanar spacing d(hkl) in Å."""
        return self.lattice.d_spacing(hkl, eps=eps)

    def is_allowed(self, hkl: IntArray) -> BoolArray:
        """
        Return a boolean mask for which reflections are allowed by the
        extinction rule.

        Parameters
        ----------
        hkl : (..., 3) integer array

        Returns
        -------
        mask : (...,) bool array
        """
        hkl = np.asarray(hkl)
        if hkl.shape[-1] != 3:
            raise ValueError(f"Expected hkl with last dimension 3, got shape {hkl.shape}.")
        h = hkl[..., 0]
        k = hkl[..., 1]
        l = hkl[..., 2]
        return np.asarray(self.rule(h, k, l), dtype=bool)
