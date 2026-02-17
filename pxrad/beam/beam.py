from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Tuple

import numpy as np
from numpy.typing import NDArray

from pxrad.utils.units import energy_keV_to_wavelength_A

ArrayF = NDArray[np.floating]

# --- Beam interface ----------------------------------------------------------
class Beam(ABC):
    """
    Beam model interface.

    Conventions
    -----------
    - Energies are in keV.
    - Wavelengths are in Å (Angstrom).
    """
    
    @property
    @abstractmethod
    def energy_range_keV(self) -> Tuple[float, float]:
        """(Emin, Emax) in keV"""
        raise NotImplementedError
    
    @property
    def wavelength_range_A(self) -> Tuple[float, float]:
        """
        (λmin, λmax) in Å corresponding to the energy range.
        """
        emin, emax = self.energy_range_keV
        # Emax -> λmin ; Emin -> λmax
        lambda_min = float(energy_keV_to_wavelength_A(emax))
        lambda_max = float(energy_keV_to_wavelength_A(emin))
        return lambda_min, lambda_max
    
    def spectrum(self, E_keV: ArrayF) -> ArrayF:
        """
        Spectral weight (relative) as a function of energy.
        Default: flat spectrum (white beam).
        """
        E = np.asarray(E_keV, dtype=float)
        w = np.ones_like(E, dtype=float)
        # For safety: outside beam range -> 0 (useful in intensity models)
        emin, emax = self.energy_range_keV
        w[(E < emin) | (E > emax)] = 0.0
        return w
    
# --- Simple white beam -------------------------------------------------------
@dataclass(frozen=True, slots=True)
class WhiteBeam(Beam):
    """
    White (polychromatic) beam defined only by an energy range.
    The default spectrum is flat in energy (weight = 1 inside range).
    """
    emin_keV: float
    emax_keV: float

    def __post_init__(self) -> None:
        emin = float(self.emin_keV)
        emax = float(self.emax_keV)
        if not np.isfinite(emin) or not np.isfinite(emax):
            raise ValueError("WhiteBeam energies must be finite.")
        if emin <= 0 or emax <= 0:
            raise ValueError("WhiteBeam energies must be > 0 keV.")
        if emax <= emin:
            raise ValueError(f"WhiteBeam requires emax_keV > emin_keV (got {emin}, {emax}).")

    @property
    def energy_range_keV(self) -> Tuple[float, float]:
        return float(self.emin_keV), float(self.emax_keV)
    
# --- Tabulated spectrum beam -------------------------------------------------
@dataclass(frozen=True, slots=True)
class TabulatedBeam(Beam):
    """
    Beam defined by a tabulated spectrum S(E).

    Parameters
    ----------
    E_keV : array, shape (N,)
        Monotonically increasing energy grid in keV.
    S : array, shape (N,)
        Non-negative spectral weights (arbitrary units).
        You can normalize externally or via `normalize=True` in from_xy().
    """
    E_keV: ArrayF
    S: ArrayF

    def __post_init__(self) -> None:
        E = np.asarray(self.E_keV, dtype=float)
        S = np.asarray(self.S, dtype=float)

        if E.ndim != 1 or S.ndim != 1 or E.shape[0] != S.shape[0]:
            raise ValueError(f"TabulatedBeam expects 1D arrays of same length, got {E.shape}, {S.shape}.")
        if E.size < 2:
            raise ValueError("TabulatedBeam needs at least 2 points.")
        if not np.all(np.isfinite(E)) or not np.all(np.isfinite(S)):
            raise ValueError("TabulatedBeam arrays must be finite.")
        if np.any(E <= 0):
            raise ValueError("TabulatedBeam energies must be > 0 keV.")
        if np.any(S < 0):
            raise ValueError("TabulatedBeam spectrum S must be >= 0.")
        if np.any(np.diff(E) <= 0):
            raise ValueError("TabulatedBeam requires E_keV to be strictly increasing.")

        # Store back as contiguous float arrays (since frozen dataclass)
        object.__setattr__(self, "E_keV", np.ascontiguousarray(E, dtype=float))
        object.__setattr__(self, "S", np.ascontiguousarray(S, dtype=float))

    @property
    def energy_range_keV(self) -> Tuple[float, float]:
        E = self.E_keV
        return float(E[0]), float(E[-1])

    def spectrum(self, E_keV: ArrayF) -> ArrayF:
        """
        Linear interpolation of S(E). Outside the tabulated range -> 0.
        """
        E = np.asarray(E_keV, dtype=float)
        # np.interp works on 1D x; ravel + reshape keeps shape
        x = E.ravel()
        y = np.interp(x, self.E_keV, self.S, left=0.0, right=0.0)
        return y.reshape(E.shape)

    @classmethod
    def from_xy(
        cls,
        E_keV: ArrayF,
        S: ArrayF,
        *,
        normalize: bool = False,
        eps: float = 0.0,
    ) -> "TabulatedBeam":
        """
        Build a TabulatedBeam from arrays, with optional normalization.

        normalize:
            If True, scales S so that max(S) == 1 (unless all zeros).
        eps:
            Optional floor added to S (useful if you want to avoid exact zeros).
        """
        E = np.asarray(E_keV, dtype=float)
        W = np.asarray(S, dtype=float)

        if eps != 0.0:
            W = W + float(eps)

        if normalize:
            m = float(np.max(W))
            if m > 0:
                W = W / m

        return cls(E, W)