from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from pxrad.beam.beam import Beam, WhiteBeam
from pxrad.materials.material import Material
from pxrad.utils.types import IntArray, FloatArray


def _dmin_from_beam(beam: Beam) -> float:
    """
    Compute dmin (Å) from a beam energy range.

    Using Bragg inequality: 2 d sin(theta) = lambda, with sin(theta) <= 1
    => lambda <= 2 d  => d >= lambda/2.

    The smallest wavelength in the beam is lambda_min (from Emax),
    so the minimum possible d-spacing that can diffract is:
        dmin = lambda_min / 2
    """
    lambda_min_A, _lambda_max_A = beam.wavelength_range_A
    return 0.5 * float(lambda_min_A)


def estimate_hkl_radius(material: Material, dmin_A: float) -> int:
    """
    Conservative bound for the maximum |h|,|k|,|l| needed to capture ALL
    reflections with d >= dmin.

    We use the inequality:
        ||G||^2 = hkl^T (B^T B) hkl >= lambda_min(M) * ||hkl||^2
    where M = B^T B is positive definite.

    Condition d >= dmin is equivalent to ||G|| <= 1/dmin = gmax.
    Therefore, if ||hkl|| > gmax / sqrt(lambda_min(M)), it cannot satisfy the cutoff.

    Returns
    -------
    nmax : int
        A safe integer radius such that searching h,k,l in [-nmax, nmax] is sufficient.
    """
    if not np.isfinite(dmin_A) or dmin_A <= 0:
        raise ValueError(f"dmin_A must be finite and > 0, got {dmin_A!r}.")

    gmax = 1.0 / float(dmin_A)  # Å^-1
    B = material.B
    M = B.T @ B  # (3,3) positive definite
    evals = np.linalg.eigvalsh(M)
    lam_min = float(np.min(evals))
    if not np.isfinite(lam_min) or lam_min <= 0:
        raise ValueError("Invalid reciprocal metric: smallest eigenvalue <= 0.")

    nmax = int(np.ceil(gmax / np.sqrt(lam_min)))
    return max(nmax, 0)


def generate_hkls(
    material: Material,
    *,
    beam: Optional[Beam] = WhiteBeam(5.0, 25.0),
    dmin_A: Optional[float] = None,
    nmax: Optional[int] = None,
    include_negatives: bool = True,
    chunk_h: int = 32,
) -> IntArray:
    """
    Generate candidate Miller indices (h,k,l) for Laue simulation.

    The output is a finite list of HKLs that pass:
    - d-spacing cutoff: d(hkl) >= dmin
    - extinction rule: material.is_allowed(hkl)
    - removal of (0,0,0)

    You can specify the cutoff either by:
    - passing `beam` (preferred): dmin = lambda_min/2
    - or directly `dmin_A` in Å
    - or providing `nmax` explicitly (debug / brute-force cap)

    Parameters
    ----------
    material:
        Material providing lattice + extinction rules.
    beam:
        Beam object (keV range). If provided and dmin_A is None, we compute dmin_A from beam.
    dmin_A:
        Minimum d-spacing in Å to include. If None and beam is provided, computed from beam.
    nmax:
        Search radius in index space. If None, estimated conservatively from material + dmin_A.
    include_negatives:
        If True, generate both positive and negative HKLs. (Recommended for Laue.)
        If False, generates only non-negative indices, and removes (0,0,0).
    chunk_h:
        Process h values in chunks to reduce memory usage.

    Returns
    -------
    hkls : (N, 3) int array
        Filtered list of Miller indices.
    """
    # Resolve dmin
    if dmin_A is None:
        if beam is None:
            raise ValueError("Provide either beam=... or dmin_A=... (or nmax=... for manual bounds).")
        dmin_A = _dmin_from_beam(beam)

    dmin_A = float(dmin_A)
    if dmin_A <= 0 or not np.isfinite(dmin_A):
        raise ValueError(f"dmin_A must be finite and > 0, got {dmin_A!r}.")

    # Resolve nmax
    if nmax is None:
        nmax = estimate_hkl_radius(material, dmin_A)
    nmax = int(nmax)
    if nmax < 0:
        raise ValueError("nmax must be >= 0.")

    if not include_negatives:
        h_vals = np.arange(0, nmax + 1, dtype=int)
        k_vals = np.arange(0, nmax + 1, dtype=int)
        l_vals = np.arange(0, nmax + 1, dtype=int)
    else:
        h_vals = np.arange(-nmax, nmax + 1, dtype=int)
        k_vals = np.arange(-nmax, nmax + 1, dtype=int)
        l_vals = np.arange(-nmax, nmax + 1, dtype=int)

    # Prebuild (k,l) mesh once (size ~ (2n+1)^2)
    kk, ll = np.meshgrid(k_vals, l_vals, indexing="ij")
    kk = kk.ravel()
    ll = ll.ravel()

    out_chunks: list[np.ndarray] = []

    # Chunk over h to limit temporary array size
    for i0 in range(0, len(h_vals), max(int(chunk_h), 1)):
        hs = h_vals[i0 : i0 + max(int(chunk_h), 1)]

        # Build all combinations for this h-chunk
        # total = len(hs) * len(kk)
        hh = np.repeat(hs, kk.size)
        krep = np.tile(kk, hs.size)
        lrep = np.tile(ll, hs.size)

        hkls = np.stack([hh, krep, lrep], axis=1).astype(int, copy=False)

        # Remove (0,0,0)
        nonzero = np.any(hkls != 0, axis=1)
        hkls = hkls[nonzero]
        if hkls.size == 0:
            continue

        # d-spacing cutoff
        d = material.d_spacing(hkls)
        keep_d = d >= dmin_A
        hkls = hkls[keep_d]
        if hkls.size == 0:
            continue

        # extinction rule
        keep_rule = material.is_allowed(hkls)
        hkls = hkls[keep_rule]
        if hkls.size == 0:
            continue

        out_chunks.append(hkls)

    if not out_chunks:
        return np.empty((0, 3), dtype=int)

    hkls_all = np.concatenate(out_chunks, axis=0)

    # Optional: unique (helps reduce downstream work)
    # Note: np.unique sorts; that's fine for us.
    hkls_all = np.unique(hkls_all, axis=0)

    return hkls_all
