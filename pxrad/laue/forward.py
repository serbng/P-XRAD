from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from pxrad.materials.material import Material
from pxrad.beam.beam import Beam
from pxrad.detectors.detector import Detector
from pxrad.geometry.frames import Geometry
from pxrad.geometry.pose import DetectorPose
from pxrad.geometry.scattering import scattering_angles
from pxrad.geometry.projection import ray_to_pixel

from pxrad.laue.hkl import generate_hkls

from pxrad.utils.linalg import unit, vec3
from pxrad.utils.types import IntArray, FloatArray, BoolArray, Mat33
from pxrad.utils.units import HC_KEV_A


def _unit_rows(v: np.ndarray, *, eps: float = 1e-15) -> np.ndarray:
    """
    Normalize row-vectors to unit length.
    Rows with non-finite or too-small norm become NaN.
    """
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v, axis=-1)
    out = np.empty_like(v, dtype=float)
    good = np.isfinite(n) & (n > eps)
    out[~good] = np.nan
    out[good] = v[good] / n[good, None]
    return out


@dataclass(frozen=True, slots=True)
class LaueForwardResult:
    """
    Output of a Laue forward simulation (spot positions + kinematics).

    Conventions
    -----------
    - R: crystal->lab rotation matrix
    - hkl: (N,3) Miller indices
    - G_lab: (N,3) reciprocal vectors in lab frame, Å^-1
    - lambda_A: (N,) wavelength in Å
    - E_keV: (N,) photon energy in keV
    - uf: (N,3) outgoing ray directions (unit) in lab frame
    - angles: (N,3) (two_theta, chi, two_theta_scattering) from pxrad.geometry.scattering
    - uv: (N,2) detector pixel coords; NaN if not on detector / invalid
    - on_detector: (N,) True where uv is finite (both coords)
    """
    R: Mat33
    hkl: IntArray
    G_lab: FloatArray
    lambda_A: FloatArray
    E_keV: FloatArray
    uf: FloatArray
    angles: FloatArray
    uv: FloatArray
    on_detector: BoolArray
    
def _empty_laue_result(rotation_matrix):
    return LaueForwardResult(
            R = rotation_matrix,
            hkl         = np.empty((0, 3), dtype=int),
            G_lab       = np.empty((0, 3), dtype=float),
            lambda_A    = np.empty((0,), dtype=float),
            E_keV       = np.empty((0,), dtype=float),
            uf          = np.empty((0, 3), dtype=float),
            angles      = np.empty((0, 3), dtype=float),
            uv          = np.empty((0, 2), dtype=float),
            on_detector = np.empty((0,), dtype=bool),
        )
    
def forward_laue(
    material: Material,
    R: Mat33,
    geometry: Geometry,
    pose: DetectorPose,
    beam: Beam,
    detector: Detector,
    hkls: IntArray,
    *,
    filter_extinctions: bool = True,
    filter_energy_range: bool = True,
    filter_on_detector: bool = True,
    compute_angles: bool = True,
    eps: float = 1e-15,
) -> LaueForwardResult:
    """
    Forward-simulate Laue spot positions for a fixed crystal orientation.

    Physics model (NO 2π, consistent with pxrad.lattice):
    - G_crys = B @ hkl            (Å^-1) in crystal frame
    - G_lab  = R @ G_crys         (Å^-1)
    - Laue + elastic conditions give a unique wavelength:
          lambda = -2 (u·G_lab) / ||G_lab||^2
      where u = geometry.beam_dir is unit vector (source -> sample).
      Physical solutions require lambda > 0  (equivalently u·G_lab < 0).
    - Energy: E_keV = HC_KEV_A / lambda_A
    - Outgoing ray direction (unit):
          uf ~ u + lambda * G_lab
      (renormalized for numerical safety)
    - Projection: uv = ray_to_pixel(uf, pose, detector)  (NaNs if outside)

    Filtering
    ---------
    Optional filters (enabled by default where sensible):
    - extinctions: material.is_allowed(hkl)
    - physical lambda: finite and > 0 (always applied)
    - beam energy range: Emin <= E <= Emax
    - on-detector: finite uv

    Returns
    -------
    LaueForwardResult
        If filter_on_detector=True, arrays are already reduced to only valid spots.
        Otherwise, uv may contain NaNs and on_detector marks which ones are valid.
    """
    hkls = np.asarray(hkls)
    if hkls.ndim != 2 or hkls.shape[1] != 3:
        raise ValueError(f"hkls must have shape (N,3), got {hkls.shape}.")

    R = np.asarray(R, dtype=float)
    if R.shape != (3, 3):
        raise ValueError(f"R must have shape (3,3), got {R.shape}.")

    # Early return for empty input
    if hkls.size == 0:
        return _empty_laue_result(rotation_matrix=R)

    # Drop the (0,0,0) reflection if present (it’s unphysical)
    nonzero = np.any(hkls != 0, axis=1)
    hkls = hkls[nonzero]

    if hkls.size == 0:
        return _empty_laue_result(rotation_matrix=R)

    # Optional extinction filtering (systematic absences)
    if filter_extinctions:
        allowed = material.is_allowed(hkls)
        hkls = hkls[allowed]

    if hkls.size == 0:
        return _empty_laue_result(rotation_matrix=R)

    # Incident beam direction in lab frame (already unit by construction of Geometry)
    ui = np.asarray(geometry.beam_dir, dtype=float)
    # (Still normalize for paranoia / numerical drift)
    ui = unit(vec3(ui))

    # Reciprocal vectors in lab
    G_crys = material.G(hkls)              # (N,3) Å^-1
    G_lab = (R @ G_crys.T).T               # (N,3) Å^-1

    # Solve lambda (Å): lambda = -2 (u·G) / ||G||^2
    ug = G_lab @ ui # (N,)
    g2 = np.einsum("ni,ni->n", G_lab, G_lab)

    denom = np.where(g2 > eps, g2, np.nan)
    lambda_A = -2.0 * ug / denom

    # Physical solutions: finite and > 0
    physical = np.isfinite(lambda_A) & (lambda_A > 0.0)

    # Energy
    E_keV = HC_KEV_A / lambda_A

    to_keep = physical
    if filter_energy_range:
        Emin, Emax = beam.energy_range_keV
        to_keep = to_keep & (E_keV >= Emin) & (E_keV <= Emax)

    if not np.any(to_keep):
        return _empty_laue_result(rotation_matrix=R)

    hkls = hkls[to_keep].astype(int, copy=False)
    G_lab = G_lab[to_keep]
    lambda_A = lambda_A[to_keep]
    E_keV = E_keV[to_keep]

    # Outgoing directions uf = u + lambda * G (renormalize)
    uf = ui[None, :] + lambda_A[:, None] * G_lab
    uf = _unit_rows(uf, eps=eps)

    # Project to detector (NaNs for outside)
    uv = ray_to_pixel(uf, pose=pose, detector=detector)

    on_det = np.isfinite(uv).all(axis=1)

    # Angles (optional)
    if compute_angles:
        two_theta, chi, two_theta_scatt = scattering_angles(
            uf, beam_dir=ui, degrees=True, eps=eps
        )
        angles = np.stack([two_theta, chi, two_theta_scatt], axis=1)
    else:
        angles = np.full((uf.shape[0], 3), np.nan, dtype=float)

    if filter_on_detector:
        m = on_det
        hkls   = hkls[m]
        G_lab  = G_lab[m]
        lambda_A = lambda_A[m]
        E_keV  = E_keV[m]
        uf     = uf[m]
        angles = angles[m]
        uv     = uv[m]
        on_det = on_det[m]

    return LaueForwardResult(
        R   = R,
        hkl = hkls,
        G_lab    = G_lab,
        lambda_A = lambda_A,
        E_keV    = E_keV,
        uf       = uf,
        angles   = angles,
        uv       = uv,
        on_detector = on_det,
    )
    

def simulate_laue(
    material: Material,
    R: Mat33,
    geometry: Geometry,
    pose: DetectorPose,
    beam: Beam,
    detector: Detector,
    *,
    dmin_A: float | None = None,
    nmax: int | None = None,
    include_negatives: bool = True,
    filter_extinctions: bool = True,
    filter_energy_range: bool = True,
    filter_on_detector: bool = True,
    compute_angles: bool = True,
) -> LaueForwardResult:
    
    hkls = generate_hkls(
        material,
        beam=beam,
        dmin_A=dmin_A,
        nmax=nmax,
        include_negatives=include_negatives,
    )
    return forward_laue(
        material=material,
        R=R,
        geometry=geometry,
        pose=pose,
        beam=beam,
        detector=detector,
        hkls=hkls,
        filter_extinctions=filter_extinctions,
        filter_energy_range=filter_energy_range,
        filter_on_detector=filter_on_detector,
        compute_angles=compute_angles,
    )
