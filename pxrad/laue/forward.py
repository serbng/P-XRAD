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
    - angles: (N,2) (two_theta, chi) from pxrad.geometry.scattering
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
            angles      = np.empty((0, 2), dtype=float),
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

    This function computes, for each candidate reflection (h,k,l), the unique wavelength
    (and energy) that satisfies the Laue condition for the given crystal orientation and
    incident beam direction, then projects the outgoing rays onto the detector.

    Physics model (no 2π convention)
    --------------------------------
    The reciprocal basis follows pxrad.lattice conventions: ``B = inv(A).T`` (NO 2π),
    so reciprocal vectors have units of Å⁻¹.

    For Miller indices ``hkl``:
      - Reciprocal vector in crystal frame:
            G_crys = B @ hkl                      (Å⁻¹)
      - Rotate to lab:
            G_lab = R @ G_crys                    (Å⁻¹)
      - Let ``u_i`` be the incident beam direction in the lab frame (unit vector).
        The Laue + elastic condition yields a unique wavelength:
            lambda_A = -2 * (u_i · G_lab) / ||G_lab||²     (Å)
        Physical solutions require ``lambda_A > 0`` (equivalently ``u_i·G_lab < 0``).
      - Photon energy:
            E_keV = HC_KEV_A / lambda_A
      - Outgoing ray direction (unit vector in lab frame):
            u_f ∝ u_i + lambda_A * G_lab
        (renormalized for numerical safety)

    Projection
    ----------
    Outgoing rays are projected onto the detector using ``pxrad.geometry.projection.ray_to_pixel``.
    Rays outside the detector (or invalid geometry) yield NaN pixel coordinates.

    Angle conventions (important)
    -----------------------------
    If ``compute_angles=True``, angles are computed via ``pxrad.geometry.scattering.scattering_angles``:
      - ``two_theta`` is the *scattering angle* between incident and outgoing directions:
            two_theta = arccos(u_f · u_i)
        This matches the common "2θ" reported by LaueTools for spot geometry (what you observed
        as LaueTools' ``two_theta``).
      - ``chi`` is the azimuthal angle of the outgoing ray around the beam axis in a beam-aligned basis.

    Orientation matrix convention (LaueTools note)
    ----------------------------------------------
    Here ``R`` is the rotation that maps vectors from the *crystal frame* to the *lab frame*.
    Many LaueTools routines use the inverse mapping (lab → crystal). For a pure rotation matrix:
        R_pxrad = inv(R_LT) = R_LT.T

    Filtering
    ---------
    The following filters can be applied:
      - Extinctions (systematic absences): ``material.is_allowed(hkl)`` if ``filter_extinctions=True``.
      - Physical Laue solutions: finite ``lambda_A`` and ``lambda_A > 0`` (always applied).
      - Beam energy window: ``Emin <= E_keV <= Emax`` if ``filter_energy_range=True``.
      - On-detector: finite ``uv`` if ``filter_on_detector=True``.

    Parameters
    ----------
    material:
        Material providing the reciprocal basis (via lattice) and extinction rules.
    R:
        (3,3) rotation matrix mapping crystal → lab.
    geometry:
        Experimental geometry in the fixed lab frame. Uses ``geometry.beam_dir`` as incident direction.
    pose:
        Detector pose (position/orientation) in the lab frame.
    beam:
        Beam model providing the energy range in keV (used for filtering if enabled).
    detector:
        Detector description (shape, pixel size, etc.) used by the projection model.
    hkls:
        (N,3) integer array of candidate Miller indices (e.g. from ``pxrad.laue.hkl.generate_hkls``).
    filter_extinctions:
        If True, remove reflections forbidden by ``material.is_allowed``.
    filter_energy_range:
        If True, keep only reflections with energies inside ``beam.energy_range_keV``.
    filter_on_detector:
        If True, keep only reflections that project onto the detector (finite ``uv``).
        If False, returned ``uv`` may contain NaNs and ``on_detector`` marks valid hits.
    compute_angles:
        If True, compute and return angles (two_theta, chi) for each retained reflection.
    eps:
        Numerical threshold used to detect degenerate vectors and avoid division by zero.

    Returns
    -------
    LaueForwardResult
        Dataclass containing arrays for the retained reflections:
        ``hkl, G_lab, lambda_A, E_keV, uf, angles, uv, on_detector``.
        If ``filter_on_detector=True``, all returned reflections are guaranteed on the detector.

    Notes
    -----
    - This is a *positions-only* forward simulation: intensities are not modeled here.
    - For repeated simulations at many orientations, precompute ``hkls`` once and reuse it
      (and potentially cache ``material.G(hkls)``) to reduce overhead.
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
    g2 = np.where(g2 > eps, g2, np.nan)
    
    lambda_A = -2.0 * ug / g2

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
    # Remove NaNs, that mean "bad direction" or "out of detector"
    on_det = np.isfinite(uv).all(axis=1) & np.isfinite(uv).all(axis=1)

    # Angles (optional)
    if compute_angles:
        two_theta, chi = scattering_angles(uf, beam_dir=ui, degrees=True, eps=eps)
        angles = np.stack([two_theta, chi], axis=1)
    else:
        angles = np.full((uf.shape[0], 2), np.nan, dtype=float)

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
    """
    Convenience wrapper to simulate a Laue pattern (generate HKLs + forward project).

    This function first generates a finite set of candidate reflections using
    ``pxrad.laue.hkl.generate_hkls`` and then calls ``forward_laue`` to compute
    energies, outgoing directions, and detector positions.

    HKL generation
    --------------
    Candidate HKLs are generated by applying:
        - a d-spacing cutoff (either provided via ``dmin_A`` or derived conservatively
        from the beam maximum energy),
        - extinction rules from the material,
        - removal of (0,0,0),
        - optional inclusion of negative indices.

    Use this wrapper for convenience. For high-throughput use (many orientations),
    prefer calling ``generate_hkls`` once and reusing the result with ``forward_laue``.

    Parameters
    ----------
    material, R, geometry, pose, beam, detector:
        Passed through to ``forward_laue`` (see its docstring for details).
    dmin_A:
        Minimum d-spacing (Å) used to generate candidate HKLs. If None, it is computed
        from the beam energy range via ``dmin = lambda_min/2``.
    nmax:
        Optional explicit search radius in index space for HKL generation. If None, it is
        estimated conservatively from ``material`` and ``dmin_A``.
    include_negatives:
        If True, generate both positive and negative HKLs (recommended for Laue).
    filter_extinctions, filter_energy_range, filter_on_detector, compute_angles:
        Passed to ``forward_laue``.

    Returns
    -------
    LaueForwardResult
        Same output as ``forward_laue`` for the generated candidate HKLs.

    See Also
    --------
    pxrad.laue.hkl.generate_hkls
    pxrad.laue.forward.forward_laue
    """
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
