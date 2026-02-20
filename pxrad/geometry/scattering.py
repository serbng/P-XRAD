from __future__ import annotations

from typing import Tuple
import numpy as np

from pxrad.utils.linalg import unit, normalize, vec3
from pxrad.utils.types import NDArray, Vec3
from pxrad.geometry.frames import LAB_FRAME, ensure_right_handed

def _to_local_basis(beam_dir: Vec3) -> Tuple[Vec3, Vec3, Vec3]:
    """
    Build a right-handed orthonormal basis (x', y', z') with z' aligned to beam_dir.

    The lab frame is fixed; this basis is only a temporary coordinate system to
    express outgoing rays and measure angles relative to the beam direction.

    Notes
    -----
    The choice of x' is defined using a stable hint axis (lab x_hat unless nearly
    parallel to beam_dir, in which case we use lab y_hat).
    """
    z = unit(vec3(beam_dir))
    
    x_hint = LAB_FRAME.x_hat
    if abs(np.dot(z, x_hint)) > 0.99:
        x_hint = LAB_FRAME.y_hat

    x, y, z = ensure_right_handed(z_hat=z, x_hint=x_hint)
    return x, y, z

def scattering_angles(
    ray_dir: NDArray,
    *,
    beam_dir: Vec3 = LAB_FRAME.z_hat,
    degrees: bool = True,
    eps: float = 1e-15,
) -> Tuple[NDArray, NDArray]:
    """
    Compute scattering angles for outgoing ray direction(s) relative to the incident beam.

    This function returns:
      - two_theta: the *scattering angle* between the incident beam direction and the
        outgoing ray direction, i.e. ``two_theta = arccos(u_f · u_i)``.
      - chi: the azimuthal angle of the outgoing ray around the beam axis.

    **Important convention**
    ------------------------
    In pxrad, ``two_theta`` is the *scattering angle* (angle between incident and outgoing
    directions). This corresponds to what LaueTools often reports as ``twicetheta`` / the
    "2θ" output used for Laue spot geometry (not the diffractometer-style in the x'z' plane).

    Geometry
    --------
    The laboratory frame is fixed. We construct a temporary beam-aligned orthonormal basis
    (x', y', z') such that:
      - z' is parallel to ``beam_dir`` (incident direction)
      - (x', y', z') is right-handed

    Any outgoing direction u_f (unit) is decomposed as:
        u_f = u_x x' + u_y y' + u_z z'

    Returned angles
    ---------------
    two_theta:
        Total scattering angle between incident and outgoing directions:
            two_theta = arccos(u_z)
        because u_z = u_f · z' when both are unit vectors.

    chi:
        Azimuth of the projection of u_f onto the transverse x'y' plane, measured
        from +x' toward +y':
            chi = atan2(u_y, u_x)
        chi is in (-pi, pi] (or degrees in (-180, 180]).

    Parameters
    ----------
    ray_dir : (3,) or (N,3) array_like
        Outgoing ray direction(s) in the lab frame. Need not be unit length.
    beam_dir : (3,) array_like, keyword-only
        Incident beam direction in the lab frame. Must be finite and non-zero.
    degrees : bool, keyword-only
        If True, return angles in degrees. If False, return radians.
    eps : float, keyword-only
        Rays with norm <= eps (or non-finite) yield NaN angles.

    Returns
    -------
    two_theta : float or (N,) ndarray
        Scattering angle between incident beam and outgoing ray.
    chi : float or (N,) ndarray
        Azimuth around the beam axis in the beam-aligned basis.

    Notes
    -----
    - If ``ray_dir`` is a single (3,) vector, scalars are returned.
      If it is (N,3), arrays of shape (N,) are returned.
    - Invalid rays (non-finite or too small) return NaNs.
    """
    ray_dir = np.asarray(ray_dir, dtype=float)
    single_ray = (ray_dir.ndim == 1)
    if single_ray:
        ray_dir = ray_dir[None, :]
    if ray_dir.ndim != 2 or ray_dir.shape[1] != 3:
        raise ValueError(f"ray_dir must have shape (3,) or (N,3), got {ray_dir.shape}.")
    
    # Mask on invalid direction(s)
    n = np.linalg.norm(ray_dir, axis=1)
    valid = np.isfinite(n) & (n > eps)
    # Direction(s) of the scattered wavevector(s)
    uf = np.empty_like(ray_dir, dtype=float)
    uf[valid] = normalize(ray_dir[valid], axis=1, eps=eps)
    uf[~valid] = np.nan
    
    # Reference frame with z' // beam_dir
    x, y, z = _to_local_basis(beam_dir)
    
    # Components of uf in the local reference frame
    uf_x = uf @ x
    uf_y = uf @ y
    uf_z = uf @ z
    
    # For numerical safety
    uf_z = np.clip(uf_z, -1.0, 1.0)
    
    #two_theta_xz = np.arctan2(np.abs(uf_x), uf_z) # diffractometer-like 2theta
    chi       = np.arctan2(uf_y, uf_x) # diffractometer-like chi
    two_theta = np.arccos(uf_z)        # real scattering angle
    
    if degrees:
        two_theta = np.degrees(two_theta)
        chi       = np.degrees(chi)
    
    if single_ray:
        return two_theta[0], chi[0]
    
    return two_theta, chi