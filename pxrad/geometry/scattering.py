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
    eps: float = 1e-15    
) -> Tuple[NDArray, NDArray, NDArray]:
    """
    Compute scattering angles (2θ and χ) for outgoing ray direction(s) relative to a beam direction.

    Geometry
    --------
    The laboratory frame is fixed. We construct a *beam-aligned* orthonormal basis (x', y', z')
    where z' is parallel to ``beam_dir``. This basis is used only to define angles.

    Let ``u_f`` be the outgoing ray direction (normalized) and write it in that basis:

        u_f = u_x * x' + u_y * y' + u_z * z'

    Returned angles
    ---------------
    This function returns three angles (in radians or degrees):

    1) two_theta:
       Angle between z' and the projection of u_f onto the x'z' plane.

       The projection is (u_x, 0, u_z). The angle is:

           two_theta = atan2(|u_x|, u_z)

       With the absolute value, ``two_theta_xz`` is always non-negative.
       If you want a signed left/right angle in the x' direction, replace |u_x| with u_x.

    2) chi:
       Azimuthal angle of the projection of u_f onto the transverse x'y' plane,
       measured from +x' toward +y':

           chi = atan2(u_y, u_x)

       This is a signed angle in (-pi, pi].

    3) two_theta_scattering:
       The *total* scattering angle between the incident beam direction and u_f:

           two_theta_scattering = arccos(u_z)

       (because u_z = u_f · z' when both are unit vectors).

    Parameters
    ----------
    ray_dir : (3,) or (N,3) array_like
        Outgoing ray direction(s) in the lab frame. Need not be unit length.
    beam_dir : (3,) array_like, keyword-only
        Incident beam direction in the lab frame (default: LAB_FRAME.z_hat).
        Must be finite and non-zero.
    degrees : bool, keyword-only
        If True, return angles in degrees. If False, return radians.
    eps : float, keyword-only
        Threshold for detecting invalid ray directions. Any ray with norm < eps (or non-finite)
        yields NaN angles.

    Returns
    -------
    two_theta : float or (N,) ndarray
        2θ defined in the x'z' plane (non-negative by default).
    chi : float or (N,) ndarray
        χ azimuth in the x'y' plane (signed).
    two_theta_scattering : float or (N,) ndarray
        Total scattering angle between beam and outgoing ray.

    Notes
    -----
    - If ``ray_dir`` is a single (3,) vector, scalars are returned.
      If it is (N,3), arrays of shape (N,) are returned.
    - Rays that cannot be normalized (norm < eps or non-finite) return NaNs.
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
    
    two_theta = np.arctan2(np.abs(uf_x), uf_z) # diffractometer-like 2theta
    chi       = np.arctan2(uf_y, uf_x)         # diffractometer-like chi
    two_theta_scattering = np.arccos(uf_z)     # real scattering angle
    
    if degrees:
        two_theta = np.degrees(two_theta)
        chi       = np.degrees(chi)
        two_theta_scattering = np.degrees(two_theta_scattering)
    
    if single_ray:
        return two_theta[0], chi[0], two_theta_scattering[0]
    
    return two_theta, chi, two_theta_scattering