from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

import numpy as np
from numpy.typing import NDArray

from pxrad.detectors.detector import Detector
from pxrad.geometry.frames import Geometry
from pxrad.geometry.pose import DetectorPose
from pxrad.geometry.projection import ray_to_pixel
from pxrad.utils.linalg import vec2, vec3, unit

from scipy.optimize import least_squares

Vec2 = NDArray[np.floating] # (2, )
Vec3 = NDArray[np.floating] # (3, )

@dataclass(frozen=True, slots=True)
class Calibration:
    pose: DetectorPose
    success: bool
    rms: float
    

def _skew_matrix(w: Vec3) -> NDArray:
    """
    Return the 3x3 skew-symmetric matrix [w]_x such that [w]_x @ v == w × v.

    Parameters
    ----------
    w : (3,) array_like
        Vector defining the cross-product operator.

    Returns
    -------
    (3,3) ndarray
        Skew-symmetric cross-product matrix.
    """
    wx, wy, wz = vec3(w)
    
    return np.array([
        [0.0, -wz,  wy],
        [ wz, 0.0, -wx],
        [-wy,  wx, 0.0]
    ], dtype=float)
    
def rodrigues_rotation_matrix(rvec: Vec3) -> NDArray:
    """
    Convert a rotation vector (axis * angle) to a 3x3 rotation matrix using Rodrigues' formula.

    The rotation vector representation is:
      - direction(rvec) = rotation axis
      - norm(rvec)      = rotation angle (radians)

    Parameters
    ----------
    rvec : (3,) array_like
        Rotation vector in radians.

    Returns
    -------
    (3,3) ndarray
        Proper rotation matrix (orthonormal, det ~ 1).

    Notes
    -----
    For small angles (||rvec|| ~ 0), this returns the identity.
    """
    rvec = np.asarray(rvec, dtype=float)
    theta = float(np.linalg.norm(rvec))
    if theta < 1e-15:
        return np.eye(3)

    k = rvec / theta
    K = _skew_matrix(k)
    c = np.cos(theta)
    s = np.sin(theta)
    I = np.eye(3, dtype=float)

    # R = I*c + (1-c) kk^T + s [k]_x
    return I * c + (1 - c) * np.outer(k, k) + s * K

def _pose_from_params(
    x: NDArray,
    geometry: Geometry,
    detector: Detector,
    *,
    dist0: float,
    spin0: float
) -> DetectorPose:
    """
    Build a DetectorPose from a parameter vector `x` interpreted as deltas w.r.t. a nominal pose.

    The nominal pose is:
        pose0 = DetectorPose.nominal(geometry, detector, distance=dist0, spin=spin0)

    Conventions / frames
    --------------------
    - dist0, delta_dist are meters.
    - poni is in the *detector frame* (meters), consistent with DetectorPose and projection.py.
    - spin is radians.
    - tilt is a rotation vector (radians) applied in the *lab frame* to the nominal det_norm.
      det_norm is then renormalized.

    Returns
    -------
    DetectorPose
        Pose corresponding to x.
    """
    delta_dist = float(x[0])
    delta_poni = vec2(x[[1,2]])
    delta_spin = float(x[3])
    tilt = vec3(x[[4,5,6]])
    
    # Nominal pose:
    pose0 = DetectorPose.nominal(geometry, detector, distance=dist0, spin=spin0)
    
    dist = float(dist0 + delta_dist)
    spin = float(spin0 + delta_spin)
    poni = vec2(pose0.poni + delta_poni)

    # tilt updates detector normal in the lab frame
    R = rodrigues_rotation_matrix(tilt)
    det_norm = unit(R @ pose0.det_norm)
    
    return DetectorPose(
        det_dir=pose0.det_dir,
        det_norm=det_norm,
        distance=dist,
        spin=spin,
        poni=poni
    )
    
def _residuals(
    x: NDArray,
    geometry: Geometry,
    detector: Detector,
    rays: NDArray,
    uv_exp: NDArray,
    *,
    dist0: float,
    spin0: float
) -> NDArray:
    
    pose = _pose_from_params(x, geometry, detector, dist0=dist0, spin0=spin0)
    uv_pred = ray_to_pixel(rays, pose, detector)
    
    # Mask removing the NaNs. finite contains the indices of the row without NaNs
    finite = np.isfinite(uv_pred).all(axis=1) & np.isfinite(uv_exp).all(axis=1)
    if not np.any(finite):
        # If all of the elements of finite are False return a big residual
        # so that the solver will know "this is bad"
        return np.array([1e6], dtype=float)
    
    r = (uv_pred[finite] - uv_exp[finite]).reshape(-1)
    
    return r # shape (2*M,) where M is the number of valid pairs.

def calibrate(
    geometry: Geometry,
    detector: Detector,
    *,
    rays: NDArray,
    uv_exp: NDArray,
    dist0: float = 80e-3,
    spin0: float = 0.0,
    x0: Optional[NDArray] = None,
    method: str = "trf"
) -> Calibration:
    rays = np.asarray(rays, dtype=float)
    uv_exp = np.asarray(uv_exp, dtype=float)
    
    if rays.ndim != 2 or rays.shape[1] != 3:
        raise ValueError(f"rays must have shape (N,3), got {rays.shape}")
    if uv_exp.ndim != 2 or uv_exp.shape[1] != 2:
        raise ValueError(f"uv_obs must have shape (N,2), got {uv_exp.shape}")
    if rays.shape[0] != uv_exp.shape[0]:
        raise ValueError("rays and uv_obs must have the same length N")
    
    # Initial guess of the deviation from the nominal position
    if x0 is None:
        x0 = np.zeros(7, dtype=float)
    else:
        x0 = np.asarray(x0, dtype=float)
        if x0.shape != (7,):
            raise ValueError(f"x0 must have shape (7,), got {x0.shape}")
        
    fun = lambda x: _residuals(
        x, 
        geometry, 
        detector,
        rays,
        uv_exp,
        dist0=dist0,
        spin0=spin0
    )
    
    res = least_squares(fun, x0, method=method)
    
    # Optimal pose
    pose = _pose_from_params(res.x, geometry, detector, dist0=dist0, spin0=spin0)
    # Residues
    r   = fun(res.x)
    # RMS value in pixels
    rms = float(np.sqrt(np.mean(r**2))) if r.size > 0 else float("nan")
    
    
    
    