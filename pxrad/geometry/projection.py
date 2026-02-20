from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple
import numpy as np

from pxrad.detectors.detector import Detector
from pxrad.geometry.pose import DetectorPose
from pxrad.utils.linalg import vec2, vec3, unit
from pxrad.utils.types import Vec3, NDArray
    
def det_basis_from_norm_spin(
    det_norm: Vec3, 
    spin: float, 
    ref_axis: Vec3 = np.array([0.0, 1.0, 0.0])
) -> Tuple[Vec3, Vec3, Vec3]:
    """
    Build an orthonormal detector basis (ex, ey, ez) in the lab frame.

    The detector plane normal is `ez = unit(det_norm)`. A reference axis (default: lab y_hat)
    is projected onto the detector plane to define a stable in-plane direction (ey0),
    then a spin rotation around ez defines the final (ex, ey).

    Parameters
    ----------
    det_norm : (3,) array_like
        Detector plane normal in lab frame (does not need to be unit; it will be normalized).
    spin : float
        Rotation angle (radians) around ez. Defines the in-plane orientation of ex/ey.
    ref_axis : (3,) array_like, optional
        Reference lab axis used to define the in-plane "up" direction before applying spin.
        If nearly parallel to det_norm, it is replaced by x_hat.

    Returns
    -------
    ex : (3,) ndarray
        Detector x axis expressed in the lab frame (unit vector, lies in detector plane).
    ey : (3,) ndarray
        Detector y axis expressed in the lab frame (unit vector, lies in detector plane).
    ez : (3,) ndarray
        Detector normal expressed in the lab frame (unit vector).
    """
    n = unit(vec3(det_norm))
    a = unit(vec3(ref_axis))
    # If a and n are almost parallel, change ref_axis
    if abs(np.dot(a, n)) > 0.99:
        a = unit(vec3([1.0, 0.0, 0.0]))
    # Projection of a, normally y_hat, on the plane
    ey0 = unit(a - np.dot(a, n) * n)
    ex0 = unit(np.cross(ey0, n))
    
    cspin, sspin = np.cos(spin), np.sin(spin)
    
    ex =  cspin * ex0 + sspin * ey0
    ey = -sspin * ex0 + cspin * ey0
    
    return ex, ey, n

def intersection_ray_detector(
    ray_dir: NDArray, 
    poni: Vec3, 
    det_norm: Vec3, 
    *, 
    eps: float = 1e-15
) -> NDArray:
    """
    Intersect ray(s) from the sample origin with the detector plane.

    The detector plane is defined by:
      - a point on plane: `poni_lab` (PONI point in lab, meters)
      - normal: `det_norm` (lab frame)

    Each ray is defined by:
        x(t) = t * d
    with origin at the sample (0,0,0) and direction d (unit vector).

    Parameters
    ----------
    ray_dir : (3,) or (N,3) array_like
        Ray direction(s) in the lab frame. Will be normalized internally.
    poni : (3,) array_like
        Point on plane (PONI) in lab frame, meters.
    det_norm : (3,) array_like
        Plane normal in lab frame. Will be normalized internally.
    eps : float, optional
        Threshold for near-parallel rays: if |d·n| < eps, the intersection parameter t is set to NaN.

    Returns
    -------
    P : (3,) or (N,3) ndarray
        Intersection point(s) in the lab frame, meters.
        For near-parallel rays, returns NaNs for those entries.
    """
    n = unit(vec3(det_norm))
    d = unit(ray_dir)
    
    num = float(np.dot(poni, n))
    # To extent dot product to batches, it is computed using np.sum
    den = np.sum(d * n, axis=-1)
    
    # Avoid doing the division where the denominator is close to 0
    # and when the solution is found in a direction opposite the one
    # of the detector.
    t = np.full_like(den, np.nan, dtype=float)
    # Given our convention n \cdot d is always negative. So I do the
    # division only when the denominator is negative (and in magnitude)
    # larger than eps
    np.divide(num, den, out=t, where=den < -eps)
    
    return d * t[..., None] # result is in [m]

def in_detector_bounds(u: NDArray, v: NDArray, shape: Tuple[int, int]) -> NDArray[np.bool]:
    """
    Check whether pixel coordinates lie within the detector image bounds.

    Parameters
    ----------
    u, v : float or array_like
        Pixel coordinates. Can be scalars or arrays of same broadcastable shape.
    shape : (H, W)
        Detector image shape in pixels.

    Returns
    -------
    valid : bool or ndarray of bool
        True where 0 <= u < W and 0 <= v < H.
    """
    H, W = shape
    return (u >= 0) & (u < W) & (v >= 0) & (v < H)

def ray_to_pixel(
    ray_dir: NDArray, 
    pose: DetectorPose, 
    detector: Detector
) -> NDArray:
    """
    Project ray direction(s) to detector pixel coordinates.

    Reference frames
    ----------------
    - `ray_dir` is in the **lab frame** (origin at sample).
    - Detector plane is defined by `pose.det_norm` and the PONI point:
        p_poni = pose.distance * unit(pose.det_dir)
    - Detector axes (ex, ey) are derived from (det_norm, spin) and are expressed in **lab frame**.
    - `pose.poni = (poni_x, poni_y)` is in the **detector frame** (meters).

    Pixel mapping
    -------------
    Pixel origin (u=0,v=0) corresponds to a point p00 in lab:
        p00 = p_poni - poni_x * ex - poni_y * ey

    For an intersection point P on the plane:
        x = (P - p00)·ex   [meters]
        y = (P - p00)·ey   [meters]
        u = x / pixelsize_x
        v = y / pixelsize_y

    Parameters
    ----------
    ray_dir : (3,) or (N,3) array_like
        Ray direction(s) in lab frame.
    pose : DetectorPose
        Detector pose definition.
    detector : Detector
        Detector metadata (pixelsize, shape).

    Returns
    -------
    uv : (2,) or (N,2) ndarray
        Pixel coordinates (u, v). Rays that are either:
          - near-parallel to the plane (|d·n| < eps), or
          - whose (u,v) fall outside `detector.shape`
        are returned as NaNs.
    """
    ray_dir = np.asarray(ray_dir, dtype=float)
    
    single_ray = ray_dir.ndim == 1
    if single_ray:
        ray_dir = ray_dir[None, :] # (1, 3)
        
    ray_dir = unit(ray_dir) # (N, 3)
    
    ex, ey, ez = det_basis_from_norm_spin(pose.det_norm, pose.spin)
    poni_lab_frame = pose.poni_lab_frame() # vector (3,)
    poni_det_frame = pose.poni             # vector (2,)
    
    # pixel origin (0,0) in the lab frame in meters
    p00 = poni_lab_frame - poni_det_frame[0] * ex \
                         - poni_det_frame[1] * ey
    
    # points on the detector in the lab frame
    P = intersection_ray_detector(ray_dir, poni_lab_frame, pose.det_norm)
    
    x = np.sum((P - p00) * ex, axis=-1)
    y = np.sum((P - p00) * ey, axis=-1)
    
    px_s = vec2(detector.pixelsize)
    
    # Remember that pixelsize = (py, px)
    u = x / px_s[1]
    v = y / px_s[0]
    
    uv = np.stack((u, v), axis=-1)
    
    valid = in_detector_bounds(u, v, detector.shape)
    
    uv = np.where(valid[..., None], uv, np.nan)
    
    return uv[0] if single_ray else uv