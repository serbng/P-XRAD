from __future__ import annotations
from dataclasses import dataclass
import numpy as np

from pxrad.detectors.detector import Detector
from pxrad.utils.linalg import vec2, vec3, unit, normalize

@dataclass(frozen=True)
class DetectorPose:
    det_dir: np.ndarray    # (3,) sample -> PONI (unit)
    det_norm: np.ndarray   # (3,) plane normal (unit)
    distance: float        # meters (|p_poni|)
    spin: float            # radians
    poni: np.ndarray       # (2,) meters in detector frame
    
    def poni_lab_frame(self) -> np.ndarray:
        return vec3(self.distance * unit(self.det_dir))
    
def det_basis_from_norm_spin(det_norm, spin, ref_axis=np.array([0.0, 1.0, 0.0])):
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

def intersection_ray_detector(ray_dir, poni, det_norm, *, eps: float = 1e-15):
    n = unit(vec3(det_norm))
    d = unit(ray_dir)
    
    num = float(np.dot(poni, n))
    # To extent dot product to batches, it is computed using np.sum
    den = np.sum(d * n, axis=-1)
    
    t = num / den
    # Where the ray is almost parallel to the detector, give nan
    t = np.where(np.abs(den) < eps, np.nan, t)
    
    return d * t[..., None] # result is in [m]

def in_detector_bounds(u, v, shape):
    H, W = shape
    return (u >= 0) & (u < W) & (v >= 0) & (v < H)

def ray_to_pixel(ray_dir, pose: DetectorPose, detector: Detector):
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
    
    px_s = detector.pixelsize
    
    u = x / px_s[0]
    v = y / px_s[1]
    
    uv = np.stack((u, v), axis=-1)
    
    valid = in_detector_bounds(u, v, detector.shape)
    
    uv = np.where(valid[..., None], uv, np.nan)
    
    return uv[0] if single_ray else uv