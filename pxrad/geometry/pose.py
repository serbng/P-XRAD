from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray

from pxrad.utils.linalg import vec3, unit

Vec3 = NDArray[np.floating] # intended shape (3,)
Vec2 = NDArray[np.floating] # intended shape (2,)


@dataclass(frozen=True)
class DetectorPose:
    """
    Detector pose and PONI definition.

    All vectors are expressed in the **lab frame** unless stated otherwise.

    Parameters
    ----------
    det_dir : (3,) array_like
        Unit vector in the lab frame pointing from sample (origin) to the PONI point.
        After calibration, `det_dir` is intended to point exactly to the PONI.
    det_norm : (3,) array_like
        Unit normal vector of the detector plane in the lab frame.
        Its sign is conventional (depending on your setup, d·n may be < 0 for all valid rays).
    distance : float
        Distance from sample to the PONI point, in meters. The PONI point in lab is:
            p_poni = distance * det_dir
    spin : float
        In-plane rotation (radians) defining the detector x/y axes around `det_norm`.
    poni : (2,) array_like
        PONI offset expressed in the **detector frame**, in meters:
        (poni_x, poni_y) is the location of the PONI relative to the pixel origin (0,0)
        along the detector axes (ex, ey) returned by `det_basis_from_norm_spin`.

        More precisely, if p_poni is the PONI point in lab and (ex, ey) are detector axes in lab,
        then the pixel origin p00 in lab is:
            p00 = p_poni - poni_x * ex - poni_y * ey
    """
    det_dir: Vec3   
    det_norm: Vec3 
    distance: float
    spin: float    
    poni: Vec2    
    
    def poni_lab_frame(self) -> Vec3:
        return vec3(self.distance * unit(self.det_dir))