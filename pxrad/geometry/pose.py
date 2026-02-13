from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray

from pxrad.utils.linalg import vec2, vec3, unit
from pxrad.geometry.frames import Geometry
from pxrad.detectors.detector import Detector

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
    
    @classmethod
    def nominal(
        cls,
        geometry: Geometry,
        detector: Detector,
        *,
        distance: float = 80e-3, # 8 cm from the sample
        spin: float = 0
    ) -> "DetectorPose":
        """
        Construct a nominal (idealized) detector pose from a `Geometry` and `Detector`.

        This helper provides a deterministic starting pose for calibration and projection.
        It assumes an *ideal alignment* where the detector plane is perfectly facing the
        sample: the detector normal is opposite to the detector direction.

        Reference frames and conventions
        --------------------------------
        - All 3D vectors are expressed in the fixed laboratory frame (LAB_FRAME).
        - `det_dir` is taken from `geometry.det_dir` (unit vector, sample → detector/PONI).
        - `det_norm` is set to `-geometry.det_dir` (unit vector), meaning the detector plane
          is orthogonal to `det_dir` and faces the sample.
        - The PONI offset `poni` is set to the physical detector center in the detector frame:
          `detector.size / 2`, expressed in meters as (height/2, width/2).
        - `distance` is the sample → PONI distance in meters.
        - `spin` is the in-plane rotation (radians) about `det_norm`.

        Notes
        -----
        - This method works for all geometry modes, including CUSTOM, as long as the provided
          `Geometry` instance has a valid `det_dir`.
        - The choice `det_norm = -det_dir` is a convention; depending on your downstream
          sign conventions you may obtain d·n < 0 for rays pointing toward the detector.

        Parameters
        ----------
        geometry : Geometry
            Experimental geometry in the lab frame. Provides `det_dir`.
        detector : Detector
            Detector metadata. Used here only to place the PONI at the detector center.
        distance : float, optional
            Sample → PONI distance in meters. Default is 80e-3 (8 cm).
        spin : float, optional
            In-plane rotation (radians). Default is 0.

        Returns
        -------
        DetectorPose
            Nominal detector pose suitable as an initial guess for calibration.
        """
        return cls(
            det_dir  =  geometry.det_dir,
            det_norm = -geometry.det_dir, # the detector faces perfectly the sample
            distance = distance,
            spin = spin,
            poni = detector.size / 2.0
        )