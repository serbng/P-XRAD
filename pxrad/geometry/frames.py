from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Tuple, Optional
import numpy as np
from numpy.typing import NDArray

from pxrad.utils.linalg import vec3, unit

Vec3 = NDArray[np.floating] # shape (3,)

class GeometryMode(Enum):
    """
    Available detector placements in the fixed laboratory frame.

    The lab frame is fixed and right-handed. GeometryMode only selects a *nominal*
    detector direction `det_dir` (sample -> detector), unless mode is CUSTOM.

    Notes
    -----
    - TRANSMISSION: detector downstream along +z
    - TOP_REFLECTION: detector along +x
    - BACK_REFLECTION: detector upstream along -z
    - BOTTOM_REFLECTION: detector along -x
    - CUSTOM: user provides `det_dir`
    """
    TRANSMISSION      = auto()
    TOP_REFLECTION    = auto()
    BACK_REFLECTION   = auto()
    BOTTOM_REFLECTION = auto()
    CUSTOM            = auto()


def ensure_right_handed(z_hat: Vec3, x_hint: Vec3) -> Tuple[Vec3, Vec3, Vec3]:
    """
    Build an orthonormal right-handed basis (x, y, z) from two directions.

    Parameters
    ----------
    z_hat : (3,) array_like
        Desired z axis direction.
    x_hint : (3,) array_like
        A vector indicating roughly the +x direction. It does not need to be orthogonal to z_hat.

    Returns
    -------
    x : (3,) ndarray
        Unit x axis.
    y : (3,) ndarray
        Unit y axis, enforced by the fixed right-handed convention.
    z : (3,) ndarray
        Unit z axis.

    Convention
    ----------
    The handedness convention is FIXED:
        y = z × x

    Implementation notes
    --------------------
    Uses a Gram–Schmidt step to make x perpendicular to z, then constructs y and 
    re-orthogonalizes x for numerical stability.
    """
    z = unit(vec3(z_hat))
    xh = vec3(x_hint)

    # Make x perpendicular to z (Gram–Schmidt step)
    x = xh - np.dot(xh, z) * z
    x = unit(x)

    # Right-handed rule
    y = np.cross(z, x)
    y = unit(y)

    # Recompute x for numerical orthogonality
    x = np.cross(y, z)
    x = unit(x)

    return x, y, z

############################
# LAB FRAME DEFINITION
############################

@dataclass(frozen=True, slots=True)
class LabFrame:
    """
    Fixed laboratory frame (x_hat, y_hat, z_hat).

    Attributes
    ----------
    x_hat, y_hat, z_hat : (3,) ndarray
        Orthonormal right-handed basis vectors.

    Notes
    -----
    This frame is global and fixed across the whole library. Geometry modes and detector
    poses are expressed with respect to this frame.
    """
    x_hat: np.ndarray
    y_hat: np.ndarray
    z_hat: np.ndarray

    @classmethod
    def default(cls) -> "LabFrame":
        """
        Construct the default lab frame used by pxrad.

        Default convention (aligned to TOP_REFLECTION intuition):
        - +z is the direction source -> sample
        - +x is the direction sample -> detector (top reflection)

        Returns
        -------
        LabFrame
            The default right-handed lab frame.
        """
        z_hat  = np.array([0.0, 0.0, 1.0])
        x_hint = np.array([1.0, 0.0, 0.0])
        x, y, z = ensure_right_handed(z_hat=z_hat, x_hint=x_hint)
        return cls(x_hat=x, y_hat=y, z_hat=z)
    
    def as_matrix(self) -> NDArray:
        """
        Return the 3x3 basis matrix with columns [x_hat, y_hat, z_hat].

        Returns
        -------
        (3,3) ndarray
            Columns are the basis vectors in lab coordinates.
        """
        return np.column_stack([self.x_hat, self.y_hat, self.z_hat])
    
    def validate(self, *, atol: float = 1e-12) -> None:
        """
        Validate that the frame is orthonormal and right-handed.

        Parameters
        ----------
        atol : float
            Absolute tolerance used in the checks.

        Raises
        ------
        ValueError
            If any basis vector is not unit length, not orthogonal, or not right-handed under:
                y = z × x
        """
        x, y, z = self.x_hat, self.y_hat, self.z_hat

        # unit length
        for name, v in (("x_hat", x), ("y_hat", y), ("z_hat", z)):
            if not np.isclose(np.linalg.norm(v), 1.0, atol=atol):
                raise ValueError(f"{name} is not unit length.")

        # orthogonality
        if not np.isclose(np.dot(x, y), 0.0, atol=atol):
            raise ValueError("x_hat and y_hat are not orthogonal.")
        if not np.isclose(np.dot(y, z), 0.0, atol=atol):
            raise ValueError("y_hat and z_hat are not orthogonal.")
        if not np.isclose(np.dot(z, x), 0.0, atol=atol):
            raise ValueError("z_hat and x_hat are not orthogonal.")

        # explicit handedness rule
        y_ref = np.cross(z, x)
        if not np.allclose(y, y_ref, atol=atol):
            raise ValueError("Not right-handed under the rule y = z × x.")

LAB_FRAME: LabFrame = LabFrame.default()
LAB_FRAME.validate()

############################
# Geometry class
############################

def _det_dir_from_mode(mode: GeometryMode, custom_det_dir: Optional[Vec3] = None) -> Vec3:
    """
    Return the nominal detector direction det_dir (sample -> detector) in the lab frame.

    Parameters
    ----------
    mode : GeometryMode
        Geometry mode selecting a nominal detector placement.
    custom_det_dir : (3,) array_like, optional
        Required only for CUSTOM mode. Will be normalized.

    Returns
    -------
    det_dir : (3,) ndarray
        Unit vector in lab frame pointing from sample -> detector.

    Raises
    ------
    ValueError
        If mode is CUSTOM and custom_det_dir is not provided.
    """
    if mode is GeometryMode.TRANSMISSION:
        return np.array([0.0, 0.0, 1.0])
    if mode is GeometryMode.TOP_REFLECTION:
        return np.array([1.0, 0.0, 0.0])
    if mode is GeometryMode.BACK_REFLECTION:
        return np.array([0.0, 0.0, -1.0])
    if mode is GeometryMode.BOTTOM_REFLECTION:
        return np.array([-1.0, 0.0, 0.0])
    if mode is GeometryMode.CUSTOM:
        if custom_det_dir is None:
            raise ValueError("GeometryMode.CUSTOM requires det_dir (a 3-vector in LAB_FRAME).")
        return unit(vec3(custom_det_dir))
    raise ValueError(f"Unhandled mode: {mode!r}")

@dataclass(frozen=True, slots=True)
class Geometry:
    """
    Experimental geometry definition in the fixed lab frame.

    Parameters
    ----------
    mode : GeometryMode
        Detector placement mode.
    beam_dir : (3,) array_like, optional
        Unit vector in lab frame pointing from source -> sample.
        Default is LAB_FRAME.z_hat.
    det_dir : (3,) array_like, optional
        Unit vector in lab frame pointing from sample -> detector.
        Only allowed (and required) when mode=CUSTOM.

    Attributes
    ----------
    beam_dir : (3,) ndarray
        Normalized beam direction (source -> sample) in lab frame.
    det_dir : (3,) ndarray
        Normalized detector direction (sample -> detector) in lab frame.

    Rules
    -----
    - If mode == CUSTOM, det_dir must be provided.
    - If mode != CUSTOM, det_dir must NOT be provided (it is derived from mode).
    """
    mode: GeometryMode
    beam_dir: Vec3 = field(default_factory = lambda: LAB_FRAME.z_hat.copy())
    det_dir: Optional[Vec3] = None  # only allowed/required for CUSTOM

    def __post_init__(self):
        # Normalize beam_dir (always)
        bd = unit(vec3(self.beam_dir))
        object.__setattr__(self, "beam_dir", bd)

        # Resolve det_dir depending on mode
        if self.mode is GeometryMode.CUSTOM:
            if self.det_dir is None:
                raise ValueError("mode=CUSTOM requires det_dir (a 3-vector in LAB_FRAME).")
            dd = unit(vec3(self.det_dir))
        else:
            if self.det_dir is not None:
                raise ValueError("det_dir is only allowed when mode=CUSTOM.")
            dd = unit(_det_dir_from_mode(self.mode))

        object.__setattr__(self, "det_dir", dd)

    @property
    def frame(self) -> LabFrame:
        """
        Return the (global) lab frame.
        """
        return LAB_FRAME

    def describe(self) -> str:
        """
        Return a multi-line human-readable description of the geometry.
        """
        return (
            "Geometry("
            f"\n    mode={self.mode.name},"
            f"\n    det_dir={self.det_dir.tolist()},"
            f"\n    beam_dir={self.beam_dir.tolist()},"
            f"\n    z_hat={self.frame.z_hat.tolist()},"
            f"\n    x_hat={self.frame.x_hat.tolist()},"
            f"\n    y_hat={self.frame.y_hat.tolist()}\n)"
        )