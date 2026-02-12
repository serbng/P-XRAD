from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
import numpy as np

from pxrad.utils.linalg import vec3, unit

# Finite set of geometry modes
class GeometryMode(Enum):
    """Available detector placements in the fixed laboratory frame"""
    TRANSMISSION      = auto()
    TOP_REFLECTION    = auto()
    BACK_REFLECTION   = auto()
    BOTTOM_REFLECTION = auto()
    CUSTOM            = auto()


def ensure_right_handed(z_hat, x_hint) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build an orthonormal right-handed basis (x, y, z) from:
      - z_hat: desired z direction
      - x_hint: vector indicating roughly the +x direction

    Convention is FIXED: y = z × x.
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
    Fixed lab frame (x_hat, y_hat, z_hat) with explicit right-handed convention.
    """
    x_hat: np.ndarray
    y_hat: np.ndarray
    z_hat: np.ndarray

    @classmethod
    def default(cls) -> "LabFrame":
        # Default convention for the lab frame definition, assuming
        # TOP_REFLECTION GeometryMode:
        # +z is the direction source -> sample
        # +x is the direction sample -> detector
        z_hat  = np.array([0.0, 0.0, 1.0])
        x_hint = np.array([1.0, 0.0, 0.0])
        x, y, z = ensure_right_handed(z_hat=z_hat, x_hint=x_hint)
        return cls(x_hat=x, y_hat=y, z_hat=z)
    
    def as_matrix(self) -> np.ndarray:
        """
        3x3 matrix with columns [x_hat, y_hat, z_hat].
        """
        return np.column_stack([self.x_hat, self.y_hat, self.z_hat])
    
    def validate(self, *, atol: float = 1e-12) -> None:
        """Raise if basis is not orthonormal and right-handed (y = z × x)."""
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

def _det_dir_from_mode(mode: GeometryMode, custom_det_dir: np.ndarray | None = None) -> np.ndarray:
    """
    Unit vector pointing from sample -> detector in the fixed lab frame.
    If mode is CUSTOM, det_dir must be provided.
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
    Experiment geometry in the lab frame (x,y,z).

    mode:
    beam_dir: unit vector (source -> sample) in LAB_FRAME
    det_dir : unit vector (sample -> detector center) in LAB_FRAME

    Rules:
      - If mode == CUSTOM, det_dir must be provided.
      - If mode != CUSTOM, det_dir must NOT be provided (it is derived from mode).
    """
    mode: GeometryMode
    beam_dir: np.ndarray = field(default_factory=lambda: LAB_FRAME.z_hat.copy())
    det_dir: np.ndarray | None = None  # only allowed/required for CUSTOM

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
        # convenient accessor; still a single global frame
        return LAB_FRAME

    def describe(self) -> str:
        return (
            "Geometry("
            f"\n    mode={self.mode.name},"
            f"\n    det_dir={self.det_dir.tolist()},"
            f"\n    beam_dir={self.beam_dir.tolist()},"
            f"\n    z_hat={self.frame.z_hat.tolist()},"
            f"\n    x_hat={self.frame.x_hat.tolist()},"
            f"\n    y_hat={self.frame.y_hat.tolist()}\n)"
        )