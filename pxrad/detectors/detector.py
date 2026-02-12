from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from numpy.typing import ArrayLike, DTypeLike

from pxrad.utils.linalg import vec2, ensure_finite

@dataclass(frozen=True, slots=True)
class Detector:
    """
    Detector description
    
    name: 
    shape: (ny, nx) pixel dimensions
    pixel_size: (py, px) in meters
    """
    name: str
    shape: tuple[int, int] # (ny, nx)
    pixelsize: ArrayLike  # (py, px) in meters
    encoding: DTypeLike
    file_extension: str
    # distortion: DistortionModel | None = None # to be added later
    # notes: str
    
    def __post_init__(self):
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("Detector.name must be a non-empty string.")
        
        ny, nx = self.shape
        if not isinstance(ny, int) or not isinstance(nx, int) or ny <= 0 or nx <= 0:
            raise ValueError(f"Detector.shape must be (ny,nx) positive ints, got {self.shape}.")

        ps = ensure_finite(vec2(self.pixelsize), "pixelsize")

        if np.any(ps <= 0):
            raise ValueError(f"Detector.pixelsize must be positive, got {ps}.")
        object.__setattr__(self, "pixelsize", ps)
        
        dt = np.dtype(self.encoding)
        if dt.kind not in ("u", "i", "f"):
            raise ValueError(f"Detector.encoding must be integer or float dtype, got {dt}.")
        object.__setattr__(self, "encoding", dt)
        
        ext = str(self.file_extension).strip().lower()
        if ext.startswith("."):
            ext = ext[1:]
        if not ext:
            raise ValueError("Detector.file_extension must be non-empty.")
        object.__setattr__(self, "file_extension", ext)

    @property
    def ny(self) -> int:
        return self.shape[0]

    @property
    def nx(self) -> int:
        return self.shape[1]

    @property
    def size(self) -> tuple[float, float]:
        """Physical size (height, width) in meters."""
        py, px = self.pixelsize
        ny, nx = self.shape
        return (float(ny * py), float(nx * px))
    
    @property
    def diameter(self) -> float:
        # detectors are square, so I take as diameter the diagonal of the square
        rx, ry = self.size
        d = max(rx, ry) * np.sqrt(2)
        return float(d)
    
    @property
    def radius(self) -> float:
        return self.diameter / 2