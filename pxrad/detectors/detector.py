from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Union
import numpy as np
from numpy.typing import ArrayLike, NDArray, DTypeLike

from pxrad.utils.linalg import vec2, ensure_finite

Vec2 = NDArray[np.floating] # shape (2,)

@dataclass(frozen=True, slots=True)
class Detector:
    """"
    Detector metadata and basic geometry.

    Parameters
    ----------
    name : str
        Human-readable detector identifier (e.g. "EIGER_4M").
    shape : (ny, nx)
        Image dimensions in pixels: (number of rows, number of columns).
    pixelsize : (py, px) array_like
        Pixel size in meters per pixel: (pixel height, pixel width).
        Note: the order matches `shape` = (ny, nx).
    encoding : dtype-like
        Numpy dtype (or something convertible to dtype) describing the stored pixel type
        (e.g. "uint16", np.uint32, np.float32).
    file_extension : str
        Default file extension for raw frames (e.g. "tif", "h5"). A leading "." is allowed.

    Notes
    -----
    This class only stores detector metadata. Geometric pose (distance/tilt/poni/etc.)
    lives elsewhere (e.g. geometry/projection and calibration).
    """
    name: str
    shape: Tuple[int, int] # (ny, nx)
    pixelsize: ArrayLike  # (py, px) in meters
    encoding: DTypeLike
    file_extension: str
    # distortion: DistortionModel | None = None # to be added later
    # notes: str
    
    def __post_init__(self):
        # name
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("Detector.name must be a non-empty string.")
        
        # shape
        ny, nx = self.shape
        if not isinstance(ny, int) or not isinstance(nx, int) or ny <= 0 or nx <= 0:
            raise ValueError(f"Detector.shape must be (ny,nx) positive ints, got {self.shape}.")

        # pixelsize: enforce finite, positive, float, shape (2,)
        ps = ensure_finite(vec2(self.pixelsize), "pixelsize")

        if np.any(ps <= 0):
            raise ValueError(f"Detector.pixelsize must be positive, got {ps}.")
        object.__setattr__(self, "pixelsize", ps)
        
        # encoding
        dt = np.dtype(self.encoding)
        if dt.kind not in ("u", "i", "f"):
            raise ValueError(f"Detector.encoding must be integer or float dtype, got {dt}.")
        object.__setattr__(self, "encoding", dt)
        
        # file_extension: normalize to lowercase without dot
        ext = str(self.file_extension).strip().lower()
        if ext.startswith("."):
            ext = ext[1:]
        if not ext:
            raise ValueError("Detector.file_extension must be non-empty.")
        object.__setattr__(self, "file_extension", ext)

    @property
    def ny(self) -> int:
        """Number of rows (vertical pixels)."""
        return self.shape[0]

    @property
    def nx(self) -> int:
        """Number of columns (horizontal pixels)."""
        return self.shape[1]

    @property
    def size(self) -> Vec2:
        """
        Physical detector size in meters.

        Returns
        -------
        (height, width) : (2,) ndarray
            height = ny * py, width = nx * px
        """
        py, px = self.pixelsize
        ny, nx = self.shape
        return np.array([ny * py, nx * px])
    
    @property
    def diagonal(self) -> float:
        """
        Physical diagonal length in meters.

        Returns
        -------
        float
            sqrt(height^2 + width^2)
        """
        h, w = self.size
        return float(np.hypot(h, w))
    
    @property
    def radius(self) -> float:
        """
        Half of the diagonal length in meters.

        Notes
        -----
        This is a convenience number sometimes used as a loose "active radius" when you
        want a single scalar scale. For rectangular detectors, this corresponds to the
        circumradius of the bounding rectangle (half-diagonal).
        """
        return 0.5 * self.diagonal