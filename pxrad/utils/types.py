from __future__ import annotations

import numpy as np
from numpy.typing import NDArray, ArrayLike

FloatArray = NDArray[np.floating]
IntArray   = NDArray[np.integer]
BoolArray  = NDArray[np.bool_]

Vec2 = NDArray[np.floating]  # intended shape (2,)
Vec3 = NDArray[np.floating]  # intended shape (3,)
Mat33 = NDArray[np.floating] # intended shape (3,3)

__all__ = [
    "ArrayLike",
    "FloatArray", "IntArray", "BoolArray",
    "Vec2", "Vec3", "Mat33",
]