import numpy as np

############################
# LINEAR ALGEBRA UTILITIES
############################

def vec2(v) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    if v.shape != (2,):
        raise ValueError(f"Expected a 2-vector with shape (2,), got {v.shape}")
    return v
    
def vec3(v) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    if v.shape != (3,):
        raise ValueError(f"Expected a 3-vector with shape (3,), got {v.shape}")
    return v

def ensure_finite(v, name="vector") -> np.ndarray:
    v = np.asarray(v, dtype=float)
    if not np.all(np.isfinite(v)):
        raise ValueError(f"{name} must be finite.")
    return v

def unit(v, *, eps: float = 1e-15) -> np.ndarray:
    """Return v normalized to unit length"""
    v = np.asarray(v, dtype=float)
    n = float(np.linalg.norm(v))
    if n < eps:
        raise ValueError("Cannot normalize a near-zero vector")
    return v / n