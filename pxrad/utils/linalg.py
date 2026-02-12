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

def unit(v, *, axis: int = -1, eps: float = 1e-15) -> np.ndarray:
    """
    Strict normalization to unit length.

    Works for shapes:
      - (3,)
      - (N, 3)
      - (..., 3) in general (normalizes along `axis`, default last)

    Raises ValueError if any vector norm is < eps.
    """
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v, axis=axis, keepdims=True)
    if np.any(n < eps):
        raise ValueError(f"Cannot normalize vector(s) with norm < {eps}.")
    return v / n

def normalize(v: np.ndarray, *, axis: int = -1, eps: float = 1e-15) -> np.ndarray:
    """
    Normalize vectors to unit length.

    Works with shapes:
      - (3,)
      - (N, 3)
      - (..., 3) in general (normalizes along `axis`, default last).

    Parameters
    ----------
    v : array_like
        Input vector(s).
    axis : int
        Axis along which to compute the norm (default: -1).
    eps : float
        Small value to avoid division by zero.

    Returns
    -------
    out : np.ndarray
        Normalized vector(s) with same shape as input.

    Notes
    -----
    If a vector has norm < eps, it is left unchanged (effectively divided by eps),
    so the output magnitude may be ~0. If you prefer raising in that case, see below.
    """
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v, axis=axis, keepdims=True)
    return v / np.maximum(n, eps)