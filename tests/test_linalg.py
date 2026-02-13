import numpy as np
import pytest

from pxrad.utils.linalg import vec2, vec3, ensure_finite, unit, normalize


def test_vec2_accepts_shape_2():
    v = vec2([1, 2])
    assert isinstance(v, np.ndarray)
    assert v.shape == (2,)
    assert v.dtype == float


def test_vec2_rejects_wrong_shape():
    with pytest.raises(ValueError):
        vec2([1, 2, 3])
    with pytest.raises(ValueError):
        vec2([[1, 2]])
    with pytest.raises(ValueError):
        vec2(np.zeros((2, 1)))


def test_vec3_accepts_shape_3():
    v = vec3([1, 2, 3])
    assert isinstance(v, np.ndarray)
    assert v.shape == (3,)
    assert v.dtype == float


def test_vec3_rejects_wrong_shape():
    with pytest.raises(ValueError):
        vec3([1, 2])
    with pytest.raises(ValueError):
        vec3([[1, 2, 3]])
    with pytest.raises(ValueError):
        vec3(np.zeros((3, 1)))


def test_ensure_finite_passes_finite():
    v = ensure_finite([1.0, 2.0, 3.0], name="v")
    assert np.all(np.isfinite(v))


def test_ensure_finite_raises_on_nan_inf():
    with pytest.raises(ValueError):
        ensure_finite([1.0, np.nan])
    with pytest.raises(ValueError):
        ensure_finite([1.0, np.inf])


def test_unit_single_vector_normalizes():
    v = unit([3.0, 0.0, 4.0])
    assert v.shape == (3,)
    assert np.isclose(np.linalg.norm(v), 1.0, atol=1e-14)


def test_unit_raises_on_near_zero():
    with pytest.raises(ValueError):
        unit([0.0, 0.0, 0.0])
    with pytest.raises(ValueError):
        unit([1e-300, 0.0, 0.0], eps=1e-15)


def test_unit_batch_vectors_normalizes_rows():
    # This assumes your unit() is batch-safe (N,3).
    # If your current unit() is only (3,), remove this test or switch to normalize().
    V = np.array([[3.0, 0.0, 4.0],
                  [0.0, 5.0, 0.0]], dtype=float)
    U = unit(V)
    assert U.shape == V.shape
    norms = np.linalg.norm(U, axis=-1)
    assert np.allclose(norms, 1.0, atol=1e-14)


def test_normalize_single_vector_is_unit_length():
    v = normalize([3.0, 0.0, 4.0])
    assert v.shape == (3,)
    assert np.isclose(np.linalg.norm(v), 1.0, atol=1e-14)


def test_normalize_batch_vectors_is_unit_length():
    V = np.array([[3.0, 0.0, 4.0],
                  [0.0, 5.0, 0.0]], dtype=float)
    U = normalize(V)
    assert U.shape == V.shape
    norms = np.linalg.norm(U, axis=-1)
    assert np.allclose(norms, 1.0, atol=1e-14)


def test_normalize_handles_near_zero_without_raising():
    V = np.array([[0.0, 0.0, 0.0],
                  [1.0, 0.0, 0.0]], dtype=float)
    U = normalize(V, eps=1e-12)
    assert U.shape == V.shape
    # the zero vector stays zero-ish (divided by eps), not NaN/Inf
    assert np.all(np.isfinite(U))
    assert np.allclose(U[0], 0.0)
    assert np.isclose(np.linalg.norm(U[1]), 1.0, atol=1e-14)