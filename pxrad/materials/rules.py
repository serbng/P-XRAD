# pxrad/materials/rules.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Callable, Mapping

import numpy as np

from pxrad.utils.types import IntArray, BoolArray

# For type checking
class ExtinctionRule(Protocol):
    """
    Systematic-absence / extinction rule for reflections.

    In a crystal, not every set of Miller indices (h, k, l) produces a
    diffraction spot: some reflections are *forbidden* by translational
    symmetries (e.g. centered lattices) and/or by the atomic basis.

    This callable returns a boolean mask with the same shape as the input
    arrays, indicating which reflections are allowed.

    Requirements
    ------------
    - Must be vectorized: h, k, l are arrays of the same shape.
    - Must return a boolean array of the same shape.
    """
    def __call__(self, h: IntArray, k: IntArray, l: IntArray) -> BoolArray: ...


@dataclass(frozen=True, slots=True)
class Rule:
    """
    Wrap a user-provided callable as an ExtinctionRule.

    This is the "escape hatch" for advanced users who want to specify their
    own selection rule.

    Example
    -------
    >>> rule = Rule(lambda h,k,l: ((h+k+l) % 2) == 0, name="I-centering")
    """
    func: Callable[[IntArray, IntArray, IntArray], BoolArray]
    name: str = "custom"

    def __call__(self, h: IntArray, k: IntArray, l: IntArray) -> BoolArray:
        h0 = np.asarray(h)
        k0 = np.asarray(k)
        l0 = np.asarray(l)

        if h0.shape != k0.shape or h0.shape != l0.shape:
            raise ValueError(
                f"ExtinctionRule {self.name!r} expects h,k,l with same shape, "
                f"got {h0.shape}, {k0.shape}, {l0.shape}."
            )

        m = self.func(h, k, l)
        m = np.asarray(m, dtype=bool)

        if m.shape != h0.shape:
            raise ValueError(
                f"ExtinctionRule {self.name!r} returned shape {m.shape}, expected {h0.shape}."
            )
        return m


# --- Basic lattice-centering rules -------------------------------------------

@dataclass(frozen=True, slots=True)
class AllowAll:
    """
    Primitive lattice (P): no systematic absences from centering.

    Note: this does NOT filter out (0,0,0). That is handled at the HKL
    generation stage.
    """
    name: str = "P (primitive)"

    def __call__(self, h: IntArray, k: IntArray, l: IntArray) -> BoolArray:
        return np.ones_like(np.asarray(h), dtype=bool)


@dataclass(frozen=True, slots=True)
class BodyCenteredI:
    """
    Body-centered lattice (I): allowed if h + k + l is even.

    Intuition
    ---------
    A body-centered lattice has an additional translation by (1/2, 1/2, 1/2).
    For some (h,k,l), waves scattered from the corner and body-center cancel.
    """
    name: str = "I (body-centered)"

    def __call__(self, h: IntArray, k: IntArray, l: IntArray) -> BoolArray:
        s = np.asarray(h) + np.asarray(k) + np.asarray(l)
        return (s % 2) == 0


@dataclass(frozen=True, slots=True)
class FaceCenteredF:
    """
    Face-centered lattice (F): allowed if h,k,l are all even OR all odd.

    Intuition
    ---------
    Face-centering adds translations like (1/2,1/2,0) etc., which cancels
    reflections unless the parity pattern of (h,k,l) matches the centering.
    """
    name: str = "F (face-centered)"

    def __call__(self, h: IntArray, k: IntArray, l: IntArray) -> BoolArray:
        h = np.asarray(h); k = np.asarray(k); l = np.asarray(l)
        he = (h % 2) == 0
        ke = (k % 2) == 0
        le = (l % 2) == 0
        all_even = he & ke & le
        all_odd  = (~he) & (~ke) & (~le)
        return all_even | all_odd


@dataclass(frozen=True, slots=True)
class ACentered:
    """
    A-centered lattice: allowed if k + l is even.
    (Centering translation is on the bc faces.)
    """
    name: str = "A-centered"

    def __call__(self, h: IntArray, k: IntArray, l: IntArray) -> BoolArray:
        return ((np.asarray(k) + np.asarray(l)) % 2) == 0


@dataclass(frozen=True, slots=True)
class BCentered:
    """
    B-centered lattice: allowed if h + l is even.
    (Centering translation is on the ac faces.)
    """
    name: str = "B-centered"

    def __call__(self, h: IntArray, k: IntArray, l: IntArray) -> BoolArray:
        return ((np.asarray(h) + np.asarray(l)) % 2) == 0


@dataclass(frozen=True, slots=True)
class CCentered:
    """
    C-centered lattice: allowed if h + k is even.
    (Centering translation is on the ab faces.)
    """
    name: str = "C-centered"

    def __call__(self, h: IntArray, k: IntArray, l: IntArray) -> BoolArray:
        return ((np.asarray(h) + np.asarray(k)) % 2) == 0


@dataclass(frozen=True, slots=True)
class RhombohedralR_hex:
    """
    Rhombohedral lattice, expressed in the *hexagonal indexing convention*.

    A common reflection condition in this setting is:
        -h + k + l = 3n

    If you don't know what this is, ignore it until you need trigonal/rhombohedral
    materials. It's here because it is a frequent "gotcha" in practice.
    """
    name: str = "R (hex setting)"

    def __call__(self, h: IntArray, k: IntArray, l: IntArray) -> BoolArray:
        t = (-np.asarray(h) + np.asarray(k) + np.asarray(l))
        return (t % 3) == 0


# --- Composite rules ---------------------------------------------------------

@dataclass(frozen=True, slots=True)
class AndRule:
    """
    Logical AND of multiple extinction rules.

    Useful when a material has, e.g., a centered lattice rule plus an additional
    basis-specific rule.
    """
    rules: tuple[ExtinctionRule, ...]
    name: str = "AND"

    def __call__(self, h: IntArray, k: IntArray, l: IntArray) -> BoolArray:
        if not self.rules:
            return np.ones_like(np.asarray(h), dtype=bool)
        m = np.ones_like(np.asarray(h), dtype=bool)
        for r in self.rules:
            m &= r(h, k, l)
        return m


# --- Common basis-specific rule: Diamond cubic -------------------------------

@dataclass(frozen=True, slots=True)
class Diamond:
    """
    Diamond cubic (Fd-3m) extinction rule (Si, Ge, diamond).

    Practical selection rule used in indexing:
    1) Must satisfy F-centering (all even or all odd).
    2) If all odd  -> allowed.
    3) If all even -> allowed only if (h + k + l) mod 4 == 0.

    This captures the additional (1/4,1/4,1/4)-type basis translation on top
    of the FCC centering.
    """
    name: str = "Diamond (Fd-3m)"

    def __call__(self, h: IntArray, k: IntArray, l: IntArray) -> BoolArray:
        h = np.asarray(h); k = np.asarray(k); l = np.asarray(l)

        he = (h % 2) == 0
        ke = (k % 2) == 0
        le = (l % 2) == 0

        all_even = he & ke & le
        all_odd  = (~he) & (~ke) & (~le)

        # F-centering condition:
        f = all_even | all_odd

        # Diamond basis condition on top of F:
        s4 = ((h + k + l) % 4) == 0
        return f & (all_odd | (all_even & s4))


# --- Convenience: choose rule by lattice centering symbol --------------------

_CENTERING_RULES: Mapping[str, ExtinctionRule] = {
    "P": AllowAll(),
    "I": BodyCenteredI(),
    "F": FaceCenteredF(),
    "A": ACentered(),
    "B": BCentered(),
    "C": CCentered(),
    "R": RhombohedralR_hex(),
}

def rule_from_centering(symbol: str) -> ExtinctionRule:
    """
    Build an extinction rule from a *lattice centering symbol*.

    What is a "centering symbol"?
    -----------------------------
    Many crystals share the same *translational symmetry* of the lattice.
    Crystallographers summarize this with a one-letter code:

      - "P": primitive (no extra lattice points)
      - "I": body-centered (extra point at the cell center)
      - "F": face-centered (extra points at the centers of all faces)
      - "A", "B", "C": base-centered (extra points on one pair of opposite faces)
      - "R": rhombohedral/trigonal lattice in hexagonal indexing (special case)

    These extra lattice points cause *systematic absences* (forbidden reflections)
    that depend only on (h,k,l) parity sums like "h+k+l even".

    Parameters
    ----------
    symbol:
        One of: "P", "I", "F", "A", "B", "C", "R" (case-insensitive).

    Returns
    -------
    ExtinctionRule:
        A callable rule (h,k,l) -> mask.

    Examples
    --------
    >>> rule = rule_from_centering("I")     # allowed if h+k+l is even
    >>> rule = rule_from_centering("F")     # allowed if all even or all odd
    >>> rule = rule_from_centering("P")     # allow all

    Notes
    -----
    This function covers only the lattice-centering part. Some materials
    have additional absences from the atomic basis (e.g. diamond cubic),
    which you should represent with a more specific rule like Diamond().
    """
    s = symbol.strip().upper()
    try:
        return _CENTERING_RULES[s]
    except KeyError as e:
        raise ValueError(
            f"Unknown centering symbol {symbol!r}. Expected one of: {', '.join(_CENTERING_RULES)}."
        ) from e

