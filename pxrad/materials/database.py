from __future__ import annotations

from typing import Dict, List

from pxrad.lattice.lattice import Lattice
from pxrad.materials.material import Material
from pxrad.materials import rules


# -----------------------------------------------------------------------------
# Preset materials database
# -----------------------------------------------------------------------------
# Units:
# - lengths in Å
# - angles in degrees


_PRESETS: Dict[str, Material] = {
    # --- Elemental semiconductors (diamond cubic) ----------------------------
    "Si": Material(
        name="Si",
        lattice=Lattice(
            a=5.4309, 
            b=5.4309, 
            c=5.4309,
            alpha=90.0, 
            beta=90.0, 
            gamma=90.0
        ),
        rule=rules.Diamond(),
    ),
    
    "Ge": Material(
        name="Ge",
        lattice=Lattice(
            a=5.6575, 
            b=5.6575, 
            c=5.6575,
            alpha=90.0, 
            beta=90.0, 
            gamma=90.0
        ),
        rule=rules.Diamond(),
    ),

    "Al": Material(
        name="Al",
        lattice=Lattice(
            a=4.046, 
            b=4.046, 
            c=4.046,
            alpha=90.0, 
            beta=90.0, 
            gamma=90.0
        ),
        rule=rules.FaceCenteredF(),
    ),
    
    "Fe_alpha": Material(
        name="Fe (alpha, BCC)",
        lattice=Lattice(
            a=2.8665, 
            b=2.8665, 
            c=2.8665,
            alpha=90.0, 
            beta=90.0, 
            gamma=90.0
        ),
        rule=rules.BodyCenteredI(),
    ),

    # --- Silicon carbide polytypes -------------------------------------------
    # 3C-SiC (beta-SiC) is cubic (zinc blende / F-lattice). For peak positions,
    # F-centering is the key first-order selection rule.
    "SiC-3C": Material(
        name="SiC-3C",
        lattice=Lattice(
            a=4.3596, 
            b=4.3596, 
            c=4.3596,
            alpha=90.0, 
            beta=90.0, gamma=90.0
        ),
        rule=rules.FaceCenteredF(),
    ),

    # 4H-SiC is hexagonal (a=a, gamma=120). Systematic absences from the full
    # space group can be added later. For now, allow all (positions-only mode).
    "SiC-4H": Material(
        name="SiC-4H",
        lattice=Lattice(
            a=3.0730, 
            b=3.0730, 
            c=10.053,
            alpha=90.0, 
            beta=90.0, 
            gamma=120.0
        ),
        rule=rules.AllowAll(),
    ),
}

def get_material(name: str) -> Material:
    """
    Return a preset Material by name.

    Parameters
    ----------
    name:
        Preset key (case-insensitive). Examples: "Si", "Ge", "Al",
        "Fe_alpha", "3C-SiC", "4H-SiC".

    Returns
    -------
    Material
        A ready-to-use Material instance.
    """
    key = name.strip()
    if key in _PRESETS:
        return _PRESETS[key]

    # case-insensitive fallback
    key_upper = key.upper()
    for k in _PRESETS:
        if k.upper() == key_upper:
            return _PRESETS[k]

    raise KeyError(f"Unknown material preset {name!r}. Available: {list_materials()}")


def list_materials() -> List[str]:
    """List available material preset keys."""
    return sorted(_PRESETS.keys())


def register_material(key: str, material: Material, *, overwrite: bool = False) -> None:
    """
    Register a new preset material at runtime.

    This is convenient for notebooks / user scripts without requiring edits
    to the library source.

    Parameters
    ----------
    key:
        Name under which the material will be stored.
    material:
        The Material instance to register.
    overwrite:
        If False (default), raises if key already exists.
    """
    k = key.strip()
    if not k:
        raise ValueError("Material key must be a non-empty string.")
    if (k in _PRESETS) and not overwrite:
        raise KeyError(f"Material preset {k!r} already exists. Use overwrite=True to replace it.")
    _PRESETS[k] = material