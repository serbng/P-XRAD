from __future__ import annotations
import numpy as np
from pxrad.detectors.detector import Detector

_PRESETS: dict[str, Detector] = {
    "sCMOS": Detector(
            name="sCMOS", 
            shape=(2016, 2018), # (ny, nx)
            pixelsize=(73.4e-6, 73.4e-6),
            encoding="uint16",
            file_extension="tif"
        ),
    "EIGER_4M": Detector(
            name="EIGER_4M",
            shape=(2070, 2167), # (ny, nx)
            pixelsize=(75e-6, 75e-6),
            encoding="uint32",
            file_extension="h5"
        )
}

for k, d in _PRESETS.items():
    if k != d.name:
        raise RuntimeError(f"Preset key {k!r} != detector.name {d.name!r}")

def get_detector(name: str) -> Detector:
    try:
        return _PRESETS[name]
    except KeyError as e:
        raise KeyError(f"Unknown detector preset {name!r}. Available: {list(_PRESETS)}") from e

def list_detectors() -> list[str]:
    return sorted(_PRESETS.keys())