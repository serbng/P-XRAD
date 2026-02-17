from typing import Union
from numpy.typing import NDArray
from numpy import pi

HC_KEV_A: float = 12.398419843320026  # keV * Å

def energy_keV_to_wavelength_A(E_keV: Union[NDArray, float]) -> Union[NDArray, float]:
    """Convert photon energy in keV to wavelength in Å."""
    return HC_KEV_A / E_keV


def wavelength_A_to_energy_keV(lambda_A: Union[NDArray, float]) -> Union[NDArray, float]:
    """Convert photon wavelength in Å to energy in keV."""
    return HC_KEV_A / lambda_A

def deg2rad(x: float) -> float:
    return float(x) * pi / 180.0