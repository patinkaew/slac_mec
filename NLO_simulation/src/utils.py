import numpy as np
from pint import UnitRegistry

def polar_to_rect(spectral_amplitude, spectral_phase):
    return np.multiply(spectral_amplitude, np.exp(1j*spectral_phase))

def rect_to_polar(complex_amplitude):
    amp = np.absolute(complex_amplitude)
    phase = np.angle(complex_amplitude)
    return amp, phase

def convert_unit(value, from_unit, to_unit):
    ureg = UnitRegistry()
    quantity = value * ureg(from_unit)
    return quantity.to(ureg(to_unit)).magnitude
