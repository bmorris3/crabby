from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import numpy as np
import batman
from copy import deepcopy
import astropy.units as u

__all__ = ['transit_model_e', 'transit_model_e_depth_t0', 'params_e']

# Planet b:
params_e = batman.TransitParams()
params_e.per = 0.736539
params_e.t0 = 2455733.013
params_e.inc = 83.3
params_e.rp = 0.0187

# a/rs = b/cosi
b = 0.41

eccentricity = 0 # np.sqrt(ecosw**2 + esinw**2)
omega = 90 # np.degrees(np.arctan2(esinw, ecosw))

ecc_factor = (np.sqrt(1 - eccentricity**2) /
              (1 + eccentricity * np.sin(np.radians(omega))))

params_e.a = b / np.cos(np.radians(params_e.inc)) / ecc_factor
params_e.ecc = eccentricity
params_e.w = omega
params_e.u = [0.6373, 0.0554]  # Morris+ 2017a
params_e.limb_dark = 'quadratic'

params_e.duration = 0.0658 * u.day


def transit_model_e(times, params=params_e):
    """
    Get a transit model for TRAPPIST-1b at ``times``.

    Parameters
    ----------
    times : `~numpy.ndarray`
        Times in Julian date
    params : `~batman.TransitParams`
        Transiting planet parameters

    Returns
    -------
    flux : `numpy.ndarray`
        Fluxes at each time
    """
    m = batman.TransitModel(params, times)
    model = m.light_curve(params)
    return model


def transit_model_e_depth_t0(times, depth, t0, f0=1.0):
    """
    Get a transit model for TRAPPIST-1b at ``times``.

    Parameters
    ----------
    times : `~numpy.ndarray`
        Times in Julian date
    depth : float
        Depth of transit
    t0 : float
        Mid-transit time [JD]
    Returns
    -------
    flux : `numpy.ndarray`
        Fluxes at each time
    """
    params = deepcopy(params_e)
    params.t0 = t0
    params.rp = np.sqrt(depth)
    m = batman.TransitModel(params, times)
    model = f0*m.light_curve(params)
    return model
