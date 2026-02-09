"""
Determine the number of phase screens required for split-step propagation,
and compute the per-screen Fried parameter.

Follows the algorithm in NumberOfScreens.m from the SLTurbulence library.
"""

import numpy as np


def cn2_from_rytov(rytov, k, z):
    """Recover Cn^2 from the Rytov variance for a plane wave.

    sigma_R^2 = 1.23 * Cn^2 * k^{7/6} * z^{11/6}
    """
    return rytov / (1.23 * k ** (7.0 / 6.0) * z ** (11.0 / 6.0))


def r0_from_cn2(cn2, k, z):
    """Compute the Fried parameter from Cn^2.

    r0 = (0.423 * Cn^2 * k^2 * z)^{-3/5}
    """
    return (0.423 * cn2 * k ** 2 * z) ** (-3.0 / 5.0)


def number_of_screens(rytov, k, z, criterion=0.1):
    """Compute the required number of phase screens and per-screen r0.

    Parameters
    ----------
    rytov : float
        Target Rytov variance for the full path.
    k : float
        Wavenumber  2*pi/lambda.
    z : float
        Total propagation distance in meters.
    criterion : float
        Maximum allowable Rytov variance per slab (default 0.1).

    Returns
    -------
    Ns : int
        Number of phase screens.
    r0_total : float
        Fried parameter for the whole path (meters).
    r0_per_screen : float
        Fried parameter for each individual screen (meters).
    dz : float
        Propagation distance per slab (meters).
    cn2 : float
        Refractive index structure constant (m^{-2/3}).
    """
    cn2 = cn2_from_rytov(rytov, k, z)
    r0_total = r0_from_cn2(cn2, k, z)

    # Start with one screen and increase until per-slab Rytov < criterion
    Ns = 1
    while True:
        dz = z / Ns
        # Per-slab Rytov variance (plane wave, uniform Cn^2)
        step_rytov = 1.23 * cn2 * k ** (7.0 / 6.0) * dz ** (11.0 / 6.0)
        if step_rytov <= criterion:
            break
        Ns += 1
        if Ns > 500:
            break

    # Per-screen Fried parameter: each screen sees a shorter path
    # r0_step = (0.423 * cn2 * k^2 * dz)^{-3/5}
    r0_per_screen = r0_from_cn2(cn2, k, dz)

    return Ns, r0_total, r0_per_screen, dz, cn2


def number_of_screens_from_cn2(cn2, k, z, criterion=0.1):
    """Same as number_of_screens but starting from Cn^2 directly.

    Parameters
    ----------
    cn2 : float
        Refractive index structure constant (m^{-2/3}).
    k : float
        Wavenumber.
    z : float
        Total propagation distance in meters.
    criterion : float
        Maximum allowable Rytov variance per slab.

    Returns
    -------
    Ns, r0_total, r0_per_screen, dz
    """
    rytov = 1.23 * cn2 * k ** (7.0 / 6.0) * z ** (11.0 / 6.0)
    r0_total = r0_from_cn2(cn2, k, z)

    Ns = 1
    while True:
        dz = z / Ns
        step_rytov = 1.23 * cn2 * k ** (7.0 / 6.0) * dz ** (11.0 / 6.0)
        if step_rytov <= criterion:
            break
        Ns += 1
        if Ns > 500:
            break

    r0_per_screen = r0_from_cn2(cn2, k, dz)
    return Ns, r0_total, r0_per_screen, dz
