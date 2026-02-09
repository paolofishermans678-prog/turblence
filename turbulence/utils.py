"""
Utility functions for grid generation and common operations.
"""

import numpy as np


def make_grid(N, dx):
    """Create centered coordinate grids.

    Parameters
    ----------
    N : int
        Number of grid points along each axis.
    dx : float
        Grid spacing in meters.

    Returns
    -------
    X, Y : ndarray
        2-D coordinate arrays centered at zero.
    """
    x = (np.arange(N) - N / 2) * dx
    X, Y = np.meshgrid(x, x)
    return X, Y


def make_freq_grid(N, dx):
    """Create centered spatial-frequency grids.

    Parameters
    ----------
    N : int
        Number of grid points.
    dx : float
        Real-space grid spacing in meters.

    Returns
    -------
    FX, FY : ndarray
        2-D frequency grids (cycles / m).
    """
    df = 1.0 / (N * dx)
    f = (np.arange(N) - N / 2) * df
    FX, FY = np.meshgrid(f, f)
    return FX, FY
