"""
Angular spectrum propagation of optical fields.

Implements the Fresnel transfer-function approach for paraxial beam propagation,
consistent with the AngularSpecProp.m function in the SLTurbulence library.
"""

import numpy as np


def angular_spectrum_prop(u1, dx, wavelength, z):
    """Propagate a 2-D complex field using the angular-spectrum (Fresnel TF) method.

    Parameters
    ----------
    u1 : ndarray, shape (N, N)
        Input complex optical field.
    dx : float
        Grid spacing in meters (same in x and y).
    wavelength : float
        Optical wavelength in meters.
    z : float
        Propagation distance in meters.

    Returns
    -------
    u2 : ndarray, shape (N, N)
        Propagated complex field.
    """
    N = u1.shape[0]
    Lx = N * dx
    df = 1.0 / Lx
    f = (np.arange(N) - N / 2) * df
    FX, FY = np.meshgrid(f, f)

    # Fresnel transfer function (angular spectrum propagator, paraxial)
    H = np.exp(-1j * np.pi * wavelength * z * (FX ** 2 + FY ** 2))

    # Propagate: FFT -> multiply by H -> IFFT
    U1 = np.fft.fft2(np.fft.fftshift(u1))
    U2 = U1 * np.fft.fftshift(H)
    u2 = np.fft.ifftshift(np.fft.ifft2(U2))
    return u2
