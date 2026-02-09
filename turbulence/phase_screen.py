"""
Kolmogorov / von-Karman phase screen generation via the Fourier (spectral) method
with optional sub-harmonic compensation.

Reference
---------
* Lane, Gleason & Zuber, "Simulation of a Kolmogorov phase screen,"
  Waves in Random Media 2, 209-224 (1992).
* Peters, Cocotos & Forbes, "Structured light in atmospheric turbulence,"
  Advances in Optics and Photonics 17(1), 113 (2025).
"""

import numpy as np
from .utils import make_grid, make_freq_grid


def _kolmogorov_psd(f_mag, r0):
    """Kolmogorov power spectral density of phase fluctuations.

    Phi(f) = 0.023 * r0^{-5/3} * |f|^{-11/3}
    """
    psd = np.zeros_like(f_mag)
    nonzero = f_mag > 0
    psd[nonzero] = 0.023 * r0 ** (-5.0 / 3.0) * f_mag[nonzero] ** (-11.0 / 3.0)
    return psd


def _von_karman_psd(f_mag, r0, L0=np.inf, l0=0.0):
    """Modified von-Karman power spectral density.

    Phi(f) = 0.023 * r0^{-5/3} * (|f|^2 + 1/L0^2)^{-11/6}
             * exp(-|f|^2 * l0^2)

    When L0 = inf and l0 = 0 this reduces to the Kolmogorov spectrum.
    """
    f2 = f_mag ** 2
    kappa0_sq = 0.0 if np.isinf(L0) else (1.0 / L0) ** 2
    psd = 0.023 * r0 ** (-5.0 / 3.0) * (f2 + kappa0_sq) ** (-11.0 / 6.0)
    if l0 > 0:
        psd *= np.exp(-f2 * l0 ** 2)
    # Zero the DC if kappa0 == 0 to avoid divergence
    psd[f_mag == 0] = 0.0
    return psd


def fourier_phase_screen(N, dx, r0, seed=None, numsub=3, L0=np.inf, l0=0.0):
    """Generate a single Kolmogorov / von-Karman phase screen using the
    Fourier method with sub-harmonic compensation.

    Parameters
    ----------
    N : int
        Grid size (N x N pixels).
    dx : float
        Pixel spacing in meters.
    r0 : float
        Fried parameter (coherence length) in meters.
    seed : int or None
        Random seed for reproducibility.
    numsub : int
        Number of sub-harmonic levels (0 = none, typically 3).
    L0 : float
        Outer scale of turbulence in meters (default inf = Kolmogorov).
    l0 : float
        Inner scale of turbulence in meters (default 0).

    Returns
    -------
    screen : ndarray, shape (N, N)
        Real-valued, zero-mean phase screen in radians.
    """
    rng = np.random.default_rng(seed)
    Lx = N * dx  # physical side length

    # ----- high-frequency component via FFT -----
    FX, FY = make_freq_grid(N, dx)
    f_mag = np.sqrt(FX ** 2 + FY ** 2)

    if np.isinf(L0) and l0 == 0:
        psd = _kolmogorov_psd(f_mag, r0)
    else:
        psd = _von_karman_psd(f_mag, r0, L0, l0)

    # Complex Gaussian noise weighted by sqrt(PSD)
    cn = (rng.standard_normal((N, N)) + 1j * rng.standard_normal((N, N)))
    cn *= np.sqrt(psd) / Lx  # factor 1/Lx accounts for discrete PSD scaling

    cn[N // 2, N // 2] = 0.0  # zero DC

    # To spatial domain
    screen = np.fft.ifft2(np.fft.ifftshift(cn)).real * (N * Lx)

    # ----- sub-harmonic compensation -----
    if numsub > 0:
        X, Y = make_grid(N, dx)
        df = 1.0 / Lx
        screen_sh = np.zeros((N, N))

        for b in range(1, numsub + 1):
            dfs = df / (3.0 ** b)
            fs_1d = np.array([-1, 0, 1]) * dfs
            FXs, FYs = np.meshgrid(fs_1d, fs_1d)
            f_mag_s = np.sqrt(FXs ** 2 + FYs ** 2)

            if np.isinf(L0) and l0 == 0:
                psd_s = _kolmogorov_psd(f_mag_s, r0)
            else:
                psd_s = _von_karman_psd(f_mag_s, r0, L0, l0)

            cn_s = (rng.standard_normal((3, 3)) + 1j * rng.standard_normal((3, 3)))
            cn_s *= np.sqrt(psd_s) / Lx * (1.0 / (3.0 ** b))
            cn_s[1, 1] = 0.0  # zero DC

            for ii in range(3):
                for jj in range(3):
                    screen_sh += (
                        cn_s[ii, jj]
                        * np.exp(1j * 2 * np.pi * (FXs[ii, jj] * X + FYs[ii, jj] * Y))
                    ).real

        screen = screen + screen_sh

    # Zero-mean
    screen -= screen.mean()
    return screen
