"""
Ground-to-air drone quantum communication channel simulation.

Simulates a Gaussian beacon beam propagating through atmospheric turbulence
using the split-step method (multi-layer phase screens + angular spectrum
diffraction), and computes dual-plane intensity images (pupil + focal) for
deep-learning-based wavefront sensing.

Physical model
--------------
* Transmitter  : Gaussian beam, lambda = 810 nm
* Channel      : Kolmogorov turbulence, split-step propagation
* Receiver     : Cassegrain telescope D = 260 mm, central obstruction 50 mm
* Coupling     : Equivalent focal length 1160 mm -> single-mode fiber (w0 = 2.5 um)

Outputs (per realisation)
-------------------------
* pupil_intensity : N x N  normalised intensity at the telescope aperture
* focal_intensity : N x N  normalised intensity at the focal / fiber-coupling plane
* phase_screen    : N x N  unwrapped turbulent phase at the aperture (radians)

Based on the SLTurbulence MATLAB library (Peters, Cocotos & Forbes, 2025).
"""

import numpy as np
from turbulence.phase_screen import fourier_phase_screen
from turbulence.propagation import angular_spectrum_prop
from turbulence.screen_count import number_of_screens, number_of_screens_from_cn2
from turbulence.utils import make_grid, make_freq_grid


# ===========================================================================
#  Physical constants & system parameters
# ===========================================================================
WAVELENGTH = 810e-9           # 810 nm (entangled photon wavelength)
K = 2 * np.pi / WAVELENGTH   # wavenumber

# --- Receiver (Cassegrain telescope) ---
D_PRIMARY = 260e-3            # effective clear aperture 260 mm
D_SECONDARY = 50e-3           # secondary mirror 50 mm  (obstruction ratio ~0.179)
EPSILON = D_SECONDARY / (280e-3)  # obstruction ratio w.r.t. physical primary 280 mm

# --- Fiber coupling optics ---
F_COUPLING = 116e-3           # fiber coupling lens focal length 116 mm
BEAM_DIAMETER_AFTER_COMPRESSION = 26e-3   # 260 mm / 10
F_EFF = F_COUPLING * D_PRIMARY / BEAM_DIAMETER_AFTER_COMPRESSION  # 1160 mm
W0_FIBER = 2.5e-6             # single-mode fiber mode-field radius (780-HP)


# ===========================================================================
#  Sampling design helpers
# ===========================================================================

def choose_grid_params(L, wavelength=WAVELENGTH, D_recv=D_PRIMARY):
    """Choose grid size N and pixel spacing dx for the simulation.

    Constraints
    -----------
    1. The receiver aperture D must be well sampled: D / dx >> 1.
    2. Angular-spectrum propagation requires dx^2 >= lambda * dz
       (Fresnel number condition) for each slab, or at least the grid
       should be wide enough so that the beam does not wrap around.
    3. The beam at the receiver (after diffraction) may be wider than D
       due to divergence, so the grid must be larger than D.

    Strategy: fix N = 512 (good balance of resolution vs. speed) and
    choose dx so that the physical grid side is >= 4 * D_recv, capped
    to satisfy sampling.  For very long paths the grid is enlarged.

    Parameters
    ----------
    L : float
        Total propagation distance (m).
    wavelength : float
        Wavelength (m).
    D_recv : float
        Receiver diameter (m).

    Returns
    -------
    N : int
        Grid size.
    dx : float
        Pixel spacing (m).
    """
    N = 512

    # The Gaussian beam half-angle divergence ~ lambda / (pi * w0_tx).
    # For a reasonable transmitter waist (e.g. 25 mm), the beam at distance L
    # will have a radius ~ w0 * sqrt(1 + (L / z_R)^2).
    # We size the grid to be at least 4x the receiver diameter OR large enough
    # to capture the diffracted beam.

    # Minimum grid side to cover the receiver with padding
    L_min = 4.0 * D_recv  # 1.04 m for D = 260 mm

    # Diffraction-limited beam spread (plane wave reference)
    # After propagation through distance L, a point source would spread to
    # ~ lambda * L / (N * dx), so we also want N * dx >= sqrt(lambda * L) * factor
    L_diff = max(L_min, 2.0 * np.sqrt(wavelength * L))

    # Use the larger of the two constraints
    L_grid = max(L_min, L_diff)

    dx = L_grid / N

    # Ensure dx is fine enough to resolve the aperture (at least 40 px across D)
    dx_max = D_recv / 40.0
    if dx > dx_max:
        dx = dx_max
        # May need to increase N to keep the grid large enough
        N_needed = int(np.ceil(L_grid / dx))
        # Round up to the next power of 2 for FFT efficiency
        N = int(2 ** np.ceil(np.log2(N_needed)))
        dx = L_grid / N

    return N, dx


def compute_beam_waist_at_transmitter(L, D_recv=D_PRIMARY, wavelength=WAVELENGTH):
    """Compute a reasonable Gaussian beam waist at the transmitter so that
    the beam roughly fills the receiver aperture at distance L.

    w(L) ~ w0 * sqrt(1 + (L / z_R)^2),  z_R = pi * w0^2 / lambda

    We want w(L) ~ D_recv / 2, solving for w0 gives a reasonable starting point.
    If L >> z_R this simplifies to w(L) ~ lambda * L / (pi * w0),
    so w0 ~ lambda * L / (pi * D_recv / 2).

    For short distances the waist is set to a minimum of 10 mm.
    """
    # Far-field approximation
    w0 = wavelength * L / (np.pi * D_recv / 2)
    w0 = max(w0, 10e-3)  # at least 10 mm waist
    w0 = min(w0, 50e-3)  # cap at 50 mm
    return w0


# ===========================================================================
#  Core simulation
# ===========================================================================

def gaussian_beam(X, Y, w0, wavelength=WAVELENGTH):
    """Initial Gaussian beam field at z = 0.

    U(rho) = exp(-rho^2 / w0^2)

    (unit peak amplitude, no curvature at the waist)
    """
    rho2 = X ** 2 + Y ** 2
    return np.exp(-rho2 / w0 ** 2)


def cassegrain_aperture(X, Y, D_outer=D_PRIMARY, D_inner=D_SECONDARY):
    """Annular aperture function for a Cassegrain telescope.

    W(rho) = 1  if D_inner/2 <= rho <= D_outer/2
             0  otherwise
    """
    rho = np.sqrt(X ** 2 + Y ** 2)
    W = np.zeros_like(rho)
    W[(rho >= D_inner / 2) & (rho <= D_outer / 2)] = 1.0
    return W


def propagate_through_turbulence(u0, N, dx, wavelength, L, cn2,
                                  criterion=0.1, numsub=3, seed=None,
                                  L0=np.inf, l0=0.0):
    """Split-step propagation through turbulent atmosphere.

    Parameters
    ----------
    u0 : ndarray (N, N)
        Initial complex field.
    N : int
        Grid size.
    dx : float
        Pixel spacing (m).
    wavelength : float
        Wavelength (m).
    L : float
        Total propagation distance (m).
    cn2 : float
        Refractive-index structure constant Cn^2 (m^{-2/3}).
    criterion : float
        Max Rytov variance per slab.
    numsub : int
        Sub-harmonic levels for phase screen generation.
    seed : int or None
        Base random seed.
    L0 : float
        Outer scale (m).
    l0 : float
        Inner scale (m).

    Returns
    -------
    u : ndarray (N, N)
        Complex field at the receiver plane.
    Ns : int
        Number of phase screens used.
    r0_total : float
        Fried parameter for the whole path.
    """
    k = 2 * np.pi / wavelength
    Ns_info = number_of_screens_from_cn2(cn2, k, L, criterion)
    Ns, r0_total, r0_per_screen, dz = Ns_info

    # Ensure at least one screen
    Ns = max(Ns, 1)
    dz = L / Ns

    # Per-screen r0 from the per-slab Cn2
    # r0_screen = (0.423 * cn2 * k^2 * dz)^{-3/5}
    r0_screen = (0.423 * cn2 * k ** 2 * dz) ** (-3.0 / 5.0)

    u = u0.copy().astype(np.complex128)
    rng_base = seed if seed is not None else 0

    for i in range(Ns):
        # Generate phase screen for this slab
        screen = fourier_phase_screen(
            N, dx, r0_screen, seed=rng_base + i, numsub=numsub, L0=L0, l0=l0
        )
        # Apply phase screen
        u = u * np.exp(1j * screen)
        # Propagate to next screen (or to the receiver)
        u = angular_spectrum_prop(u, dx, wavelength, dz)

    return u, Ns, r0_total


def simulate_single_realisation(
    L,
    cn2,
    N=None,
    dx=None,
    w0_tx=None,
    wavelength=WAVELENGTH,
    criterion=0.1,
    numsub=3,
    seed=None,
    L0=np.inf,
    l0=0.0,
):
    """Run one full simulation: transmitter -> turbulence -> receiver.

    Parameters
    ----------
    L : float
        Propagation distance (m).
    cn2 : float
        Cn^2 (m^{-2/3}).
    N : int or None
        Grid size (auto-selected if None).
    dx : float or None
        Pixel spacing (auto-selected if None).
    w0_tx : float or None
        Transmitter beam waist (auto-selected if None).
    wavelength : float
        Wavelength (m).
    criterion : float
        Max Rytov variance per slab.
    numsub : int
        Sub-harmonic levels.
    seed : int or None
        Random seed.
    L0, l0 : float
        Outer / inner turbulence scale (m).

    Returns
    -------
    result : dict with keys
        'pupil_intensity'  : (N, N) normalised pupil-plane intensity
        'focal_intensity'  : (N, N) normalised focal-plane intensity
        'phase'            : (N, N) unwrapped phase at the aperture (rad)
        'params'           : dict of simulation parameters
    """
    # --- Grid parameters ---
    if N is None or dx is None:
        N_auto, dx_auto = choose_grid_params(L, wavelength)
        N = N or N_auto
        dx = dx or dx_auto

    # --- Transmitter beam ---
    if w0_tx is None:
        w0_tx = compute_beam_waist_at_transmitter(L)

    X, Y = make_grid(N, dx)
    u0 = gaussian_beam(X, Y, w0_tx, wavelength)

    # --- Propagation through turbulence ---
    u_recv, Ns, r0_total = propagate_through_turbulence(
        u0, N, dx, wavelength, L, cn2,
        criterion=criterion, numsub=numsub, seed=seed, L0=L0, l0=l0,
    )

    # --- Receiver: Cassegrain aperture ---
    W = cassegrain_aperture(X, Y)
    u_pupil = W * u_recv

    # --- Pupil-plane intensity ---
    I_pupil = np.abs(u_pupil) ** 2

    # --- Phase at the aperture (within the annular region) ---
    phase_raw = np.angle(u_recv)  # [-pi, pi]
    # Unwrap phase (2-D unwrapping)
    try:
        from skimage.restoration import unwrap_phase as skimage_unwrap
        phase_unwrapped = skimage_unwrap(phase_raw)
    except ImportError:
        # Fallback: row-then-column 1-D unwrapping
        phase_unwrapped = np.unwrap(np.unwrap(phase_raw, axis=0), axis=1)

    # Mask phase to the aperture region
    phase_masked = phase_unwrapped * W

    # --- Focal-plane intensity via FFT (lens focusing) ---
    # Focal-plane pixel size: delta_x_focal = lambda * f_eff / (N * dx_pupil)
    u_focal = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(u_pupil)))
    I_focal = np.abs(u_focal) ** 2

    dx_focal = wavelength * F_EFF / (N * dx)

    # --- Normalise intensities to [0, 1] ---
    I_pupil_max = I_pupil.max()
    I_focal_max = I_focal.max()
    I_pupil_norm = I_pupil / I_pupil_max if I_pupil_max > 0 else I_pupil
    I_focal_norm = I_focal / I_focal_max if I_focal_max > 0 else I_focal

    # --- Fiber coupling efficiency (for reference) ---
    # Overlap integral with the fiber Gaussian mode
    r_focal = np.sqrt(X ** 2 + Y ** 2)  # reuse grid, but at focal scale
    # Actually need to build a focal-plane grid
    x_focal = (np.arange(N) - N / 2) * dx_focal
    Xf, Yf = np.meshgrid(x_focal, x_focal)
    fiber_mode = np.exp(-(Xf ** 2 + Yf ** 2) / W0_FIBER ** 2)
    # Normalise both fields
    u_focal_flat = u_focal.ravel()
    fiber_flat = fiber_mode.ravel()
    overlap = np.abs(np.sum(u_focal_flat * np.conj(fiber_flat))) ** 2
    norm_u = np.sum(np.abs(u_focal_flat) ** 2)
    norm_f = np.sum(np.abs(fiber_flat) ** 2)
    coupling_eff = overlap / (norm_u * norm_f) if (norm_u * norm_f) > 0 else 0.0

    # --- Rytov variance ---
    k = 2 * np.pi / wavelength
    rytov = 1.23 * cn2 * k ** (7.0 / 6.0) * L ** (11.0 / 6.0)

    params = {
        'L': L,
        'cn2': cn2,
        'N': N,
        'dx': dx,
        'dx_focal': dx_focal,
        'w0_tx': w0_tx,
        'wavelength': wavelength,
        'Ns': Ns,
        'r0': r0_total,
        'D_over_r0': D_PRIMARY / r0_total,
        'rytov': rytov,
        'f_eff': F_EFF,
        'coupling_efficiency': coupling_eff,
        'numsub': numsub,
        'criterion': criterion,
        'seed': seed,
    }

    return {
        'pupil_intensity': I_pupil_norm,
        'focal_intensity': I_focal_norm,
        'phase': phase_masked,
        'params': params,
    }


# ===========================================================================
#  Convenience: quick demo
# ===========================================================================

def main():
    """Run a single demonstration simulation and plot results."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    # --- Simulation parameters ---
    L = 5e3                # 5 km
    cn2 = 1e-15            # moderate turbulence

    print("=" * 60)
    print("Gaussian Beam Turbulence Simulation")
    print("=" * 60)

    result = simulate_single_realisation(L=L, cn2=cn2, seed=42)
    p = result['params']

    print(f"  Propagation distance : {p['L']/1e3:.1f} km")
    print(f"  Cn2                  : {p['cn2']:.2e} m^(-2/3)")
    print(f"  Rytov variance       : {p['rytov']:.4f}")
    print(f"  Fried parameter r0   : {p['r0']*100:.2f} cm")
    print(f"  D / r0               : {p['D_over_r0']:.2f}")
    print(f"  Number of screens    : {p['Ns']}")
    print(f"  Grid size            : {p['N']} x {p['N']}")
    print(f"  Pixel spacing (pupil): {p['dx']*1e3:.3f} mm")
    print(f"  Pixel spacing (focal): {p['dx_focal']*1e6:.3f} um")
    print(f"  Equiv. focal length  : {p['f_eff']*1e3:.1f} mm")
    print(f"  Coupling efficiency  : {p['coupling_efficiency']:.6f}")
    print("=" * 60)

    # --- Plot ---
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    im0 = axes[0].imshow(
        result['pupil_intensity'], cmap='hot',
        extent=[-p['N']*p['dx']/2*1e3, p['N']*p['dx']/2*1e3,
                -p['N']*p['dx']/2*1e3, p['N']*p['dx']/2*1e3],
    )
    axes[0].set_title('Pupil Intensity')
    axes[0].set_xlabel('x (mm)')
    axes[0].set_ylabel('y (mm)')
    plt.colorbar(im0, ax=axes[0], shrink=0.8)

    im1 = axes[1].imshow(
        result['focal_intensity'], cmap='hot',
        extent=[-p['N']*p['dx_focal']/2*1e6, p['N']*p['dx_focal']/2*1e6,
                -p['N']*p['dx_focal']/2*1e6, p['N']*p['dx_focal']/2*1e6],
    )
    axes[1].set_title('Focal Intensity')
    axes[1].set_xlabel('x (um)')
    axes[1].set_ylabel('y (um)')
    plt.colorbar(im1, ax=axes[1], shrink=0.8)

    im2 = axes[2].imshow(
        result['phase'], cmap='RdBu_r',
        extent=[-p['N']*p['dx']/2*1e3, p['N']*p['dx']/2*1e3,
                -p['N']*p['dx']/2*1e3, p['N']*p['dx']/2*1e3],
    )
    axes[2].set_title('Aperture Phase (unwrapped)')
    axes[2].set_xlabel('x (mm)')
    axes[2].set_ylabel('y (mm)')
    plt.colorbar(im2, ax=axes[2], shrink=0.8)

    plt.tight_layout()
    plt.savefig('simulation_demo.png', dpi=150)
    print("Figure saved to simulation_demo.png")


if __name__ == '__main__':
    main()
