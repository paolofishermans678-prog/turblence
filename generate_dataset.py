"""
Batch dataset generation for deep-learning wavefront sensing.

Generates a dataset of (pupil_intensity, focal_intensity, phase) triplets
across a range of propagation distances and turbulence strengths, suitable
for training a neural network to predict the wavefront phase from dual-plane
intensity measurements.

Output format
-------------
Each sample is stored as a compressed .npz file:
    input  : (N, N, 2)  -- channel 0 = pupil intensity, channel 1 = focal intensity
    label  : (N, N, 1)  -- unwrapped phase at the aperture

A metadata CSV file is also generated with per-sample simulation parameters.

Usage
-----
    python generate_dataset.py --output_dir ./dataset --num_samples 1000
"""

import argparse
import os
import csv
import time
import numpy as np
from simulate_gaussian_beam import simulate_single_realisation, WAVELENGTH, K


# ===========================================================================
#  HV turbulence profile (optional, for realistic Cn2 at different altitudes)
# ===========================================================================

def hufnagel_valley_cn2(h, v_rms=21.0, cn2_ground=1.7e-14):
    """Hufnagel-Valley (HV) model for Cn^2 as a function of altitude h.

    Parameters
    ----------
    h : float or array
        Altitude above ground in meters.
    v_rms : float
        RMS wind speed in the upper atmosphere (m/s), default 21 m/s.
    cn2_ground : float
        Ground-level Cn^2 (m^{-2/3}), default 1.7e-14.

    Returns
    -------
    cn2 : float or array
        Cn^2 at the given altitude(s).
    """
    h_km = np.asarray(h) / 1000.0  # convert to km for the standard HV formula
    h_m = np.asarray(h, dtype=float)
    term1 = 5.94e-53 * (v_rms / 27.0) ** 2 * h_m ** 10 * np.exp(-h_m / 1000.0)
    term2 = 2.7e-16 * np.exp(-h_m / 1500.0)
    term3 = cn2_ground * np.exp(-h_m / 100.0)
    return term1 + term2 + term3


def path_integrated_cn2(L, h_ground=0.0, elevation_deg=90.0, num_layers=100):
    """Compute the path-averaged Cn^2 for a slant path using the HV model.

    For a vertical path (elevation = 90 deg) of length L, this integrates
    the HV profile from h_ground to h_ground + L.

    For simplicity in the ground-to-drone scenario, we treat the path
    as having a uniform Cn^2 equal to the path average.

    Parameters
    ----------
    L : float
        Path length in meters.
    h_ground : float
        Ground station altitude ASL in meters.
    elevation_deg : float
        Elevation angle in degrees (90 = vertical).
    num_layers : int
        Number of integration layers.

    Returns
    -------
    cn2_avg : float
        Path-averaged Cn^2.
    """
    sin_el = np.sin(np.radians(elevation_deg))
    # Heights along the slant path
    s = np.linspace(0, L, num_layers)
    h = h_ground + s * sin_el
    cn2_profile = hufnagel_valley_cn2(h)
    cn2_avg = np.trapz(cn2_profile, s) / L
    return cn2_avg


# ===========================================================================
#  Dataset generation
# ===========================================================================

def generate_dataset(
    output_dir,
    num_samples=1000,
    L_range=(3e3, 30e3),
    cn2_range=(1e-16, 1e-13),
    N=256,
    wavelength=WAVELENGTH,
    criterion=0.1,
    numsub=3,
    use_hv_profile=False,
    elevation_deg=45.0,
    seed_base=0,
):
    """Generate a dataset of turbulence simulation samples.

    Parameters
    ----------
    output_dir : str
        Directory to save samples.
    num_samples : int
        Total number of samples.
    L_range : tuple (L_min, L_max)
        Propagation distance range in meters.
    cn2_range : tuple (cn2_min, cn2_max)
        Cn^2 range (used when use_hv_profile=False).
    N : int
        Grid size for all samples.
    wavelength : float
        Wavelength (m).
    criterion : float
        Max Rytov per slab.
    numsub : int
        Sub-harmonic levels.
    use_hv_profile : bool
        If True, compute Cn^2 from the HV profile for each distance.
    elevation_deg : float
        Elevation angle (used with HV profile).
    seed_base : int
        Starting random seed.
    """
    os.makedirs(output_dir, exist_ok=True)
    samples_dir = os.path.join(output_dir, 'samples')
    os.makedirs(samples_dir, exist_ok=True)

    rng = np.random.default_rng(seed_base)

    # CSV metadata
    csv_path = os.path.join(output_dir, 'metadata.csv')
    csv_fields = [
        'sample_id', 'L_m', 'cn2', 'rytov', 'r0_m', 'D_over_r0',
        'Ns', 'N', 'dx_m', 'dx_focal_m', 'w0_tx_m',
        'coupling_efficiency', 'seed',
    ]

    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=csv_fields)
        writer.writeheader()

        for i in range(num_samples):
            t0 = time.time()
            seed_i = seed_base + i

            # Random propagation distance (log-uniform in the range)
            log_L = rng.uniform(np.log10(L_range[0]), np.log10(L_range[1]))
            L = 10 ** log_L

            # Cn2
            if use_hv_profile:
                cn2 = path_integrated_cn2(L, elevation_deg=elevation_deg)
            else:
                log_cn2 = rng.uniform(np.log10(cn2_range[0]), np.log10(cn2_range[1]))
                cn2 = 10 ** log_cn2

            # Check Rytov variance -- skip extremely strong turbulence
            k = 2 * np.pi / wavelength
            rytov = 1.23 * cn2 * k ** (7.0 / 6.0) * L ** (11.0 / 6.0)

            # For the grid we need dx chosen per-sample for proper sampling
            # Use a fixed N but let dx adapt
            from simulate_gaussian_beam import choose_grid_params
            N_auto, dx_auto = choose_grid_params(L, wavelength)
            # Use the requested N, but keep dx from the auto calculation
            # Scale dx so the grid covers the same physical extent
            dx = dx_auto * N_auto / N

            try:
                result = simulate_single_realisation(
                    L=L, cn2=cn2, N=N, dx=dx,
                    wavelength=wavelength,
                    criterion=criterion, numsub=numsub,
                    seed=seed_i,
                )
            except Exception as e:
                print(f"  [SKIP] Sample {i}: {e}")
                continue

            p = result['params']

            # --- Save sample ---
            # Input: (N, N, 2)
            input_data = np.stack(
                [result['pupil_intensity'], result['focal_intensity']],
                axis=-1,
            ).astype(np.float32)

            # Label: (N, N, 1)
            label_data = result['phase'][:, :, np.newaxis].astype(np.float32)

            sample_path = os.path.join(samples_dir, f'sample_{i:06d}.npz')
            np.savez_compressed(sample_path, input=input_data, label=label_data)

            # --- Write metadata ---
            writer.writerow({
                'sample_id': i,
                'L_m': f'{L:.1f}',
                'cn2': f'{cn2:.4e}',
                'rytov': f'{p["rytov"]:.4f}',
                'r0_m': f'{p["r0"]:.6f}',
                'D_over_r0': f'{p["D_over_r0"]:.2f}',
                'Ns': p['Ns'],
                'N': N,
                'dx_m': f'{p["dx"]:.6e}',
                'dx_focal_m': f'{p["dx_focal"]:.6e}',
                'w0_tx_m': f'{p["w0_tx"]:.6e}',
                'coupling_efficiency': f'{p["coupling_efficiency"]:.6f}',
                'seed': seed_i,
            })

            dt = time.time() - t0
            if (i + 1) % 10 == 0 or i == 0:
                print(
                    f"  [{i+1}/{num_samples}]  L={L/1e3:.1f}km  "
                    f"Cn2={cn2:.2e}  Rytov={p['rytov']:.3f}  "
                    f"D/r0={p['D_over_r0']:.1f}  Ns={p['Ns']}  "
                    f"eta={p['coupling_efficiency']:.4f}  "
                    f"({dt:.1f}s)"
                )

    print(f"\nDataset saved to {output_dir}/")
    print(f"  Samples : {samples_dir}/sample_XXXXXX.npz")
    print(f"  Metadata: {csv_path}")


# ===========================================================================
#  CLI
# ===========================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description='Generate turbulence simulation dataset for DL wavefront sensing.',
    )
    parser.add_argument('--output_dir', type=str, default='./dataset',
                        help='Output directory')
    parser.add_argument('--num_samples', type=int, default=1000,
                        help='Number of samples to generate')
    parser.add_argument('--N', type=int, default=256,
                        help='Grid size (pixels)')
    parser.add_argument('--L_min', type=float, default=3e3,
                        help='Minimum propagation distance (m)')
    parser.add_argument('--L_max', type=float, default=30e3,
                        help='Maximum propagation distance (m)')
    parser.add_argument('--cn2_min', type=float, default=1e-16,
                        help='Minimum Cn2 (m^-2/3)')
    parser.add_argument('--cn2_max', type=float, default=1e-13,
                        help='Maximum Cn2 (m^-2/3)')
    parser.add_argument('--use_hv', action='store_true',
                        help='Use Hufnagel-Valley profile for Cn2')
    parser.add_argument('--elevation', type=float, default=45.0,
                        help='Elevation angle in degrees (for HV profile)')
    parser.add_argument('--criterion', type=float, default=0.1,
                        help='Max Rytov variance per phase screen slab')
    parser.add_argument('--numsub', type=int, default=3,
                        help='Number of sub-harmonic levels')
    parser.add_argument('--seed', type=int, default=0,
                        help='Base random seed')
    return parser.parse_args()


def main():
    args = parse_args()
    print("=" * 60)
    print("Turbulence Simulation Dataset Generator")
    print("=" * 60)
    print(f"  Output directory : {args.output_dir}")
    print(f"  Samples          : {args.num_samples}")
    print(f"  Grid size        : {args.N} x {args.N}")
    print(f"  Distance range   : {args.L_min/1e3:.0f} - {args.L_max/1e3:.0f} km")
    if args.use_hv:
        print(f"  Cn2 model        : Hufnagel-Valley (elev={args.elevation} deg)")
    else:
        print(f"  Cn2 range        : {args.cn2_min:.1e} - {args.cn2_max:.1e}")
    print(f"  Criterion        : {args.criterion}")
    print(f"  Sub-harmonics    : {args.numsub}")
    print(f"  Base seed        : {args.seed}")
    print("=" * 60)

    generate_dataset(
        output_dir=args.output_dir,
        num_samples=args.num_samples,
        L_range=(args.L_min, args.L_max),
        cn2_range=(args.cn2_min, args.cn2_max),
        N=args.N,
        criterion=args.criterion,
        numsub=args.numsub,
        use_hv_profile=args.use_hv,
        elevation_deg=args.elevation,
        seed_base=args.seed,
    )


if __name__ == '__main__':
    main()
