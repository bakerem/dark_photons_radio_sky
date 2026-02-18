from epspy import meps
import os, sys
import argparse

sys.path.append("../")


import pysm3
import pysm3.units as u
import astropy.units as un
from tqdm import tqdm
from astropy.config import set_temp_cache, get_cache_dir_path

import healpy as hp

import h5py

import numpy as np

import pymaster as nmt

# import py3nj

from joblib import Parallel, delayed


# Load plot settings

parser = argparse.ArgumentParser()
# ---- required arguments ---- :
parser.add_argument("--output_dir", type=str, help="Path to output directory")

parser.add_argument("--nside", type=int, help="Nside parameter for the maps")

parser.add_argument(
    "--max_flux", type=float, help="Maximum flux for the point sources in Jy"
)

parser.add_argument(
    "--fstart", type=float, default=None, help="Starting frequency in MHz"
)

parser.add_argument(
    "--fstop", type=float, default=None, help="Stopping frequency in MHz"
)

parser.add_argument(
    "--n_freqs", type=int, default=None, help="Number of frequencies to generate"
)

parser.add_argument(
    "--njobs", type=int, default=-1, help="Number of jobs to run in parallel"
)

parser.add_argument(
    "--fwhm", type=float, default=0.5, help="FWHM of the Gaussian smoothing in arcmin"
)

# ---- optional arguments ----

parser.add_argument(
    "--config_file", type=str, default=None, help="Path to config file [default = None]"
)

parser.add_argument(
    "--generate_pt_srcs",
    action="store_true",
    help="Whether to generate point sources or not",
)

parser.add_argument(
    "--planck",
    action="store_true",
    help="Whether to use Planck frequencies and beams [default = False]",
)

parser.add_argument(
    "--ska",
    action="store_true",
    help="Whether to use SKA frequencies [default = False]",
)


args = parser.parse_args()
njobs = args.njobs


if args.ska and args.planck:
    raise ValueError("Cannot use both --ska and --planck flags at the same time.")  

if (args.planck or args.ska) and args.n_freqs is not None:
    raise ValueError("Can't specify --n_freqs when using --ska flag.")

output_dir = args.output_dir
if not os.path.exists(output_dir):
    os.makedirs(output_dir, exist_ok=True)

nside = args.nside
log2nside = int(np.log2(nside))

if args.planck:
    nu_centers = np.array(
        [
            30,
            44,
            70,
            100,
            143,
            217,
            353,
            545,
            # 857,
        ]
    )  # in GHz
    nu_centers = nu_centers * 1e3  # convert to MHz
elif args.ska:
    nu_centers = np.array([410, 560, 770, 1050, 1430, 4940, 6740, 9190, 12530])
else:
    nu_centers = np.linspace(args.fstart, args.fstop, args.n_freqs) # in MHz
# generate point source maps

if args.generate_pt_srcs:
    if os.path.exists(f"{output_dir}/Tb_nu_map{nside}.npy"):
        print(f"Point source maps already exist in {output_dir}Tb_nu_map{nside}.npy")
        point_source_maps = np.swapaxes(
            np.load(f"{output_dir}Tb_nu_map{nside}.npy"), 0, 1
        )
    else:
        print(f"Generating point source maps for nside {nside} in {output_dir}")
        obj = meps.eps(
            log2Nside=log2nside,
            logSmin=-6,
            dndS_form=0,
            logSmax=np.log10(args.max_flux),
            # nu_o = 140e6,
            path=output_dir,
            lbl=f"{nside}",
        )
        # meps.save_eps(obj, f"epspy_map_mK_21cm_nside{nside}.hdf5")
        obj.ref_freq()

        obj.gen_freq(nu=1e6 * nu_centers)
        point_source_maps = np.swapaxes(
            np.load(f"{output_dir}/Tb_nu_map{nside}.npy"), 0, 1
        )
else:
    point_source_maps = np.zeros((nu_centers.shape[0], hp.nside2npix(nside)))
    gal_map = hp.read_map("/projectnb/darkcosmo/dark_photon_project/21cmfast_cache/halo_data/correct_virial_mass/smooth_gal_map_K22.fits")
    mean_radio_brightnesses = np.mean(np.load("/projectnb/darkcosmo/dark_photon_project/21cmfast_cache/pyilc_Planck/Tb_nu_map2048.npy"), axis=0)
    rng = np.random.default_rng()
    spectral_indices = rng.normal(loc=2.681, scale=0.5, size=len(gal_map))
    corr_radio_map0 = gal_map * mean_radio_brightnesses[4] + mean_radio_brightnesses[4] # K_CMB
    point_source_maps = np.array([corr_radio_map0 * (f / nu_centers[4])**(-spectral_indices) for f in nu_centers])
    np.save(
        f"{output_dir}/nu_glob.npy", nu_centers * 1e6
    )  # Save nu_centers for later use


# generate everything else
full_ls = np.arange(3 * nside)

# all of this is in K_CMB
nu_centers = nu_centers * u.MHz
freefreesky = pysm3.Sky(nside=nside, preset_strings=["f1"])
freefree_maps = [
    freefreesky.get_emission(freq) for freq in nu_centers
]  # Get free-free maps for each frequency
freefree_maps = [
    freefree_maps[i].to(u.K_CMB, equivalencies=u.cmb_equivalencies(nu_centers[i]))[0, :]
    for i in range(len(nu_centers))
]  # Convert to mK_CMB

syncsky = pysm3.Sky(nside=nside, preset_strings=["s1"])
sync_maps = [
    syncsky.get_emission(freq) for freq in nu_centers
]  # Get free-free maps for each frequency
sync_maps = [
    sync_maps[i].to(u.K_CMB, equivalencies=u.cmb_equivalencies(nu_centers[i]))[0, :]
    for i in range(len(nu_centers))
]  # Convert to mK_CMB

cmbsky = pysm3.Sky(nside=nside, preset_strings=["c1"])
cmb_maps = [
    cmbsky.get_emission(freq) for freq in nu_centers
]  # Get cmb maps for each frequency
cmb_maps = np.array(
    [
        cmb_maps[i].to(u.K_CMB, equivalencies=u.cmb_equivalencies(nu_centers[i]))[0, :]
        for i in range(len(nu_centers))
    ]
) 

# if args.planck:
with set_temp_cache("/scratch/"):

    dustsky = pysm3.Sky(nside=nside, preset_strings=["d1"])
    dust_maps = [
        (
            dustsky.get_emission(freq)[0, :]
            if freq.to(u.GHz) > 1 * u.GHz
            else np.zeros(hp.nside2npix(nside)) * u.K_RJ
        )
        for freq in tqdm(nu_centers)
    ]
    dust_maps = [
        dust_maps[i].to(u.K_CMB, equivalencies=u.cmb_equivalencies(nu_centers[i]))
        for i in range(len(nu_centers))
    ]  # Convert to K_CMB

    amesky = pysm3.Sky(nside=nside, preset_strings=["a1"])
    ame_maps = [
        (
            amesky.get_emission(freq)[0, :]
            if freq.to(u.GHz) > 1 * u.GHz
            else np.zeros(hp.nside2npix(nside)) * u.K_RJ
        )
        for freq in tqdm(nu_centers)
    ]
    ame_maps = [
        ame_maps[i].to(u.K_CMB, equivalencies=u.cmb_equivalencies(nu_centers[i]))
        for i in range(len(nu_centers))
    ]  # Convert to mK_CMB

    tszsky = pysm3.Sky(nside=nside, preset_strings=["tsz1"])
    tsz_maps = [
        (
            tszsky.get_emission(freq)[0, :]
            if freq.to(u.GHz) > 1 * u.GHz
            else np.zeros(hp.nside2npix(nside)) * u.K_RJ
        )
        for freq in tqdm(nu_centers)
    ]
    tsz_maps = [
        tsz_maps[i].to(u.K_CMB, equivalencies=u.cmb_equivalencies(nu_centers[i]))
        for i in range(len(nu_centers))
    ]  # Convert to K_CMB

    cosky = pysm3.Sky(nside=nside, preset_strings=["co1"])
    co_maps = [
        (
            cosky.get_emission(freq)[0, :]
            if freq.to(u.GHz) > 1 * u.GHz
            else np.zeros(hp.nside2npix(nside)) * u.K_RJ
        )
        for freq in tqdm(nu_centers)
    ]
    co_maps = [
        co_maps[i].to(u.K_CMB, equivalencies=u.cmb_equivalencies(nu_centers[i]))
        for i in range(len(nu_centers))
    ]

foregrounds = np.sum(
        np.array(
            [
                point_source_maps,
                freefree_maps,
                sync_maps,
                cmb_maps,
                dust_maps,
                ame_maps,
                tsz_maps,
                co_maps,
            ]
        ),
        axis=0,
    )

if args.planck:
    beams = np.deg2rad(
        np.array([32.29, 27.94, 13.08, 9.66, 7.22, 4.90, 4.92, 4.67]) / 60
    )
else:
    # Convert FWHM from arcmin to radians: divide by 60 to get degrees, then use np.deg2rad
    beams = np.deg2rad(args.fwhm / 60) * np.ones(nu_centers.shape)

# Smooth and remove monopole
foregrounds = np.array(
    Parallel(n_jobs=njobs)(
        delayed(hp.smoothing)(foregrounds[i, :], fwhm=beams[i])
        for i in range(len(nu_centers))
    )
)
foregrounds = np.array(
    Parallel(n_jobs=njobs)(
        delayed(hp.remove_monopole)(foregrounds[i, :]) for i in range(len(nu_centers))
    )
)

for i, nu in enumerate(nu_centers):
    hp.write_map(
        f"{output_dir}/foregrounds_{nu.value/1000:.3f}_unwise_pt_src.fits", foregrounds[i]
    )  # saved in K since this is what's needed for pyilc


if args.planck:
    noise_ps_amps = np.sqrt(
        np.array(
            [
                0.00190,
                0.00222,
                0.00373,
                0.000507,
                9.21e-5,
                0.000185,
                0.00200,
                0.0551,
                30.9,
            ]
        )
        / (1e6) ** 2
    )  # K_CMB

    thermal_map = np.array(
        [
            hp.synfast(noise_ps_amps[i] ** 2 * np.ones(3 * nside), nside)
            for i in range(len(nu_centers))
        ]
    )
else:
    sigma_rms = 7.5 * un.uJy  # / (6 * un.arcsec)**2
    beam_area = hp.nside2pixarea(2048, degrees=True) * un.deg**2
    noises = []
    for nu in nu_centers:
        equiv = un.brightness_temperature(nu)
        noise = (7.5 * un.uJy / beam_area).to(un.K, equivalencies=equiv)
        noises.append(noise.value)
    thermal_map = np.array(
        [
            np.random.normal(scale=noises[i], size=(hp.nside2npix(nside)))
            for i in range(len(nu_centers))
        ]
    )

maps_w_thermal = foregrounds + thermal_map

np.save(
    f"{output_dir}/foregrounds_with_thermal_mK_unwise_pt_src.npy", maps_w_thermal * 1000
)  # save in mK

for i, nu in enumerate(nu_centers):
    hp.write_map(
        f"{output_dir}/foregrounds_nu{nu.value/1000:.3f}_thermal_unwise_pt_src.fits",
        maps_w_thermal[i],
        overwrite=True,
    )
