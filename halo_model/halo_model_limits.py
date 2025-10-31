"""
Author: Ethan Baker, Boston University
Github: github.com/bakerem
Compute dark photon constraints using halo model predictions.

This script computes auto- and cross-correlation limits for dark photon
coupling strength (epsilon) using NaMaster pseudo-Cl estimation on ILC cleaned
21cm maps and halo model galaxy catalogs. It performs likelihood-based
parameter estimation over a range of dark photon masses.

The script accepts command-line arguments for output directory, map names,
binning parameters, and halo model choice (K22 or K23).
"""

import os
import sys

sys.path.append("../")

import healpy as hp
import pymaster as nmt

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.optimize import minimize
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()
# ---- required arguments ---- :
parser.add_argument("output_dir", type=str, help="Path to output directory")

parser.add_argument("clean_map_name", type=str, help="Name of the clean map to use")

parser.add_argument("mask_file", type=str, help="Path to the mask")

parser.add_argument("model", type=str, choices=["K22", "K23"])

parser.add_argument("nside", type=int, help="Nside of the maps to use")

parser.add_argument(
    "bin_size", type=int, help="Size of the bins to use for the covariance matrix"
)

parser.add_argument("beam", type=float, help="FWHM of the beam in arcmin")

args = parser.parse_args()

output_dir = args.output_dir
os.makedirs(output_dir, exist_ok=True)

nside = args.nside

full_ls = np.arange(3 * nside)
clean_map = hp.read_map(f"{output_dir}/{args.clean_map_name}")
clean_map = np.where(np.isnan(clean_map), 0, clean_map)
apodized_mask = hp.read_map(args.mask_file)
binning = nmt.NmtBin.from_nside_linear(nside, args.bin_size)
beam = hp.gauss_beam(np.deg2rad(args.beam / 60), lmax=3 * nside - 1)


# load the clean power spectrum or generate it
clean_nmt_map = nmt.NmtField(mask=apodized_mask, maps=[clean_map], beam=beam, spin=0)
PPw = nmt.NmtWorkspace.from_fields(
    clean_nmt_map,
    clean_nmt_map,
    binning,
)

if not os.path.exists(f"{output_dir}/clean_auto_Cls_binning{args.bin_size}.npy"):
    clean_auto_Cls = nmt.compute_full_master(
        clean_nmt_map, clean_nmt_map, b=binning, workspace=PPw
    )[0]
    np.save(f"{output_dir}/clean_auto_Cls_binning{args.bin_size}.npy", clean_auto_Cls)
else:
    clean_auto_Cls = np.load(f"{output_dir}/clean_auto_Cls_binning{args.bin_size}.npy")
    # clean_nmt_map = np.load(f"{output_dir}/clean_nmt_map_binning{args.bin_size}.npy", allow_pickle=True)

# quit()
loaded_ls = np.concatenate(
    [
        [0],
        np.load(
            "/projectnb/darkcosmo/dark_photon_project/21cmfast_cache/halo_data/correct_virial_mass/ls.npy"
        ),
    ]
)
# load galaxy fields
gg_Cl = np.load(
    f"/projectnb/darkcosmo/dark_photon_project/21cmfast_cache/halo_data/correct_virial_mass/Cl_gg_model{args.model}.npy"
)
gg_Cl = np.concatenate([[0], gg_Cl])  # Add the l=0 mode
gg_Cl_interp = interp1d(loaded_ls, gg_Cl, bounds_error=False, fill_value=0)

if not os.path.exists(
    f"/projectnb/darkcosmo/dark_photon_project/21cmfast_cache/halo_data/correct_virial_mass/smooth_gal_map_{args.model}.fits"
):
    gal_map = hp.synfast(gg_Cl_interp(full_ls), nside, fwhm=np.deg2rad(0.5 / 60))
    gal_map = hp.remove_monopole(gal_map)
    hp.write_map(
        f"/projectnb/darkcosmo/dark_photon_project/21cmfast_cache/halo_data/correct_virial_mass/smooth_gal_map_{args.model}.fits",
        gal_map,
    )
else:
    gal_map = hp.read_map(
        f"/projectnb/darkcosmo/dark_photon_project/21cmfast_cache/halo_data/correct_virial_mass/smooth_gal_map_{args.model}.fits"
    )

# compute the covariance between the galaxy map and clean pyilc maps
gal_nmt_map = nmt.NmtField(mask=apodized_mask, maps=[gal_map], spin=0)
ggw = nmt.NmtWorkspace.from_fields(
    gal_nmt_map,
    gal_nmt_map,
    binning,
)

if not os.path.exists(
    f"/projectnb/darkcosmo/dark_photon_project/21cmfast_cache/halo_data/correct_virial_mass/gal_Cls_binning{args.bin_size}_{args.model}.npy"
):
    gal_Cls = nmt.compute_full_master(
        gal_nmt_map, gal_nmt_map, b=binning, workspace=ggw
    )[0]
    np.save(
        f"/projectnb/darkcosmo/dark_photon_project/21cmfast_cache/halo_data/correct_virial_mass/gal_Cls_binning{args.bin_size}_{args.model}.npy",
        gal_Cls,
    )
    # np.save(f"/projectnb/darkcosmo/dark_photon_project/21cmfast_cache/halo_data/correct_virial_mass/gal_nmt_map_binning{args.bin_size}_{args.model}.npy", gal_nmt_map)
else:
    gal_Cls = np.load(
        f"/projectnb/darkcosmo/dark_photon_project/21cmfast_cache/halo_data/correct_virial_mass/gal_Cls_binning{args.bin_size}_{args.model}.npy"
    )
    # gal_nmt_map = np.load(f"/projectnb/darkcosmo/dark_photon_project/21cmfast_cache/halo_data/correct_virial_mass/gal_nmt_map_binning{args.bin_size}_{args.model}.npy")

gPw = nmt.NmtWorkspace.from_fields(
    clean_nmt_map,
    gal_nmt_map,
    binning,
)
if not os.path.exists(
    f"{output_dir}/cross_Pg_Cls_binning{args.bin_size}_{args.model}.npy"
):
    cross_Pg_Cls = nmt.compute_full_master(
        clean_nmt_map, gal_nmt_map, b=binning, workspace=gPw
    )[0]

    np.save(
        f"{output_dir}/cross_Pg_Cls_binning{args.bin_size}_{args.model}.npy",
        cross_Pg_Cls,
    )
else:
    cross_Pg_Cls = np.load(
        f"{output_dir}/cross_Pg_Cls_binning{args.bin_size}_{args.model}.npy"
    )

    # generate covariance matrices between clean pyilc maps
if not os.path.exists(
    f"{output_dir}/covariance_clean_PP_binning{args.bin_size}_{args.model}.npy"
):
    PPcw = nmt.NmtCovarianceWorkspace.from_fields(
        clean_nmt_map, clean_nmt_map, clean_nmt_map, clean_nmt_map
    )
    coupled_PP_Cl = nmt.compute_coupled_cell(clean_nmt_map, clean_nmt_map) / np.mean(
        apodized_mask**2
    )
    PPcovar = nmt.gaussian_covariance(
        PPcw,
        0,
        0,
        0,
        0,
        coupled_PP_Cl,
        coupled_PP_Cl,
        coupled_PP_Cl,
        coupled_PP_Cl,
        PPw,
        wb=PPw,
    )

    np.save(
        f"{output_dir}/covariance_clean_PP_binning{args.bin_size}_{args.model}.npy",
        PPcovar,
    )
else:
    PPcovar = np.load(
        f"{output_dir}/covariance_clean_PP_binning{args.bin_size}_{args.model}.npy"
    )


if not os.path.exists(
    f"{output_dir}/covariance_clean_gP_binning{args.bin_size}_{args.model}.npy"
):
    gPcw = nmt.NmtCovarianceWorkspace.from_fields(
        clean_nmt_map, gal_nmt_map, clean_nmt_map, gal_nmt_map
    )
    coupled_Pg_Cl = nmt.compute_coupled_cell(clean_nmt_map, gal_nmt_map) / np.mean(
        apodized_mask**2
    )
    gPcovar = nmt.gaussian_covariance(
        gPcw,
        0,
        0,
        0,
        0,
        [gg_Cl_interp(full_ls)],
        coupled_Pg_Cl,
        coupled_Pg_Cl,
        coupled_PP_Cl,
        gPw,
        wb=gPw,
    )

    # save these--this only needs to be done once for each clean map
    np.save(
        f"{output_dir}/covariance_clean_gP_binning{args.bin_size}_{args.model}.npy",
        gPcovar,
    )
else:
    print("loading covariance matrices")
    gPcovar = np.load(
        f"{output_dir}/covariance_clean_gP_binning{args.bin_size}_{args.model}.npy"
    )


def compute_halo_limits(mA):
    """Compute dark photon coupling limits for a given mass using likelihood.

    This function loads pre-computed halo model power spectra for the given
    dark photon mass and performs a likelihood analysis comparing predicted
    signal to observed auto- and cross-correlation power spectra. It finds
    the maximum likelihood epsilon and computes 90% CL upper limits.

    Parameters
    ----------
    mA : float
        Dark photon mass in eV.

    Returns
    -------
    (auto_limit, cross_limit) : tuple of float
        Upper limits on epsilon from auto-correlation (epsilon^4 scaling)
        and cross-correlation (epsilon^2 scaling) analyses at 90% CL.

    Notes
    -----
    The likelihood is computed over multipoles in the range 150 < ell < 4096
    where the Limber approximation is valid. Uses Nelder-Mead minimization
    to find best-fit epsilon values and 90% confidence limits.
    """
    Tgamma0 = 2.73 * 1000  # mK
    pred_ls = np.geomspace(1, 7000, 50, dtype=int)
    pred_cross_Cls = -Tgamma0 * np.load(
        f"/projectnb/darkcosmo/dark_photon_project/21cmfast_cache/halo_data/correct_virial_mass/Cl_gP_mA{mA:.3e}_modelK22.npy"
    )
    pred_auto_Cls = Tgamma0**2 * np.load(
        f"/projectnb/darkcosmo/dark_photon_project/21cmfast_cache/halo_data/correct_virial_mass/Cl_PP_mA{mA:.3e}.npy",
    )  # / (2*np.pi)**2

    interp_pred_cross_Cls = interp1d(pred_ls, pred_cross_Cls, fill_value=0)
    interp_pred_auto_Cls = interp1d(pred_ls, pred_auto_Cls, fill_value=0)

    # only include ells above 200 where we can really be sure that the limber approximation is valid
    # lrange = (binning.get_effective_ells() > 200) & (binning.get_effective_ells() < 4000)
    lrange = (binning.get_effective_ells() > 150) & (
        binning.get_effective_ells() < 4096
    )

    @np.vectorize
    def cross_log_L(eps2):
        return (
            -0.5
            * (
                eps2 * interp_pred_cross_Cls(binning.get_effective_ells()[lrange])
                - cross_Pg_Cls[lrange]
            )
            @ np.linalg.inv(gPcovar[lrange][:, lrange])
            @ (
                eps2 * interp_pred_cross_Cls(binning.get_effective_ells()[lrange])
                - cross_Pg_Cls[lrange]
            )
        )

    @np.vectorize
    def auto_log_L(eps4):
        return (
            -0.5
            * (
                eps4 * interp_pred_auto_Cls(binning.get_effective_ells()[lrange])
                - clean_auto_Cls[lrange]
            )
            @ np.linalg.inv(PPcovar[lrange][:, lrange])
            @ (
                eps4 * interp_pred_auto_Cls(binning.get_effective_ells()[lrange])
                - clean_auto_Cls[lrange]
            )
        )

    max_auto_eps = minimize(
        lambda eps: -auto_log_L(eps**4),
        x0=0,
        options={"fatol": 0.001},
        method="Nelder-Mead",
    ).x
    max_cross_eps = minimize(
        lambda eps: -cross_log_L(eps**2),
        x0=0,
        options={"fatol": 0.001},
        method="Nelder-Mead",
    ).x

    def auto_lam(eps4):
        max_log_L = auto_log_L(max_auto_eps**4)
        if eps4 < max_auto_eps**4:
            return 1
        else:
            return np.exp(auto_log_L(eps4) - max_log_L)

    def cross_lam(eps2):
        max_log_L = cross_log_L(max_cross_eps**2)
        if eps2 < max_cross_eps**2:
            return 1
        else:
            return np.exp(cross_log_L(eps2) - max_log_L)

    # plt.loglog(eps_ary,-2 * np.log(np.vectorize(cross_lam)(eps_ary**2)))
    auto_lim = minimize(
        lambda eps: np.abs(2.71 + 2 * np.log(auto_lam(eps**4))),
        x0=max_auto_eps,
        options={"xatol": 1e-12},
        method="Nelder-Mead",
    ).x
    cross_lim = minimize(
        lambda eps: np.abs(2.71 + 2 * np.log(cross_lam(eps**2))),
        x0=1e-9,
        options={"xatol": 1e-12},
        method="Nelder-Mead",
    ).x

    # print(auto_lim, cross_lim)

    # print(f"Auto Limit is {auto_limit:.2e}. Cross Limit is {cross_limit:.2e} for mA = {mA:.3e}")
    return (
        auto_lim[0],
        cross_lim[0],
    )


def main():
    """Main execution: compute limits over mass range and save results."""
    results = []
    mA_list = np.geomspace(1e-13, 1e-11, 25)
    print("Covariance matrices computed. Computing limits...")
    for mA in tqdm(mA_list, desc="Computing limits"):
        auto_limit, cross_limit = compute_halo_limits(mA)
        results.append((mA, auto_limit, cross_limit))
    np.save(
        f"{args.output_dir}/limits_binning{args.bin_size}_{args.model}.npy", results
    )
    print(f"Results saved to {args.output_dir}/limits_binning{args.bin_size}_{args.model}.npy")


if __name__ == "__main__":
    main()

