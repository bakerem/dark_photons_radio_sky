"""
Author: Ethan Baker, Boston University
Description: This file stores the Survey class, which handles everything needed to interface with 
21cmFAST lightcones and galaxy surveys. This includes generating galaxy catalogs, computing dark photon signals,
and analyzing correlations.
A precomputed lightcone is required to initialize the class.
Github: https://github.com/bakerem
"""
import os
import sys
import subprocess

sys.path.append("../")

import numpy as np
import jax
import jax.numpy as jnp
from jax import jit, vmap, random

jax.config.update("jax_enable_x64", True)

from functools import partial

import py21cmfast as p21c
from astropy import units as un

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import colormaps as cms
import matplotlib.pylab as pylab
from plot_params import params

pylab.rcParams.update(params)

import physics as phys
import pickle

from astropy.cosmology import z_at_value
from astropy_healpix import HEALPix


import scipy.interpolate as interpolate
import scipy.integrate as integrate
from scipy import optimize
import pysm3
import healpy as hp

import h5py

from tqdm import tqdm
import pymaster as nmt

import itertools

import treecorr


seed = np.random.randint(100000, 10000000000)
key = random.key(seed)
rng = np.random.default_rng(seed)

@jit
def L_to_M(L):
    return 51.63 - 2.5 * jnp.log10(L)


@jit
def beta(MUV, z):
    # from https://arxiv.org/pdf/1508.01204 for z < 8 and https://arxiv.org/pdf/2311.06209 for z > 8
    M0 = -19.5
    c = -2.33
    z_big = jnp.tile(z[None, None, :], [MUV.shape[0], MUV.shape[1], 1])
    beta = jnp.where(z_big > 11, -0.15 * (MUV + 19) - 2.64, np.nan)
    beta = jnp.where(z_big > 10, -0.15 * (MUV + 19) - 2.45, beta)
    beta = jnp.where(z_big > 8, -0.15 * (MUV + 19) - 2.19, beta)
    beta = jnp.where(
        ((MUV >= M0) & (z_big < 8)),
        (betaM0(z_big) - c)
        * jnp.exp(-dbeta_dM0(z_big) * (MUV - M0) / (betaM0(z_big) - c))
        + c,
        beta,
    )
    beta = jnp.where(
        ((MUV < M0) & (z_big < 8)), dbeta_dM0(z_big) * (MUV - M0) + betaM0(z_big), beta
    )
    return beta


def betaM0(z):
    # from Table 3 of https://arxiv.org/pdf/1306.2950
    zs = jnp.array([2.5, 3.8, 5.0, 5.9, 7.0, 8.0])
    beta_M0s = jnp.array([-1.70, -1.85, -1.91, -2.00, -2.05, -2.13])
    return jnp.interp(z, zs, beta_M0s)


def dbeta_dM0(z):
    # from Table 3 of https://arxiv.org/pdf/1306.2950
    zs = jnp.array([2.5, 3.8, 5.0, 5.9, 7.0, 8.0])
    dbeta_dM0s = jnp.array([-0.20, -0.11, -0.14, -0.20, -0.20, -0.15])
    return jnp.interp(z, zs, dbeta_dM0s)


def compute_mAB(MUV, z, cosmo):
    """
    Calculate the apparent magnitude from the luminosity.
    # following https://arxiv.org/pdf/1508.01204
    """
    mu = 5 * jnp.log10(cosmo.luminosity_distance(np.array(z)).to(un.pc) / (10 * un.pc))
    z = jnp.array(z)
    sigma_beta = 0.34
    AUV = 4.43 + 1.99 * beta(MUV, z) + 0.79 * np.log(10) * sigma_beta**2
    AUV = jnp.where(
        AUV < 0, 0, AUV
    )  # set to 0 from model in  https://arxiv.org/pdf/1211.2825
    return MUV + mu + AUV - 2.5 * np.log10(1 + z)


def compute_obs_flux(L, z, cosmo):
    return (
        L * (1 + z) / (4 * np.pi * (cosmo.luminosity_distance(z) * 3.0857e24) ** 2)
    )  # convert Mpc to cm


def ai(nu):
    # nu is in MHz so let's convert to omega in units of eV
    # normalized to 150 MHz
    return 1 / (2 * jnp.pi * nu * 6.58212e-10)


class Survey:
    """
    A class to represent a galaxy survey, manage survey parameters, generate catalogs,
    compute dark photon signals, and perform correlation analyses between galaxy catalogs
    and foreground-cleaned maps.

    This class provides methods for loading lightcone data, generating and masking
    foregrounds, computing galaxy catalogs for various emission lines, simulating
    dark photon conversion signals, and performing auto- and cross-correlation analyses
    relevant for cosmological studies.

    Attributes and methods are tailored for surveys such as Subaru, Roman, and JWST,
    and support both analytic estimates of the dark photon signal and conversions that happen
    in the IGM during the 21cmFAST simulation.
    """

    def __init__(
        self,
        name: str,
        line: str,
        cache: str,
        lc_name: str,
        lightconer_name: str,
        nside: int,
        clobber: bool = False,
        clean_map_loc: str | None = None,
        foregrounds_path: str | None = None,
        use_pixel_ilc: bool = False,
        include_analytic: bool = False,
        analytic_path: str | None = None,
        regen_foregrounds: bool = False,
        foreground_gen_properties: dict | None = None,
    ):
        """
        Initialize the Survey object with the specified parameters.

        Parameters
        ----------
        name: str
            Name of the survey (e.g., "Subaru", "Roman", "JWST").
        line: str
            Emission line to compute the catalog for (e.g., "Lya", "Halpha", "OIII", "Hbeta").
        cache: str
            Path to the cache directory where the lightcone data is stored.
        lc_name: str
            Name of the lightcone file (without extension).
        lightconer_name: str
            Name of the lightconer file (without extension).
        nside: int
            Nside resolution parameter for the foreground maps.
        clobber: bool, optional
            If True, will overwrite existing files in the cache directory. Default is False.
        clean_map_loc: str | None, optional
            Path to an already cleaned map file. If None, must use pixel ilc.
        use_pixel_ilc: bool, optional
            If True, will use pixel-based internal linear combination (ILC) for the foregrounds.
            If False, will use the provided clean map. Default is False. Cannot provide both `clean_map_loc` and `use_pixel_ilc`.
        include_analytic: bool, optional
            If True, will include analytic dark photon signal that has already been generated for the autocorrelation. Default is False.
        analytic_path: str | None, optional
            Path to the analytic dark photon signal file. Required if `include_analytic` is True.
        regen_foregrounds: bool, optional
            If True, will regenerate the foreground maps. Default is False.
        foreground_gen_properties: dict | None, optional
            Properties for generating foreground maps. If None, will use default properties.
            Default properties include:
                - max_flux: 1e-3 # maximum flux in Jy
                - fstart: 115 # start frequency in MHz
                - fstop: 117 # stop frequency in MHz
                - n_freqs: 20 # number of frequency channels
                - gen_pt_srcs: False # whether to generate point sources
                - FWHM: 0.5 # FWHM of the beam in degrees
        """
        survey_mag_cuts = {
            "Subaru": 26,
            "Roman": 26.7,
            "JWST": 28.2,
        }

        # survey_flux_lims = {
        #     "Subaru": 2e-18,
        #     "Roman": 8.5e-17,
        #     "JWST": 2e-18,
        # }  # all in erg/s/cm^2

        survey_z_cuts = {
            "Subaru": [6.59, 6.61],
            "Roman": [7, np.inf],
            "JWST": [6, np.inf],
        }

        # survey centers are approximate but give us an idea of where to compute noise estimates
        # centers are given in galactic coordinates in degrees
        survey_centers = {
            "Subaru": [236.8222342, 42.1216283],
            "Roman": [260, -70],
            "JWST": [223.564, -54.425],
        }
        survey_area = {
            "Subaru": 16,
            "Roman": 500,
            "JWST": 0.01722,
        }
        self.name = name
        self.line = line
        if "LightCone_" in lc_name and lc_name.endswith(".h5"):
            self.prefix = lc_name.split("LightCone_")[-1].split(".h5")[0]
        else:
            raise ValueError(
                f"lc_name '{lc_name}' does not match expected format 'LightCone_... .h5'"
            )
        self.clobber = clobber
        self.nside = nside
        self.nside = nside
        self.cache = cache
        self.use_pixel_ilc = use_pixel_ilc
        self.include_analytic = include_analytic
        self.analytic_path = analytic_path
        self.clean_map_loc = clean_map_loc
        self.foregrounds_path = foregrounds_path

        self.regen_foregrounds = regen_foregrounds
        self.foreground_gen_properties = foreground_gen_properties
        if self.foreground_gen_properties is None:
            self.foreground_gen_properties = {
                "max_flux": 1e-3,
                "fstart": 115,
                "fstop": 117,
                "n_freqs": 20,
                "gen_pt_srcs": False,
                "FWHM": 0.5,
            }

        if self.include_analytic and analytic_path is None:
            raise ValueError(
                "Need to provide analytic path if include_analytic is True."
            )

        if self.use_pixel_ilc and self.clean_map_loc:
            raise ValueError(
                "Wanted to do pixel ilc and provided a clean map. Choose one or the other."
            )

        if not self.use_pixel_ilc and self.clean_map_loc is None:
            raise ValueError(
                "You must either provide a path to a clean map (clean_map_loc) or set use_pixel_ilc=True to use pixel-based ILC for foreground cleaning."
            )

        if name not in ["Subaru", "Roman", "JWST"]:
            raise NotImplementedError(
                f"Survey {name} is not implemented yet. Available surveys: ['Subaru', 'Roman', 'JWST']"
            )

        if name not in survey_mag_cuts.keys():
            raise ValueError(
                f"Survey {name} not recognized. Available surveys: {list(survey_mag_cuts.keys())}"
            )
        self.survey_mag_cut = survey_mag_cuts[name]
        # self.survey_flux_lim = survey_flux_lims[name]
        self.survey_z_min = survey_z_cuts[name][0]
        self.survey_z_max = survey_z_cuts[name][1]
        self.survey_center = survey_centers[name]
        self.survey_area = survey_area[name]

        if not hasattr(self, "cosmo"):
            self.load_lightcone(cache + lc_name, cache + lightconer_name)

        self.Tgamma0 = 2.73e3  # mK

    def load_lightcone(self, lc_path: str, lightconer_path: str):
        """
        Load the lightcone data from the specified file and initialize properties
        based on the lightcone parameters.

        Parameters
        ----------
        lc_path: str
            Path to the lightcone file (HDF5 format).
        lightconer_path: str
            Path to the lightconer file (pickle format).

        Returns
        -------
        None
        """
        lc_file = h5py.File(lc_path, "r")
        inputs = p21c.io.h5.read_inputs(lc_file["InputParameters"], safe=False)
        lc = lc_file["lightcones"]

        # rest of the code that uses lc_file remains unchanged

        self.z_start = jnp.max(np.array(inputs.node_redshifts))
        self.z_end = jnp.min(np.array(inputs.node_redshifts))
        cosmo_params = inputs.cosmo_params
        self.cosmo = cosmo_params.cosmo
        with open(lightconer_path, "rb") as f:
            ang_lcn = pickle.load(f)

        self.box_len = inputs.simulation_options.BOX_LEN
        self.hii_dim = inputs.simulation_options.HII_DIM
        self.dim = inputs.simulation_options.DIM
        self.lon = ang_lcn.longitude.reshape((self.dim, self.dim))
        self.lat = ang_lcn.latitude.reshape((self.dim, self.dim))[::-1]

        self.box_size_radians = (
            self.box_len / self.cosmo.comoving_distance(self.z_end).value
        )
        self.rs_dim = jnp.array(lc["brightness_temp"]).shape[-1]
        self.bt = jnp.flip(
            jnp.array(lc["brightness_temp"]).reshape((self.dim, self.dim, self.rs_dim)),
            axis=2,
        )
        self.hsfr = (
            jnp.flip(
                jnp.array(lc["halo_sfr"], dtype=np.float64).reshape(
                    (self.dim, self.dim, self.rs_dim)
                ),
                axis=2,
            )
        ) * 1e10  # last factor for unit conversion, 21cmFAST normalizes masses by 10^10 Msun

        self.xHI = jnp.flip(
            jnp.array(lc["neutral_fraction"]).reshape(
                (self.dim, self.dim, self.rs_dim)
            ),
            axis=2,
        )
        self.Mh = jnp.flip(
            jnp.array(lc["halo_mass"]).reshape((self.dim, self.dim, self.rs_dim)),
            axis=2,
        )

        H, D = jnp.meshgrid(self.lat[:, 0], ang_lcn.lc_distances.value)

        rgrid = jnp.arange(0, D.max(), self.box_len)
        rgrid = rgrid[rgrid > D.min()]
        rgrid = jnp.append(rgrid, D.max())
        self.rs_array = jnp.flipud(
            jnp.array(
                [
                    z_at_value(self.cosmo.comoving_distance, d * un.Mpc).value
                    for d in ang_lcn.lc_distances.value
                ]
            )
        )

        xe_box = jnp.flip(
            jnp.array(lc["xray_ionised_fraction"]).reshape(
                (self.dim, self.dim, self.rs_dim)
            ),
            axis=2,
        )
        self.xe_box = xe_box
        density_box = jnp.flip(
            jnp.array(lc["density"]).reshape((self.dim, self.dim, self.rs_dim)),
            axis=2,
        )

        n_e = (
            xe_box
            * phys.nB
            * (1 + self.rs_array) ** 3
            * (1 + density_box)
            * (1 - 3 * phys.YHe / 4)
        )

        # calculate plasma mass and related quantities that are relevant for conversions
        self.mgamma = phys.m_gamma(n_e)
        self.dlogmgamma2dz = jnp.abs(
            jnp.diff(jnp.log(self.mgamma**2), axis=2, prepend=0)
            / jnp.diff(self.rs_array, prepend=1)
        )
        # clean up memory
        del xe_box, density_box, n_e, cosmo_params, H, D, rgrid, inputs
        lc_file.close()
        return

    def generate_foregrounds(
        self,
    ):
        """
        Generate or load the foregrounds for the survey.
        This function generates the foregrounds if `regen_foregrounds` is True.
        Otherwise, they are loaded from the specified path.
        The foregrounds are generated using the `generate_foregrounds.py` script
        and saved in the `foregrounds_path` directory.

        Parameters
        ----------
        None

        Returns
        -------
        foregrounds: jnp.ndarray
            Foreground maps for the survey, with shape (n_freqs, dim, dim).
        nu_centers: jnp.ndarray
            Frequency centers for the foregrounds, in MHz.
        """
        self.full_ls = jnp.arange(3 * self.nside)
        if self.regen_foregrounds:
            print(f"Generating new foregrounds at {self.foregrounds_path}")
            print(f"Foreground properties are {self.foreground_gen_properties}")
            os.makedirs(f"{self.foregrounds_path}", exist_ok=True)
            subprocess_cmd = [
                "python",
                "../ilc/generate_foregrounds.py",
                f"--output_dir={self.foregrounds_path}/",
                f"--nside={self.nside}",
                f"--max_flux={self.foreground_gen_properties['max_flux']}",
                f"--fstart={self.foreground_gen_properties['fstart']}",
                f"--fstop={self.foreground_gen_properties['fstop']}",
                f"--n_freqs={self.foreground_gen_properties['n_freqs']}",
                "--njobs=4",
                f"--fwhm={self.foreground_gen_properties['FWHM']}",
            ]
            if self.foreground_gen_properties["gen_pt_srcs"]:
                subprocess_cmd.append("--generate_pt_srcs")
            subprocess.run(subprocess_cmd)
            print("Foreground generation complete. Now loading foregrounds.")
            foregrounds = np.load(
                f"{self.foregrounds_path}/foregrounds_with_thermal_mK.npy"
            )

        else:
            foregrounds = np.load(
                f"{self.foregrounds_path}/foregrounds_with_thermal_mK.npy"
            )

        self.nu_centers = np.load(f"{self.foregrounds_path}/nu_glob.npy") / 1e6 * un.MHz

        # mask out the entire sky except for the survey area
        vec = hp.ang2vec(self.survey_center[0], self.survey_center[1], lonlat=True)
        if self.survey_area < 25:
            self.radius = jnp.radians(
                jnp.sqrt(self.survey_area / jnp.pi)
            )  # Convert area to radius in radians
        else:
            self.radius = jnp.radians(25)
        self.disk = hp.query_disc(self.nside, vec, self.radius)  #
        self.bool_map = jnp.ones(hp.nside2npix(self.nside), dtype=bool)
        self.bool_map = self.bool_map.at[self.disk].set(False)
        self.fsky = self.disk.shape[0] / hp.nside2npix(self.nside)

        self.foregrounds = jnp.array(
            [
                self.convert_from_healpy(foregrounds[i] * ~self.bool_map)
                for i in range(len(self.nu_centers))
            ]
        )

        del foregrounds

        return self.foregrounds, self.nu_centers

    def compute_catalog(self):
        """
        Compute the catalog for the specified emission line.

        Parameters
        ----------
        None

        Returns
        -------
        flux: jnp.ndarray
            Flux map for the specified emission line, with shape (dim, dim, rs_dim).

        gal_map: jnp.ndarray
            Galaxy map for the specified emission line, with shape (dim, dim).
            The values in the map are counts of galaxies that pass the survey cuts.
        """
        # Compute mAB for each galaxy
        if os.path.exists(
            f"{self.cache}/{self.prefix}_{self.name}_{self.line}_gal_map.npy"
        ):
            self.gal_cat_3d = jnp.load(
                f"{self.cache}/{self.prefix}_{self.name}_{self.line}_gal_cat_3d.npy"
            )
            self.gal_map = jnp.load(
                f"{self.cache}/{self.prefix}_{self.name}_{self.line}_gal_map.npy"
            )
            return self.gal_map

        mAB = compute_mAB(
            L_to_M(self.hsfr / 1.15e-28), np.array(self.rs_array), self.cosmo
        )
        # Apply survey cuts
        rs_min_index = jnp.argmin(jnp.abs(self.rs_array - self.survey_z_min))
        rs_max_index = jnp.argmin(jnp.abs(self.rs_array - self.survey_z_max))
        self.gal_cat_3d = np.where(mAB < self.survey_mag_cut, mAB, 0)
        self.gal_cat_3d = jnp.array(self.gal_cat_3d)

        self.gal_cat_3d = self.gal_cat_3d.at[..., :rs_max_index].set(0)
        self.gal_cat_3d = self.gal_cat_3d.at[..., rs_min_index:].set(0)
        self.gal_map = jnp.sum(
            jnp.where(self.gal_cat_3d != 0, 1, 0), axis=-1
        )  # convert from flux to counts
        jnp.save(
            f"{self.cache}/{self.prefix}_{self.name}_{self.line}_gal_cat_3d.npy",
            self.gal_cat_3d,
        )
        jnp.save(
            f"{self.cache}/{self.prefix}_{self.name}_{self.line}_gal_map.npy",
            self.gal_map,
        )

        return self.gal_map

    def compute_dp_signal(self, mA: float | None = None) -> np.ndarray:
        """
        Compute the dark photon signal for a given mass mA.

        Parameters
        ----------
        mA: float
            Mass of the dark photon in eV.

        Returns
        -------
        Ptot: np.ndarray
            Total conversion map for the dark photon signal at mass mA with units of eV.
            To get the conversion probability, multiply by epsilon^2 omega^{-1}.
        """

        if mA is not None:
            self.mA = mA

        if self.mA is None and mA is None:
            raise ValueError("Please provide a value for mA (dark photon mass in eV).")

        mA_string = f"{self.mA:.3e}"

        save_loc = f"{self.cache}/{self.prefix}_conversion_map_mA{mA_string}.npy"
        # if os.path.exists(save_loc) and not self.clobber:
        #     self.Ptot = jnp.load(save_loc)
        #     return self.Ptot

        # if not saved, recompute

        num_crossings = jnp.where(jnp.diff(jnp.sign(self.mgamma - self.mA)), 1, 0)
        num_crossings = num_crossings.at[..., 0].set(0)
        num_crossings = jnp.concatenate(
            [jnp.zeros((self.dim, self.dim, 1)), num_crossings], axis=-1
        )
        Pgamma_i = (
            jnp.pi
            * self.mA**2
            / (
                phys.hubble(1 + self.rs_array[None, None, :])
                * 6.57895e-16
                * (1 + self.rs_array[None, None, :]) ** 2
            )
            * self.dlogmgamma2dz ** (-1)
            * num_crossings
        )

        self.Ptot = jnp.sum(Pgamma_i, axis=2)
        del num_crossings, Pgamma_i

        jnp.save(
            f"{self.cache}/{self.prefix}_conversion_map_mA{mA_string}.npy", self.Ptot
        )

        return self.Ptot

    def get_bt_plus_dp(
        self, mA: float | None = None, epsilon: float | None = None
    ) -> np.ndarray:
        """
        Get the brightness temperature map with the dark photon signal added.
        Parameters
        ----------
        mA: float, optional
            Mass of the dark photon in eV. If not provided, will use the previously set value

        epsilon: float, optional
            Coupling parameter for the dark photon. If not provided, will use the previously set value

        Returns
        -------
        bt_plus_dp: np.ndarray
            Brightness temperature map with the dark photon signal added.

        """
        if mA is not None:
            self.mA = mA

        if epsilon is not None:
            self.epsilon = epsilon

        if self.mA is None and mA is None:
            raise ValueError("Please provide a value for mA (dark photon mass in eV).")

        if self.epsilon is None and epsilon is None:
            raise ValueError(
                "Please provide a value for epsilon (dark photon mixing parameter)."
            )

        if not hasattr(self, "Ptot"):
            self.compute_dp_signal()

        # Ptot doesn't include frequency information, so we need to divide by the observed frequency at each z
        obs_w = 2 * jnp.pi * 9.34924e-7 / (1 + self.rs_array)  # 21cm frequency in eV
        signal = jnp.tile(self.Ptot[..., None], (1, 1, self.rs_array.shape[0])) / (
            obs_w[None, None, :]
        )
        self.dp_lightcone = self.Tgamma0 * self.epsilon**2 * signal
        self.bt_w_dp = self.bt - self.Tgamma0 * self.epsilon**2 * signal
        del signal

        return self.bt_w_dp

    def pixel_ilc(self) -> np.ndarray:
        """
        Perform pixel-based internal linear combination (ILC) for the dark photon conversion map.
        This method computes the cleaned foreground map while maintaining a dark-photon like signal that
        scales as $\omega^{-1}$.

        Parameters
        ----------
        None

        Returns
        -------
        clean_null: jnp.ndarray
            Cleaned foreground map with shape (dim, dim).
        """

        self.generate_foregrounds()

        map_means = jnp.nanmean(self.foregrounds, axis=(1, 2))

        Y = self.foregrounds - map_means[:, None, None]
        R = jnp.nanmean(jnp.einsum("i..., j...->ij...", Y, Y), axis=(3, 4))
        inv_cov = jnp.linalg.inv(R)
        cov_inv_prod = R @ inv_cov

        # Check if cov_inv_prod is close to identity for each matrix, otherwise the covariance matrix was not invertible
        for i in range(cov_inv_prod.shape[0]):
            identity = jnp.eye(cov_inv_prod.shape[1])
            assert jnp.allclose(
                cov_inv_prod[i], identity, atol=1e-3
            ), "The covariance matrix was not invertible. Another ILC method may be more appropriate."

        ai_list = ai(self.nu_centers / un.MHz).value

        omega_i = jnp.einsum("j, ij->i", ai_list, inv_cov) / jnp.einsum(
            "k, kl, l -> ", ai_list, inv_cov, ai_list
        )

        # Compute the cleaned map
        # null indicates there is no dark photon signal
        self.clean_null = jnp.einsum("i, i...->i...", omega_i, self.foregrounds)
        self.clean_null = self.clean_null.at[jnp.isnan(self.clean_null)].set(0)
        return self.clean_null

    def cross_correlate(
        self,
        mA: float | None = None,
    ) -> None:
        """
        Compute the cross-correlation between the galaxy catalog and the dark photon signal and the galaxy catalog and the cleaned foreground map.
        This method computes the cross-correlation for a given dark photon mass mA.

        Parameters
        ----------
        mA: float, optional
            Mass of the dark photon in eV. If not provided, will use the previously set value.

        Returns
        -------
        None
            This method does not return anything, but it computes the cross-correlation and stores the resulting correlation functions in the object attributes.
            The attributes are:
            - self.pred_cross_xi: Predicted cross-correlation between the galaxy catalog and the dark photon signal.
            - self.pred_cross_rnom: Theta values for the predicted cross-correlation.
            - self.pred_cross_cov: Covariance matrix for the predicted cross-correlation.
            - self.obs_cross_xi: Observed cross-correlation between the galaxy catalog and the cleaned foreground map.
            - self.obs_cross_rnom: Theta values for the observed cross-correlation.
            - self.obs_cross_cov: Covariance matrix for the observed cross-correlation.
        """

        self.set_config()

        if mA is not None:
            self.mA = mA

        if self.mA is None and mA is None:
            raise ValueError("Please provide a value for mA (dark photon mass in eV).")

        if self.use_pixel_ilc:
            self.pixel_ilc()
            print("ILC completed")
        else:
            self.generate_foregrounds()
            self.compute_dp_signal()
            self.clean_null = self.convert_from_healpy(hp.read_map(self.clean_map_loc))

        if not hasattr(self, "gal_map"):
            self.compute_catalog()

        # only take non-zero galaxy catalog points
        gmap_indices = np.nonzero(self.gal_map)
        small_gsurvey = self.gal_map[gmap_indices]
        theta_gsurvey = self.lat[gmap_indices]
        phi_gsurvey = self.lon[gmap_indices]
        theta_gsurvey = np.repeat(theta_gsurvey, small_gsurvey)
        phi_gsurvey = np.repeat(phi_gsurvey, small_gsurvey)

        # signal without noise (prediction)
        n_patches = 50
        prediction_cat = treecorr.Catalog(
            ra=self.lat.flatten(),
            dec=self.lon.flatten(),
            k=-self.Tgamma0 * (self.Ptot.flatten() - np.mean(self.Ptot)),
            # k=self.clean_null.flatten(),
            ra_units="rad",
            dec_units="rad",
            npatch=n_patches,
        )

        # galaxy map
        gal_cat = treecorr.Catalog(
            ra=theta_gsurvey.flatten(),
            dec=phi_gsurvey.flatten(),
            ra_units="rad",
            dec_units="rad",
            patch_centers=prediction_cat.patch_centers,
        )

        ## define things for random galaxy catalog
        random_theta = rng.uniform(
            np.min(theta_gsurvey),
            np.max(theta_gsurvey),
            10**6,
        )

        random_phi = np.arcsin(
            rng.uniform(
                np.sin(np.min(phi_gsurvey)),
                np.sin(np.max(phi_gsurvey)),
                10**6,
            )
        )

        rand = treecorr.Catalog(
            ra=random_theta.flatten(),
            dec=random_phi.flatten(),
            ra_units="rad",
            dec_units="rad",
            patch_centers=prediction_cat.patch_centers,
        )

        # signal with noise
        cleaned_cat = treecorr.Catalog(
            ra=self.lat.flatten(),
            dec=self.lon.flatten(),
            k=self.clean_null.flatten() - np.mean(self.clean_null),
            ra_units="rad",
            dec_units="rad",
            patch_centers=prediction_cat.patch_centers,
        )

        # initialize the cross-correlation objects
        pred_PP = treecorr.KKCorrelation(self.config)
        pred_PP.process(prediction_cat)
        self.rnom = pred_PP.rnom
        self.xi = pred_PP.xi

        pred_gP = treecorr.NKCorrelation(self.config)
        obs_gP = treecorr.NKCorrelation(self.config)
        pred_rP = treecorr.NKCorrelation(self.config)
        obs_rP = treecorr.NKCorrelation(self.config)

        # compute predicted cross-correlation
        pred_gP.process(gal_cat, prediction_cat)
        pred_rP.process(rand, prediction_cat)
        self.pred_cross_xi, self.pred_cross_xi_var = pred_gP.calculateXi(rk=pred_rP)
        self.pred_cross_rnom = pred_gP.rnom
        self.pred_cross_cov = pred_gP.cov

        # compute observed cross-correlation
        obs_gP.process(gal_cat, cleaned_cat)
        obs_rP.process(rand, cleaned_cat)
        self.obs_cross_xi, self.obs_cross_xi_var = obs_gP.calculateXi(rk=obs_rP)
        self.obs_cross_rnom = obs_gP.rnom
        self.obs_cross_cov = obs_gP.cov

        return None

    def auto_correlate(
        self,
        mA: float | None = None,
    ) -> None:
        """
        Compute the auto-correlation for dark photon signal and the auto-correlation for the cleaned foreground map.
        This method computes the auto-correlation for a given dark photon mass mA.

        Parameters
        ----------
        mA: float, optional
            Mass of the dark photon in eV. If not provided, will use the previously set value.

        Returns
        -------
        None
            This method does not return anything, but it computes the auto-correlation and stores the resulting correlation functions in the object attributes.
            The attributes are:
            - self.pred_auto_xi: Predicted auto-correlation for dark photon signal.
            - self.pred_auto_rnom: Theta values for the predicted auto-correlation.
            - self.pred_auto_cov: Covariance matrix for the predicted auto-correlation.
            - self.obs_auto_xi: Observed auto-correlation for the cleaned foreground map.
            - self.obs_auto_rnom: Theta values for the observed auto-correlation.
            - self.obs_auto_cov: Covariance matrix for the observed auto-correlation.
        """

        self.set_config()

        if mA is not None:
            self.mA = mA

        if self.mA is None and mA is None:
            raise ValueError("Please provide a value for mA (dark photon mass in eV).")

        if self.use_pixel_ilc:
            self.pixel_ilc()
            print("ILC completed")
        else:
            self.generate_foregrounds()
            self.compute_dp_signal()
            self.clean_null = self.convert_from_healpy(hp.read_map(self.clean_map_loc))
        # signal without noise (prediction)
        n_patches = 50
        prediction_cat = treecorr.Catalog(
            ra=self.lat.flatten(),
            dec=self.lon.flatten(),
            k=-self.Tgamma0 * (self.Ptot.flatten() - np.mean(self.Ptot)),
            ra_units="rad",
            dec_units="rad",
            npatch=n_patches,
        )

        # signal with noise
        cleaned_cat = treecorr.Catalog(
            ra=self.lat.flatten(),
            dec=self.lon.flatten(),
            k=self.clean_null.flatten() - np.mean(self.clean_null),
            ra_units="rad",
            dec_units="rad",
            patch_centers=prediction_cat.patch_centers,
        )

        # initialize the cross-correlation objects
        pred_PP = treecorr.KKCorrelation(self.config)
        pred_PP.process(prediction_cat)
        self.pred_auto_rnom = pred_PP.rnom
        self.pred_auto_xi = pred_PP.xi
        self.pred_auto_cov = pred_PP.cov

        obs_PP = treecorr.KKCorrelation(self.config)
        obs_PP.process(cleaned_cat)
        self.obs_auto_rnom = obs_PP.rnom
        self.obs_auto_xi = obs_PP.xi
        self.obs_auto_cov = obs_PP.cov

        # compute predicted cr
        if self.include_analytic:
            if not os.path.exists(f"{self.analytic_path}/analytic_w_mA{mA:.3e}.npy"):
                raise NotImplementedError("Haven't run calculations for that mass yet")

            analytic_theta = np.load(f"{self.analytic_path}/analytic_theta_ary.npy")
            analytic_xi = self.Tgamma0**2 * np.load(
                f"{self.analytic_path}/analytic_w_mA{mA:.3e}.npy"
            )

            interp_analytic_xi = interpolate.interp1d(
                analytic_theta, analytic_xi, bounds_error=True, fill_value=0
            )
            self.pred_auto_xi += interp_analytic_xi(self.pred_auto_rnom)

        return None

    def set_config(self):
        # Note to self. nbins=20, max_sep=lat/2, var_method bootstrap, npatches=50, 10**6 points in rand produces
        # the most stable correlation function so far. Now we have to adjust the covariance stuff to stabilize that.
        # seems like jackknife is less stable
        # now marked_bootstrap is most stable

        self.config = {
            "nbins": 20,
            "metric": "Arc",
            "min_sep": (np.pi / 4096),
            "max_sep": self.lat[-1, 0] / 2,
            "bin_type": "Log",
            "bin_slop": 0,
            "var_method": "bootstrap",
            "num_threads": 8,
        }
        if self.config["var_method"] == "bootstrap":
            self.config["cross_patch_weight"] = "geom"
        elif self.config["var_method"] == "jackknife":
            self.config["cross_patch_weight"] = "match"

    def auto_log_L(self, eps4: float) -> float:
        """
                Compute the log-likelihood for the auto-correlation of the dark photon signal.

                Parameters
                ----------
                eps4: float
                    Kinetic mixing for the dark photon, raised to the fourth power.
        `
                Returns
                -------
                Log_l: float
                    Log-likelihood value for the auto-correlation.
        """
        Log_l = (
            -0.5
            * (eps4 * self.pred_auto_xi - self.obs_auto_xi)
            @ np.linalg.inv(self.obs_auto_cov)
            @ (eps4 * self.pred_auto_xi - self.obs_auto_xi)
        )
        return Log_l

    def cross_log_L(self, eps2: float):
        """
        Compute the log-likelihood for the cross-correlation between the galaxy catalog and the dark photon signal.

        Parameters
        ----------
        eps2: float
            Kinetic mixing for the dark photon, raised to the second power.

        Returns
        -------
        Log_l: float
            Log-likelihood value for the cross-correlation.
        """
        Log_l = (
            -0.5
            * (eps2 * self.pred_cross_xi - self.obs_cross_xi)
            @ np.linalg.inv(self.obs_cross_cov)
            @ (eps2 * self.pred_cross_xi - self.obs_cross_xi)
        )
        return Log_l

    def find_cross_limits(self, mA: float | None = None) -> float:
        """
        Compute the cross-correlation limits for the dark photon signal.

        Parameters
        ----------
        mA: float | None
            Mass of the dark photon in eV. If not provided, will use the previously set value.

        Returns
        -------
        cross_limit: float
            Cross-correlation limit for the dark photon signal, in terms of the kinetic mixing parameter epsilon

        """
        if mA is not None:
            self.mA = mA

        if self.mA is None and mA is None:
            raise ValueError("Please provide a value for mA (dark photon mass in eV).")

        self.cross_correlate(self.mA)
        max_cross_eps = optimize.minimize(
            lambda eps: -self.cross_log_L(eps**2),
            x0=1e-6,
            tol=1e-8,
            method="Nelder-Mead",
        ).x

        def cross_lam(eps2):
            max_log_L = self.cross_log_L(max_cross_eps**2)
            if eps2 < max_cross_eps**2:
                return 1
            else:
                return np.exp(self.cross_log_L(eps2) - max_log_L)

        self.cross_limit = optimize.minimize(
            lambda eps: np.abs(2.71 + 2 * np.log(cross_lam(eps**2))),
            x0=max_cross_eps,
            options={"xatol": 1e-10},
            method="Nelder-Mead",
        ).x

        print(f"done with cross correlation, limit is: {self.cross_limit[0]:.3e}")
        # compute log likelihood
        return self.cross_limit

    def find_auto_limits(self, mA: float | None = None) -> float:
        """
        Compute the auto-correlation limits for the dark photon signal.

        Parameters
        ----------
        mA: float | None
            Mass of the dark photon in eV. If not provided, will use the previously set value.

        Returns
        -------
        auto_limit: float
            Auto-correlation limit for the dark photon signal, in terms of the kinetic mixing parameter epsilon.
        """
        if mA is not None:
            self.mA = mA

        if self.mA is None and mA is None:
            raise ValueError("Please provide a value for mA (dark photon mass in eV).")
        self.auto_correlate(self.mA)
        max_auto_eps = optimize.minimize(
            lambda eps: -self.auto_log_L(eps**4),
            x0=0,
            options={"fatol": 0.001},
            method="Nelder-Mead",
        ).x

        def auto_lam(eps4):
            max_log_L = self.auto_log_L(max_auto_eps**4)
            if eps4 < max_auto_eps**4:
                return 1
            else:
                return np.exp(self.auto_log_L(eps4) - max_log_L)

        self.auto_limit = optimize.minimize(
            lambda eps: np.abs(2.71 + 2 * np.log(auto_lam(eps**4))),
            x0=max_auto_eps,
            options={"xatol": 1e-10},
            method="Nelder-Mead",
        ).x
        print(f"done with auto correlation, limit is: {self.auto_limit[0]:.3e}")

        return self.auto_limit

    def find_limits(self, mA) -> tuple[float, float]:
        """
        Compute both cross-correlation and auto-correlation limits for the dark photon signal.

        Parameters
        ----------
        mA: float | None
            Mass of the dark photon in eV. If not provided, will use the previously set value.

        Returns
        -------
        auto_limit: float
            Auto-correlation limit for the dark photon signal, in terms of the kinetic mixing parameter epsilon
        cross_limit: float
            Cross-correlation limit for the dark photon signal, in terms of the kinetic mixing parameter epsilon.
        """
        self.find_cross_limits(mA)
        self.find_auto_limits(mA)

        return self.auto_limit, self.cross_limit

    def convert_to_healpy(self, field: np.ndarray) -> np.ndarray:
        """
        Convert a field (e.g., dark photon conversion map) to a HEALPix map.
        This method assumes the field is defined on a Cartesian grid and converts it to a HEALPix map
        using the survey center as the reference point.

        Parameters
        ----------
        field: np.ndarray
            Field to be converted, should have shape (dim, dim)

        Returns
        -------
        hpmap: np.ndarray
            HEALPix map of the field, with shape (npix,).
        This map is defined on the HEALPix grid with the specified nside.
        """

        ## UNUSED but could be useful in principle
        # turn dark photon conversion map into healpy map
        aphp = HEALPix(nside=self.nside, order="ring")
        # TODO: change this to move it to the center of the survey, but this is fine for now
        if (
            not hasattr(self, "LON")
            or not hasattr(self, "LAT")
            or not hasattr(self, "_hpindex")
        ):
            self.LON, self.LAT = np.meshgrid(
                (np.radians(self.survey_center[0]) + self.lon[0, :]) * un.radian,
                (np.radians(self.survey_center[1]) + self.lat[0, :]) * un.radian,
            )
            self.LON = self.LON.flatten()
            self.LAT = self.LAT.flatten()
            self._hpindex = aphp.lonlat_to_healpix(
                self.LON,
                self.LAT,
            )
            self._hpmask = np.zeros(aphp.npix, dtype=bool)
            self._hpmask[self._hpindex] = True
            self.total_mask = self._hpmask * ~self.bool_map

        hpmap = np.zeros(aphp.npix)
        hpmap[self._hpindex] = field.flatten()
        return hpmap

    def convert_from_healpy(self, field: np.ndarray) -> np.ndarray:
        """
        Convert a HEALPix map to a Cartesian field.
        This method assumes the field is defined on a HEALPix grid and converts it to a Cartesian field
        using the survey center as the reference point.

        Parameters
        ----------
        field: np.ndarray
            HEALPix map to be converted, should have shape (npix,).

        Returns
        -------
        cartesian_field: np.ndarray
            Cartesian field of the HEALPix map, with shape (dim, dim).
        """
        proj = hp.projector.CartesianProj(
            rot=self.survey_center,
            lonra=[-np.rad2deg(np.max(self.lon)) / 2, np.rad2deg(np.max(self.lon)) / 2],
            latra=[-np.rad2deg(np.max(self.lat)) / 2, np.rad2deg(np.max(self.lat)) / 2],
            coord="C",
            xsize=self.dim,
            ysize=self.dim,
        )
        cartesian_field = proj.projmap(
            field, vec2pix_func=partial(hp.vec2pix, self.nside)
        )
        return cartesian_field
