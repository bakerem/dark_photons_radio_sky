"""
Description: This module computes analytic angular power spectra (Cls) and two point functions for dark photon perturbations
in the low mass regime, using the lognormal PDF for density fluctuations.
It saves the results for different dark photon mass values (mA) to specified output files.
The user needs to install this https://github.com/smsharma/dark-photons-perturbations first. 
Github: https://github.com/bakerem
"""
import os, sys
import pickle

sys.path.append("../")

import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import numpy as np

from scipy.special import spherical_jn
from scipy.special import hyp2f1
from tqdm import tqdm

from scipy.interpolate import interp1d

grf_path = "/home/bakerem/dark-photons-perturbations"
sys.path.append(grf_path)

from grf.grf import PerturbedProbability, FIRAS
from grf.pk_interp import PowerSpectrumGridInterpolator
from grf.units import *


# Load plot settings
from plot_params import params

pylab.rcParams.update(params)

cols_default = plt.rcParams["axes.prop_cycle"].by_key()["color"]


# Load nonlinear matter spectrum with a baryon Jeans scale suppression
log_pspec = PowerSpectrumGridInterpolator("franken_lower")

prob = PerturbedProbability(log_pspec)
firas = FIRAS(
    log_pspec
)  # Also load a FIRAS class to access the FIRAS frequencies, as benchmarks

# Matter Power Spectrum Parameters
k_min = 1e-6
k_max = 1e6


# Frequency to evaluate dP/dz in our code. Options are 0.14, 4.86, 8.4, 8.7
nu_in_GHz = 0.14

# Frequency in keV, the natural units of this code.
omega_0 = 1 * eV  # 2 * np.pi * nu_in_GHz * 1e9 * Hz
# mA_list = np.sort(np.concatenate([mA_list1, mA_list2, mA_list3]))
# mA_list = np.concatenate([mA_list1, mA_list2])
mA_list = eV * np.geomspace(1e-15, 1e-14, 10)
mA_list = eV * np.geomspace(5e-15, 1e-13, 15)[::2]


def compute_Cls(m_Ap):
    # Dark photon mass.
    # m_Ap = 1e-14 * eV

    eps = 1.0  # 7e-7 #4.12e-7 #FIRAS bound roughly

    # z_star, which is the maximum value of z, and aligns with our halo simulation
    z_star = 4.0

    # Redshift binning.
    z_ary = np.geomspace(0.005, z_star, 1000)

    # Plasma mass squared over the redshift binning, in keV^2.
    m_A_sq_ary = firas.m_A_sq(z_ary, omega_0)

    # sigma_1^2 = log(1 + sigma^2) variance of fluctuations, dimensionless.
    sigma_1_sq_ary = firas._dP_dz(z_ary, m_Ap, k_min, k_max, omega_0, pdf="lognormal")[
        1
    ][0]
    sigma_sq_ary = np.exp(sigma_1_sq_ary) - 1.0

    # g = mA^2 / <m_gamma^2> - 1
    g_ary = m_Ap**2 / m_A_sq_ary - 1

    np.trapz(
        firas._dP_dz(z_ary, m_Ap, k_min, k_max, omega_0, pdf="lognormal")[0][0], z_ary
    )

    import pyfftlog

    def fftj0(f, logrmin, logrmax, n_pts=4096, q=0):
        """Fourier transform of function a(r).

        The actual integral computed is \int d^3 r a(r) j_0(k r), which is the Fourier transform for a function that only depends on magnitude of r.

        Parameters
        ----------
        f : function
            function to FFT, returns an array of function values, r x ....
        logrmin : float
            log10 minimum value of r to include.
        logrmax : float
            log10 maximum value of r to include.
        n_pts : int, optional
            number of data points to use, max = 4096.
        q : float, optional
            the bias of the integral to use

        Returns
        -------
        tuple of ndarray
            Returns k abscissa and result.

        Notes
        -------

        pyfftlog will evaluate \int dr k (kr)^q J_1/2(kr) a(r) (kr)^(3/2 - q), and the bias q can be set arbitrarily, although q = 0 usually gives the best performance.
        """

        # Sensible approximate choice of k_c r_c
        kr = 1

        # Tell fhti to change kr to low-ringing value
        # WARNING: kropt = 3 will fail, as interaction is not supported
        kropt = 1

        # Forward transform (changed from dir to tdir, as dir is a python fct)
        tdir = 1

        # Central point log10(r_c) of periodic interval
        logrc = (logrmin + logrmax) / 2

        # Central index (1/2 integral if n is even)
        nc = (n_pts + 1) / 2.0

        # Log-spacing of points
        dlogr = (logrmax - logrmin) / n_pts
        dlnr = dlogr * np.log(10.0)

        # Initialization. kr = k_c r_c, where c is the central point.
        kr, xsave = pyfftlog.fhti(n_pts, 0.5, dlnr, q, kr, kropt)
        logkc = np.log10(kr) - logrc

        # Actual r-binning
        r_ary = 10 ** (logrc + (np.arange(1, n_pts + 1) - nc) * dlogr)
        # Actual k-binning
        k_ary = 10 ** (logkc + (np.arange(1, n_pts + 1) - nc) * dlogr)

        # function to log-Fourier transform.
        # In general f returns something multidimensional.
        # ar_ary has dimensions ... x r_ary.
        ar_ary = np.moveaxis(f(r_ary), 0, -1) * (r_ary) ** (1.5 - q)

        # dimensions ... x r_ary
        ak_ary = np.zeros(ar_ary.shape)

        if len(ak_ary.shape) > 1:

            # Array of indices, dimensions ... x 2
            indices_ary = np.moveaxis(np.indices(ak_ary[..., 0].shape), 0, -1)

            for ind in indices_ary.reshape(-1, indices_ary.shape[-1]):

                if ind.shape == ():

                    ak_ary[ind] = (
                        (2 * np.pi) ** 1.5
                        * k_ary ** (-1.5 - q)
                        * pyfftlog.fht(ar_ary[ind].copy(), xsave, tdir)
                    )

                else:

                    ak_ary[tuple(ind)] = (
                        (2 * np.pi) ** 1.5
                        * k_ary ** (-1.5 - q)
                        * pyfftlog.fht(ar_ary[tuple(ind)].copy(), xsave, tdir)
                    )

        else:

            ak_ary = (
                (2 * np.pi) ** 1.5
                * k_ary ** (-1.5 - q)
                * pyfftlog.fht(ar_ary.copy(), xsave, tdir)
            )

        # Return as dimension ... x k_ary
        return (k_ary, ak_ary)

    # This is the correlation function MINUS sigma**2, to avoid close cancellations with sigma**2, and guarantees a negative outcome.
    def corr_func(r_ary, z_ary, power_spec, kmin=1e-4, kmax=1e4):

        # r_ary should be indexed by z_ary at axis = -1.

        k_ary = np.logspace(np.log10(kmin), np.log10(kmax), 700)

        # Dimensions z_ary x k_ary
        log_pspec_ary = power_spec(z_ary, k_ary)

        #     try:

        _ = iter(r_ary)

        # Dimensions r_ary x k_ary
        k_times_r = np.einsum("...i,j->...ij", r_ary, k_ary)

        # Dimensions r_ary x k_ary
        integrand = (
            1.0
            / (2.0 * np.pi**2)
            * np.einsum(
                "k, jk, ...jk -> ...jk",
                k_ary**2,
                10**log_pspec_ary,
                (np.sin(k_times_r) / k_times_r - 1.0),
            )
        )

        return np.trapz(integrand, k_ary)

    def p_spec_out(k_ary):

        return np.transpose(10 ** log_pspec(z_ary, k_ary))

    r_corr_func_fft_ary, corr_func_fft_ary = fftj0(
        p_spec_out, np.log10(k_min), np.log10(k_max)
    )
    corr_func_fft_ary /= (2 * np.pi) ** 3

    corr_func_fft = interp1d(r_corr_func_fft_ary, corr_func_fft_ary)

    Omega_Lambda = firas.cosmo.Ode0
    Omega_m = firas.cosmo.Odm0 + firas.cosmo.Ob0
    H0_in_per_Mpc = firas.cosmo.H(0).value * Kmps
    h = firas.cosmo.h

    def comoving_dist(z):
        """
        Returns the comoving distance to z, in Mpc/h
        """

        def indef_integral(y):

            return (
                (1.0 + y)
                / (H0_in_per_Mpc * Omega_Lambda)
                * np.sqrt(Omega_Lambda + Omega_m * (1 + y) ** 3)
                * hyp2f1(5 / 6, 1.0, 4 / 3, -Omega_m * (1 + y) ** 3 / Omega_Lambda)
            )

        return (indef_integral(z) - indef_integral(0)) * h

    chi_ary = comoving_dist(z_ary)

    r_ary = np.logspace(-5, 5, num=700)

    r_ary_for_xi = np.outer(r_ary, np.ones_like(z_ary))

    # dimensions r_ary x z_ary
    # xi_ary = corr_func(r_ary_for_xi, z_ary, log_pspec, kmin=k_min, kmax=k_max) + sigma_sq_ary
    xi_ary = np.transpose(corr_func_fft(r_ary))

    L_ary = np.log1p(g_ary) + sigma_1_sq_ary / 2
    X_ary = np.log1p(xi_ary)

    # r_ary x z_ary
    expr_to_fft_full_ary = 1.0 / np.sqrt(1 - X_ary**2 / sigma_1_sq_ary**2) * np.exp(
        -(L_ary**2) / (sigma_1_sq_ary + X_ary)
    ) - np.exp(-(L_ary**2) / sigma_1_sq_ary)

    # print(expr_to_fft_full_ary)

    # expr_to_fft_small_ary = xi_ary * g_ary**2 / sigma_sq_ary**2

    expr_to_fft_ary = np.zeros_like(expr_to_fft_full_ary)

    # expr_to_fft_ary = expr_to_fft_full_ary

    expr_to_fft_ary[expr_to_fft_full_ary > 0] = expr_to_fft_full_ary[
        expr_to_fft_full_ary > 0
    ]

    func_to_fft = interp1d(r_ary, expr_to_fft_ary, axis=0)

    fft_res = fftj0(func_to_fft, np.log10(r_ary[0]), np.log10(r_ary[-1]))

    l_over_chi_ary = fft_res[0]
    integrand = fft_res[1]

    # integrand = term_1[:,None] * term_2

    # res = np.array([
    #     np.trapz(int_z, l_r_over_chi_ary)
    #     for int_z in np.transpose(integrand)
    # ])

    # if not debug:

    #     return res

    # else:

    #     return (l_r_over_chi_ary,integrand), res

    # returns shape l_over_chi_ary x r_ary
    r_integral = interp1d(
        l_over_chi_ary,
        integrand,
        bounds_error=False,
        fill_value=(np.nan, integrand[:, -1]),
    )

    P_mean = eps**2 * np.trapz(
        firas._dP_dz(z_ary, m_Ap, k_min, k_max, omega_0, pdf="lognormal")[0][0], z_ary
    )

    def C_l(l):

        # H in h Mpc^-1
        hubble_ary = firas.cosmo.H(z_ary).value * Kmps / h

        r_int = np.diag(r_integral(l / comoving_dist(z_ary)))

        # print(np.min(l/comoving_dist(z_ary)), np.max(l/comoving_dist(z_ary)))

        # Mpc**2 / h**2 to convert keV^2 (including m_A_sq_ary below) to h^2 Mpc^-2
        prefac = (np.pi * m_Ap**4 * eps**2 / omega_0) ** 2 * Mpc**2 / h**2

        integrand = (
            1.0
            / comoving_dist(z_ary) ** 2
            / hubble_ary
            / (1.0 + z_ary) ** 4
            / (2 * np.pi * sigma_1_sq_ary * m_A_sq_ary**2 * (1.0 + g_ary) ** 2)
            * r_int
        )

        return np.trapz(prefac * integrand, z_ary)  # / P_mean**2

    l_ary = np.linspace(0, 10_000, 10000, dtype=int)
    np.save(
        f"/home/bakerem/dark_photon_21cm_constraints/halo_data/analytic_Cls/analytic_ls",
        l_ary,
    )
    C_l_ary = np.array([C_l(l) for l in l_ary])
    C_l_ary[0] = P_mean**2
    np.save(
        f"/home/bakerem/dark_photon_21cm_constraints/halo_data/analytic_Cls/analytic_Cls_mA{m_Ap/eV:.3e}.npy",
        C_l_ary,
    )

    Cl_interp = interp1d(
        l_ary, C_l_ary, bounds_error=False, fill_value=(np.nan, C_l_ary[-1])
    )

    @np.vectorize
    def w(theta, lmax=10_000):
        l_ary = np.arange(1, lmax, dtype=int)

        return np.sum(
            [
                (2 * l + 1)
                * Cl_interp(l)
                / 4
                / np.pi
                * hyp2f1(-l, l + 1, 1, (1 - np.cos(theta)) / 2)
                for l in l_ary
            ]
        )

    theta_ary = np.geomspace(5e-4, 2e-2, 100)
    np.save(
        f"/home/bakerem/dark_photon_21cm_constraints/halo_data/analytic_Cls/analytic_theta_ary",
        theta_ary,
    )
    np.save(
        f"/home/bakerem/dark_photon_21cm_constraints/halo_data/analytic_Cls/analytic_w_mA{m_Ap/eV:.3e}.npy",
        w(theta_ary),
    )
    return None


for mA in tqdm(mA_list):
    compute_Cls(mA)
