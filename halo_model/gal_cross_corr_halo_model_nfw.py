import os, sys
"""
Author: Ethan Baker, Boston University
Github: github.com/bakerem
Galaxy-Dark Photon Cross-Correlation Halo Model Calculator

This module computes the cross-correlation power spectra between galaxies and dark photon
signals using the halo model framework. It calculates angular power spectra (C_l) for:
- Galaxy auto-correlation (C_l^gg)
- Dark photon auto-correlation (C_l^PP)
- Galaxy-dark photon cross-correlation (C_l^gP)

The module implements:
- Halo Occupation Distribution (HOD) models for galaxy populations
- Gas density profiles following Battaglia et al. prescription
- Dark photon conversion in galaxy halos
- 1-halo and 2-halo term calculations
- Limber approximation for angular power spectra

Key Components:
--------------
- Custom Bhattacharya12 concentration-mass relation
- McCarthy gas profile for halo gas distribution
- HOD models from Krolewski et al. 2022 (K22) and 2023 (K23)
- Dark photon mass-dependent conversion calculations

Arrays and Grids:
----------------
- Redshift range: z = 0.005 to 4 (100 points)
- Halo mass range: 10^11 to 10^17 M_sun (100 points)
- Wavenumber range: 10^-4 to 10^3 Mpc^-1 (10^4 points)
- Multipole moments: l = 1 to 7000 (50 points, log-spaced)

Output:
-------
Saves computed power spectra to `output_dir` with naming convention:
- Cl_gg_model{model}.npy: Galaxy auto-correlation
- Cl_PP_mA{mA:.3e}.npy: Dark photon auto-correlation
- Cl_gP_mA{mA:.3e}_model{model}.npy: Cross-correlation

Constants:
----------
- kappa: Dark photon coupling constant (5.7e-38 eV^2)
- Planck18: Cosmology model from Astropy

Notes:
------
- All halo mass function (HMF) calculations use h^-1 units internally
- Conversion to physical units (Mpc, M_sun) applied for final outputs
"""
import pickle

import matplotlib_inline

sys.path.append("../")

import astropy.units as un
from astropy.cosmology import Planck18
import astropy.constants as const
import numpy as np
from scipy.integrate import simpson, quad, dblquad
from scipy.interpolate import interp1d, griddata
from scipy.optimize import fsolve

from tqdm import tqdm

import mpmath as mp

import halomod as hd
import hmf
import scipy.special as sp
import timeit

output_dir = "/projectnb/darkcosmo/dark_photon_project/21cmfast_cache/halo_data/correct_virial_mass"
## REMINDER: ALL HMF IS IN /h units, so M should be Msun/h etc.


# need to define concentration
class Bhattacharya12(hd.concentration.CMRelation):
    _defaults = {"a": None, "b": None, "c": None}
    native_mdefs = (hmf.mass_definitions.SOCritical(), hmf.mass_definitions.SOVirial())

    def cm(self, m, z):
        set_params = {
            "200c": {
                "a": 0.54,
                "b": 5.9,
                "c": -0.35,
            },
            "178c": {
                "a": 0.9,
                "b": 7.7,
                "c": -0.29,
            },
            "vir": {
                "a": 0.9,
                "b": 7.7,
                "c": -0.29,
            },
        }
        parameter_set = set_params.get(self.mdef.colossus_name, set_params["200c"])
        nu = self.growth.growth_factor(z) ** -1 * (1.12 * (m / 5e13) ** 0.3 + 0.53)
        return (
            self.growth.growth_factor(z) ** parameter_set["a"]
            * parameter_set["b"]
            * nu ** parameter_set["c"]
        )


kappa = 5.7e-38  # eV**2

# arrays of redshift, halo masses, and wavenumbers
zs = np.linspace(0.005, 4, 100)
ms = np.geomspace(
    1e11,
    1e17,
    100,
)  # Msolar
dlogm = np.diff(np.log10(ms * Planck18.h))[0]
ks = np.geomspace(
    1e-4,
    1e3,
    10**4,
)  # Mpc**-1
dlnk = np.diff(np.log(ks / Planck18.h))[0]

hm = hd.DMHaloModel(
    z=0.0,
    Mmin=np.log10(ms[0] * Planck18.h),  # Msun/h
    Mmax=np.log10(ms[-1] * Planck18.h),  # + dlogm, #Msun/h
    dlog10m=dlogm,
    lnk_min=np.log(ks[0] / Planck18.h),  # h/Mpc
    lnk_max=np.log(ks[-1] / Planck18.h) + dlnk,  # h / Mpc
    dlnk=dlnk,
    halo_concentration_model="Bhattacharya12",
    bias_model="Tinker10",
    hmf_model="Tinker08",
    mdef_model=hmf.halos.mass_definitions.SOVirial,
    cosmo_params={
        "m_nu": [0, 0, 0],
    },
)

rhocritz = Planck18.critical_density(zs).to(un.Msun / un.Mpc**3).value  # Msun / Mpc**3

bias = np.empty((len(zs), len(ms)))
n = np.empty((len(zs), len(ms)))
cs = np.empty((len(zs), len(ms)))
rvirs = np.empty((len(zs), len(ms)))
R200s = np.empty((len(zs), len(ms)))
m200s = np.empty((len(zs), len(ms)))
c200s = np.empty((len(zs), len(ms)))
lin_power = np.empty((len(zs), len(ks)))

for i, z in tqdm(enumerate(zs)):
    hm.z = z
    n[i, :] = hm.dndm * Planck18.h**4
    bias[i, :] = hm.halo_bias  # dimless
    cs[i, :] = hm.cmz_relation  # dimless
    lin_power[i, :] = (
        hm.linear_power_fnc(ks / Planck18.h) / Planck18.h**3
    )  # conver to Mpc^3
    for j, m in enumerate(ms):
        # rvirs[i, j] = fsolve(lambda r: hm.halo_profile.rho(r, m * Planck18.h, coord="r") * Planck18.h**2 - rhocritz[i] * delta_vir, 0.0001)[0] / Planck18.h
        rvirs[i, j] = (
            hm.halo_profile.halo_mass_to_radius(m * Planck18.h, at_z=z) / Planck18.h
        )  # Mpc
        m200s[i, j], R200s[i, j], c200s[i, j] = hm.mdef.change_definition(
            m * Planck18.h,
            mdef=hmf.halos.mass_definitions.SOCritical(overdensity=200),
            c=cs[i, j],
            z=z,
            cosmo=Planck18,
        )
        # R200s[i,j] = hm.halo_profile.halo_mass_to_radius(m * Planck18.h, at_z=z)/Planck18.h # Mpc
        # R200s[i, j] = fsolve(lambda r: hm.halo_profile.rho(r, m * Planck18.h, coord="r") * Planck18.h**2 - rhocritz[i] * 200, 0.0001)[0] / Planck18.h
        # m200s[i,j] = hm.halo_profile.halo_radius_to_mass(R200s[i,j]*Planck18.h, at_z=z)/Planck18.h
m200s /= Planck18.h
R200s /= Planck18.h

# R200 = rvir
# x is in units of x/xc$
# rs = R200[...,None] * xs[None, None, :] / 2
rs = np.linspace(1e-12, 1.1 * rvirs, int(1e4))  # Mpc
rs = np.moveaxis(rs, 0, -1)


rho_halo = ms[None, :] / (4 / 3 * np.pi * rvirs**3)
ANFW     = np.log(1 + cs) - cs / (1 + cs)
x        = rs / rvirs[..., None]
rho_gases = Planck18.Ob0 / Planck18.Om0 * rho_halo[..., None] / (3 * ANFW[..., None] * x * (cs[..., None]**-1 + x)**2)

chis = Planck18.comoving_distance(zs).to(un.Mpc).value  # Mpc
mgamma2 = kappa * rho_gases
Plin_z1_z2 = np.sqrt(lin_power[None, :] * lin_power[:, None])

bessel_arg1 = ks[None, None, :] * chis[:, None, None]
bessel_arg2 = ks[None, None, :] * chis[None, :, None]
integrand_without_js = Plin_z1_z2 * ks[None, None, :] ** 2


def hl(l):
    return (
        (2 * l + 1)
        / 2
        / np.pi**2
        * simpson(
            integrand_without_js
            * sp.spherical_jn(l, bessel_arg1)
            * sp.spherical_jn(l, bessel_arg2),
            x=ks,
            axis=-1,
        )
    )


interpolated_Plin = interp1d(ks, lin_power, axis=-1)

# dNgdz_dat = np.genfromtxt("/projectnb/darkcosmo/dark_photon_project/21cmfast_cache/halo_data/unwise_stuff/galaxy_dNg_dz.csv", delimiter=",").T
dNgdz_dat = np.genfromtxt(
    "/projectnb/darkcosmo/dark_photon_project/21cmfast_cache/halo_data/unwise_stuff/galaxy_dNg_dz.csv",
    delimiter=",",
).T

dNgdz = interp1d(dNgdz_dat[0], dNgdz_dat[1], bounds_error=False, fill_value=0.0)
ls = np.geomspace(1, 7000, 50, dtype=int)
np.save(f"{output_dir}/ls.npy", ls)

HOD_model = {
    "K22": {
        "logm_min": 11.97,  # K22
        "sigma_logm": 0.687,  # K22
        "alpha_s": 1.304,  # K22
        "m1prime": 10**12.87,  # K22
        "lambdaNFW": 1.087,  # K22
    },
    "K23": {
        "logm_min": 11.86,  # K23
        "sigma_logm": 0.020,  # K23
        "alpha_s": 1.06,  # K23
        "m1prime": 10**12.78,  # K23
        "lambdaNFW": 1.80,  # K23
    },
}

model = "K22"  # Change to "K23" for the K23 model


def Nc(m):
    logm_min = HOD_model[model]["logm_min"]
    sigma_logm = HOD_model[model]["sigma_logm"]
    return 0.5 * (1 + sp.erf((np.log10(m) - logm_min) / sigma_logm))


def Ns(m):
    alpha_s = HOD_model[model]["alpha_s"]
    m1prime = HOD_model[model]["m1prime"]
    return Nc(m) * (m / m1prime) ** alpha_s


ngbar = simpson(n * (Nc(ms)[None, :] + Ns(ms)[None, :]), x=ms, axis=1)


def W(z):
    return ((Planck18.H(z) / const.c) / Planck18.comoving_distance(z) ** 2).to(
        un.Mpc**-3
    ).value * dNgdz(z)


def ug(Ws, Ncs, Nss, l):
    return Ws[:, None] * ngbar[:, None] ** -1 * (Ncs[None, :] + Nss[None, :] * um(l))


def um(l):
    lambdaNFW = HOD_model[model]["lambdaNFW"]
    ks = (l + 0.5) / chis[:, None]
    q = ks * R200s / c200s * (1 + zs[:, None])

    x = lambdaNFW * c200s
    qtilde = (1 + x) * q

    Si_q, Ci_q = sp.sici(q)
    Si_qtilde, Ci_qtilde = sp.sici(qtilde)

    fNFW = (np.log(1 + x) - x / (1 + x)) ** -1
    return (
        np.cos(q) * (Ci_qtilde - Ci_q)
        + np.sin(q) * (Si_qtilde - Si_q)
        - np.sin(x * q) / qtilde
    ) * fNFW


def ug_mom2(Ws, Nss, l):
    return (
        Ws[:, None] ** 2
        * ngbar[:, None] ** -2
        * (Nss[None, :] ** 2 * um(l) ** 2 + 2 * Nss * um(l))
    )


# precompute all these arrays for speed
Ncs = Nc(ms)
Nss = Ns(ms)
Ws = W(zs)


# functions for computing the galaxy autopower spectrum
def Cl_1h_gg(l):
    return np.trapz(
        chis**2
        / ((Planck18.H(zs) / const.c).to(un.Mpc**-1).value)
        * np.trapz(n * ug_mom2(Ws, Nss, l=l), x=ms, axis=1),
        x=zs,
        axis=0,
    )


interp_lin_power = interp1d(ks, lin_power, axis=-1)


def Cl_2h_gg_limber(l):
    m_int = simpson(n * bias * ug(Ws, Ncs, Nss, l=l), x=ms, axis=1)
    Plin_at_zs = np.array(
        [interp_lin_power((l + 0.5) / chi)[chis == chi] for chi in chis]
    ).reshape(zs.shape)
    return simpson(
        chis**2
        / ((Planck18.H(zs) / const.c).to(un.Mpc**-1).value)
        * m_int**2
        * Plin_at_zs,
        x=zs,
        axis=0,
    )


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx], idx


Cl_gg = np.array([Cl_1h_gg(l) + Cl_2h_gg_limber(l) for l in ls])
print(Cl_gg[0])
np.save(f"{output_dir}/Cl_gg_model{model}.npy", Cl_gg)

hls = np.array([hl(l) for l in tqdm(ls)])


def get_Cls(mA):
    # function to find nearest value in array to given value
    if os.path.exists(f"{output_dir}/Cl_gP_mA{mA:.3e}_model{model}_nfw.npy"):
        return
    drho_drs = np.empty([len(zs), len(ms)])  # these match McCarthy code seemingly
    r_convs = np.empty([len(zs), len(ms)])  # so do these, gas profile is working right.
    errs = np.empty([len(zs), len(ms)])

    # iterate through redshifts and masses to find drho_dr when the conversion occurs
    # this is inefficient but it's really fast so it doesn't matter
    for i, z in enumerate(zs):
        for j, m in enumerate(ms):
            mA_index = find_nearest(np.sqrt(mgamma2[i, j, :]), mA)[1]
            drho_dr = np.abs(np.gradient(rho_gases[i, j, :], rs[i, j, :]))
            drho_drs[i, j] = np.abs(drho_dr[mA_index])
            r_convs[i, j] = rs[i, j, mA_index]
            if mA_index == len(mgamma2[i, j, :]) - 1:
                errs[i, j] = 0
            else:
                # errs[i,j] = np.abs((mgamma2[i, j, mA_index] - mA**2)/(mA**2))
                errs[i, j] = np.abs((mgamma2[i, j, mA_index] - mA**2) / (mA**2))

    # compute quantities for determining the monopole
    drho_drs /= 1.567e29  # unit conversion
    theta_max_array = (
        r_convs * (1 + zs[:, None,]) / chis[:,None,]
    )  # these match
    u00 = (
        (1 + zs[:, None,])**2 * r_convs**2 / chis[:,None,]** 2 / 2
    )

    epsilon = 1 
    omega0 = 1  
    const_prefact = epsilon**2 * mA**4 / omega0
    P = (
        2
        * np.pi
        * const_prefact
        / kappa
        * (drho_drs) ** -1
        * np.heaviside(rvirs - r_convs, 0.5)
        / (
            1
            + zs[
                :,
                None,
            ]
        )
    )  # these match
    deta_dz = (
        4
        * np.pi
        * chis**2
        / ((Planck18.H(zs) / const.c).to(un.Mpc**-1).value)
        * simpson(P * n * u00, x=ms, axis=1)
    )  # differs below z=2

    def approx_ul0(l, theta_max):
        mu = -0.5
        arg = 1 - theta_max**2 / 2
        return (
            np.pi
            * np.sqrt((2 * l + 1))
            / 2
            * theta_max**1.5
            * (4 - theta_max**2) ** 0.25
            * 1
            / sp.gamma(1 - mu)
            * ((1 + arg) / (1 - arg)) ** (mu / 2)
            * sp.hyp2f1(-l, l + 1, 1 - mu, (1 - arg) / 2)
        )

    def Cl_1halo_pirvu(l, theta_max):
        return (
            4
            * np.pi
            / (2 * l + 1)
            * simpson(
                simpson(
                    chis[:, None] ** 2 * approx_ul0(l, theta_max) ** 2 * P**2 * n,
                    x=ms,
                    axis=1,
                ),
                x=chis,
                axis=0,
            )
        )

    def Cl_2halo(l, theta_max):
        inner_int = simpson(
            n[
                None,
                :,
                :,
                None,
            ]
            * n[
                :,
                None,
                None,
                :,
            ]
            * bias[
                None,
                :,
                :,
                None,
            ]
            * bias[
                :,
                None,
                None,
                :,
            ]
            * P[None, :, :, None]
            * P[:, None, None, :]
            * approx_ul0(l, theta_max)[None, :, :, None]
            * approx_ul0(l, theta_max)[:, None, None, :],
            x=ms,
            axis=-1,
        )
        m_ints = simpson(
            inner_int,
            x=ms,
            axis=-1,
        )
        z_ints = simpson(
            simpson(
                m_ints
                #  * Cl_lin(l)[..., None]
                * hls[l == ls][0]
                * 4
                * np.pi
                / (2 * l + 1)
                * chis[
                    :,
                    None,
                ]
                ** 2
                * chis[
                    None,
                    :,
                ]
                ** 2,
                x=chis,
                axis=-1,
            ),
            x=chis,
            axis=-1,
        )

        return 4 * np.pi / (2 * l + 1) * z_ints

    auto_Cls = np.array(
        [Cl_1halo_pirvu(l, theta_max_array) + Cl_2halo(l, theta_max_array) for l in ls]
    )
    np.save(f"{output_dir}/Cl_PP_mA{mA:.3e}_nfw.npy", auto_Cls)

    def CgP_1h(l):
        return np.sqrt(4 * np.pi / (2 * l + 1)) * simpson(
            chis**2
            * simpson(
                n * ug(Ws, Ncs, Nss, l=l) * P * approx_ul0(l, theta_max_array),
                x=ms,
                axis=1,
            ),
            x=chis,
            axis=0,
        )

    def CgP_2h_limber(l):
        m_ints = simpson(
            P
            * approx_ul0(l, theta_max_array)
            * n
            * bias
            * simpson(n * bias * ug(Ws, Ncs, Nss, l=l), x=ms, axis=1)[:, None],
            x=ms,
            axis=1,
        )
        Plin_at_zs = np.array(
            [interp_lin_power((l + 0.5) / chi)[chis == chi] for chi in chis]
        ).reshape(zs.shape)
        return np.sqrt(4 * np.pi / (2 * l + 1)) * simpson(
            chis**2 * m_ints * Plin_at_zs, x=chis, axis=0
        )

    Cl_gP_1hs = np.array([CgP_1h(l) for l in ls])
    Cl_gP_2hs = np.array([CgP_2h_limber(l) for l in ls])

    np.save(f"{output_dir}/Cl_gP_mA{mA:.3e}_model{model}_nfw.npy", Cl_gP_1hs + Cl_gP_2hs)
    return None


mA_list = np.geomspace(1e-13, 1e-11, 25)
for mA in tqdm(mA_list):
    get_Cls(mA)
# Parallel(n_jobs=-1, verbose=10)(delayed(get_Cls)(mA) for mA in mA_list)
