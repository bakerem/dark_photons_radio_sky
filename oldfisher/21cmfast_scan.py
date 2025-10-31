import os
import sys
sys.path.append("../")

import numpy as np
from astropy.cosmology import Planck18
import py21cmfast as p21c
from astropy import units as un
from powerbox.tools import get_power


# WDIR = os.environ['DM21CM_DIR']
# sys.path.append(WDIR)
# from dm21cm.evolve import evolve

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import colormaps as cms
# mpl.rc_file(f"{WDIR}/matplotlibrc")

import physics as phys
import pickle
import h5py

from astropy.cosmology import z_at_value
import tqdm

import scipy.interpolate as interpolate


import mcfit

import treecorr


# =========================================================================
# Initialize Lightcone 
# =========================================================================

run_name = 'lightcones.h5'
cache_name = '/home/bakerem/21cmfast_cache/low_res_full'

lofar_data = np.genfromtxt(f"halo_data/lofar_data.csv", delimiter=",").T
hera_79_data = np.genfromtxt(f"halo_data/hera79_data.csv", delimiter=",").T
hera_101_data = np.genfromtxt(f"halo_data/hera_101_data.csv", delimiter=",").T

z_start = 45
z_end = 5
rs_max = 50

lc_file  =  h5py.File(f'{cache_name}/{run_name}', 'r')
    # Access data within the file
lc = lc_file["lightcones"]
 
with open(f'{cache_name}/lightconer.pkl', 'rb') as f:
    ang_lcn = pickle.load(f)
user_params = dict(lc_file["user_params"].attrs)
cosmo_params = p21c.CosmoParams(
        OMm = Planck18.Om0,
        OMb = Planck18.Ob0,
        POWER_INDEX = Planck18.meta['n'],
        SIGMA_8 = Planck18.meta['sigma8'],
        hlittle = Planck18.h,
)
cosmo = cosmo_params.cosmo

box_len = user_params["BOX_LEN"] 
hii_dim = user_params["HII_DIM"] 
rs_dim = np.array(lc["brightness_temp"]).shape[-1]


# plots the initial lightcone (before any conversions) and creates an array of
# redshifts for use later

bt = np.flip(np.array(lc["brightness_temp"]).reshape((hii_dim, hii_dim, rs_dim)), axis=2)

box_size_radians = user_params["BOX_LEN"] / cosmo.comoving_distance(z_end).value
lon = np.linspace(0, box_size_radians, hii_dim)
lat = np.linspace(0, box_size_radians, hii_dim)[::-1]  # This makes the X-values increasing from 0.
H, D = np.meshgrid(lat, ang_lcn.lc_distances.value)

rgrid = np.arange(0, D.max(), user_params["BOX_LEN"])
rgrid = rgrid[rgrid > D.min()]
rgrid = np.append(rgrid, D.max())
rs_array = np.flipud([z_at_value(cosmo.comoving_distance, d*un.Mpc).value for d in ang_lcn.lc_distances.value])

Tgamma0 = phys.TCMB(1) * 1000 / phys.kB #mK
omega0 = 5.904e-6 * (2*np.pi) # eV


# process and reshape data from the lightcone for further use
xe_box = np.flip(np.array(lc['x_e_box']).reshape((hii_dim, hii_dim, rs_dim)), axis=2)
density_box = np.flip(np.array(lc["density"]).reshape((hii_dim , hii_dim, rs_dim)), axis=2)
# Ts_box =  np.flip(np.array(lc["Ts_box"]).reshape((hii_dim , hii_dim, rs_dim)), axis=2)
n_e = xe_box *  phys.nB * (1+rs_array)**3 * (1+density_box) * (1-3*phys.YHe/4)

# calculate plasma mass and related quantities that are relevant for conversions
mgamma = phys.m_gamma(n_e)
mgamma2 = mgamma**2
logm_gamma2 = np.log(mgamma**2)
dlogmgamma2dz = np.abs(np.diff(logm_gamma2, axis=2, prepend=0)/np.diff(rs_array, prepend=1))
# tau21 = 9.85e-3 * phys.TCMB(rs_array) / phys.kB / Ts_box  * phys.omega_baryon * phys.h / 0.0327 * (phys.omega_m / 0.307)**-0.5 * ((1+rs_array)/10)**0.5 
# tau21[..., rs_array > rs_max] = np.zeros(xe_box[...,rs_array > rs_max].shape)
xobs = 0.0251 /(1+rs_array) 
xobs_array = xobs * np.ones((hii_dim, hii_dim, rs_dim))


# initialize empty arrays 


# brightness temperature array that will be modified by the conversions

# loop over redshifts and compute where conversions occur
def run_for_mA(mA, mode):
    if mA >= 1.32e-13:
        mA_string = f"{mA:.3e}"
        halo_xi = np.load(f'halo_data/mccarthy_xis/halo_xi_mA_{mA_string}_l_max35000.npy')
        # print(halo_xi)
    else:
        halo_xi = np.zeros_like(np.load(f'halo_data/mccarthy_xis/halo_xi_mA_1.000e-13_l_max35000.npy'))

    bt_w_convs = np.flip(np.array(lc['brightness_temp']).reshape((hii_dim , hii_dim, rs_dim)), axis=2)
    Pgammatot = Pgamma_i = num_crossings = np.zeros((hii_dim, hii_dim, rs_dim))
    num_crossings = np.zeros((hii_dim, hii_dim, rs_dim), dtype=int)
    cut_rs_dim = rs_array[rs_array<rs_max].shape[0]
    for k, z in enumerate(rs_array):
        if z<rs_array[1]:
            crossed_cond_array = np.where(((mgamma[...,k-1] < mA) & (mgamma[...,k] >= mA)) | ((mgamma[...,k-1] > mA) & (mgamma[...,k] <= mA)),1,0)
            num_crossings[...,k] = crossed_cond_array
            # indexed backwards so :k means every redshift above this zi
            Pgamma_i[...,k] = np.pi * mA**2 / ( phys.TCMB(1) * phys.hubble(1+z) * 6.57895e-16 * (1+z)**2) * dlogmgamma2dz[...,k]**(-1) * crossed_cond_array 
    total_num_crossings = np.cumsum(num_crossings, axis=2)


    Ptot = np.sum(Pgamma_i[...,rs_array<rs_max])
    Pgammatot = np.tile(np.sum(Pgamma_i[...,rs_array<rs_max], axis=2)[:,:,None], (1, 1, cut_rs_dim)) / (xobs_array[...,rs_array<rs_max])
    # Pgamma_fromzobs = np.cumsum(Pgamma_i[...,rs_array<rs_max], axis=2) / xobs_array[...,rs_array<rs_max] 
    # Pgamma0_tozobs = np.cumsum(np.flip(Pgamma_i[...,rs_array<rs_max], axis=2), axis=2) / xobs_array[...,rs_array<rs_max] 
    # Pgamma0_tozobs = np.cumsum(Pgamma_i[...,::-1], axis=2)[...,::-1] / xobs_array[...] 


    # add these probabilities to the brightness temperature
    # bt_w_convs[...,rs_array<rs_max] += Tgamma0 * Pgammatot * tau21[...,rs_array<rs_max]
    # bt_w_convs[...,rs_array<rs_max] += Tgamma0 * Pgamma_fromzobs * Pgamma0_tozobs * np.exp(-tau21[...,rs_array<rs_max])
    # bt_w_convs[...,rs_array<rs_max] -= bt[...,rs_array<rs_max] * Pgamma0_tozobs
    bt_w_convs[...,rs_array<rs_max] -= Tgamma0 * Pgammatot

    nchunks = 40
    T21output, chunk_indices = powerspectra(bt, n_psbins=30, nchunks=nchunks, min_k=0.1, max_k=1.0, logk=True)
    output, chunk_indices = powerspectra(Tgamma0*Pgammatot, n_psbins=30, nchunks=nchunks, min_k=0.1, max_k=1.0, logk=True)

    chunk_Pks = []
    halo_Pdeltas = []

    halo_theta = np.geomspace(1e-4, 0.2, 100)
    for i, index in enumerate(chunk_indices[:-1]):
        chunk_dists = np.flipud(ang_lcn.lc_distances)[index:chunk_indices[i+1]] 
        ave_z = (rs_array[index]+rs_array[chunk_indices[i+1]])/2

        halo_rs = np.average(chunk_dists.value) * halo_theta

        dz = box_len / hii_dim
        
        xs = np.geomspace(np.min(halo_rs), np.max(halo_rs), 100)
        interpolated_xi = interpolate.interp1d(halo_rs, halo_xi)
        Pks, Perp_P = mcfit.w2C(xs, nu=0, lowring=True)(interpolated_xi(xs),)
        Perp_P *= dz * (Tgamma0 / omega0 * (1+ave_z))**2
        circ_P = np.pi * Pks / dz * Perp_P
        

        if np.all(halo_xi == 0):
            halo_delta = np.zeros_like(xs)
        else:
            halo_delta = Pks**3 * circ_P / (2*np.pi**2)
        # if i == 5:
        #     plt.plot(Pks, halo_delta)
        #     plt.show()
        chunk_Pks.append(Pks)
        halo_Pdeltas.append(halo_delta)
        z_list = rs_array[chunk_indices]

        if 10.4 < z_list[i] and 10.4 > z_list[i+1]:
            if mode=="data":
                k_comp = hera_101_data[0]*0.67
                data_comp = hera_101_data[1]
            elif mode=="rough forecast":
                k_comp = T21output[i]["k"]
                data_comp = T21output[i]["delta"]
            halo_delta_at_data = interpolate.interp1d(chunk_Pks[i], halo_Pdeltas[i], fill_value=0, bounds_error=False)(k_comp)
            sim_delta_at_data = interpolate.interp1d(output[i]["k"], output[i]["delta"], fill_value=0, bounds_error=False)(k_comp)
            LCDM_delta_at_data = interpolate.interp1d(T21output[i]["k"], T21output[i]["delta"], fill_value=0, bounds_error=False)(k_comp)
            hera_101_epsilon_lim = find_lims(sim_delta_at_data, 0, data_comp)
        elif i != rs_array.shape[0]-1 and 9.1 < z_list[i] and 9.1 > z_list[i+1]:
            if mode=="data":
                k_comp = lofar_data[0]*0.67
                data_comp = lofar_data[1]
            elif mode=="rough forecast":
                k_comp = T21output[i]["k"]
                data_comp = T21output[i]["delta"]
            halo_delta_at_data = interpolate.interp1d(chunk_Pks[i], halo_Pdeltas[i], fill_value=0, bounds_error=False)(k_comp)
            sim_delta_at_data = interpolate.interp1d(output[i]["k"], output[i]["delta"], fill_value=0, bounds_error=False)(k_comp)
            LCDM_delta_at_data = interpolate.interp1d(T21output[i]["k"], T21output[i]["delta"], fill_value=0, bounds_error=False)(k_comp)
            lofar_epsilon_lim = find_lims(sim_delta_at_data, 0, data_comp)
        elif i != rs_array.shape[0]-1 and 7.9 < z_list[i] and 7.9 > z_list[i+1]:
            if mode=="data":
                k_comp = hera_79_data[0]*0.67
                data_comp = hera_79_data[1]
            elif mode=="rough forecast":
                k_comp = T21output[i]["k"]
                data_comp = T21output[i]["delta"]
            halo_delta_at_data = interpolate.interp1d(chunk_Pks[i], halo_Pdeltas[i], fill_value=0, bounds_error=False)(k_comp)
            sim_delta_at_data = interpolate.interp1d(output[i]["k"], output[i]["delta"], fill_value=0, bounds_error=False)(k_comp)
            LCDM_delta_at_data = interpolate.interp1d(T21output[i]["k"], T21output[i]["delta"], fill_value=0, bounds_error=False)(k_comp)
            hera_79_epsilon_lim = find_lims(sim_delta_at_data, 0, data_comp)
    return np.array([mA, hera_79_epsilon_lim, lofar_epsilon_lim, hera_101_epsilon_lim])

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]

def compute_power(
box,
length,
n_psbins,
log_bins=True,
ignore_kperp_zero=True,
ignore_kpar_zero=False,
ignore_k_zero=False,
):
    """
    Convenience function for computing the power spectrum of a 3D box that wraps get_power from powerbox.  This code is borrowed from the example
    notebook in the 21cmFAST documentation. get_power takes a 3D box and Fourier transforms it and then performs a spherical average in $k$-space. 
    """
    # Determine the weighting function required from ignoring k's.
    k_weights = np.ones(box.shape, dtype=int)
    n0 = k_weights.shape[0]
    n1 = k_weights.shape[-1]

    if ignore_kperp_zero:
        k_weights[n0 // 2, n0 // 2, :] = 0
    if ignore_kpar_zero:
        k_weights[:, :, n1 // 2] = 0
    if ignore_k_zero:
        k_weights[n0 // 2, n0 // 2, n1 // 2] = 0

    res = get_power(
        box,
        boxlength=length,
        bins=n_psbins,
        bin_ave=False,
        get_variance=False,
        log_bins=log_bins,
        k_weights=k_weights,
        # bins_upto_boxlen=True,
    )

    res = list(res)
    k = res[1]
    if log_bins:
        k = np.exp((np.log(k[1:]) + np.log(k[:-1])) / 2)
    else:
        k = (k[1:] + k[:-1]) / 2

    res[1] = k
    return res

def powerspectra(brightness_temp, n_psbins=50, nchunks=10, min_k=0.1, max_k=1.0, logk=True):
    """
    This function wraps compute_power to compute the power spectrum of many chunks of a single
    lightcone and returns the dimensionless power spectrum. 
    """
    data = []
    n_slices = rs_array.shape[0]
    chunk_indices = list(range(0,n_slices,round(n_slices / nchunks)))

    if len(chunk_indices) > nchunks:
        chunk_indices = chunk_indices[:-1]
    chunk_indices.append(n_slices-1)

    for i in range(nchunks):
        start = chunk_indices[i]
        end = chunk_indices[i + 1]
        cell_size = box_len / hii_dim
        chunklen = (end - start) * cell_size
        comoving_size = np.max(lat) * cosmo.comoving_distance(np.average([rs_array[start], rs_array[end-1]])).value
        power, k = compute_power(
            brightness_temp[:, :, start:end],
            (comoving_size, comoving_size, chunklen),
            n_psbins,
            log_bins=logk,
        )
        data.append({"k": k, "P": power, "delta": k**3 * power/ (2*np.pi**2)})
    return data, chunk_indices

def get_xi(field, chunk_indices, i, index, lat_array, lon_array):
    chunk = field[...,index:chunk_indices[i+1]]
    chunk_dists = np.flipud(ang_lcn.lc_distances)[index:chunk_indices[i+1]] * np.ones_like(chunk)
    lon_array_tile = np.tile(lon_array[:,:,None], (1,1,chunk.shape[-1]))
    lat_array_tile = np.tile(lat_array[:,:,None], (1,1,chunk.shape[-1]))
    cat = treecorr.Catalog(ra=lon_array_tile.flatten(),
                            dec=lat_array_tile.flatten(),
                            r= chunk_dists.flatten(),
                            k = chunk.flatten(),
                            ra_units='rad',
                            dec_units='rad')
    kk = treecorr.KKCorrelation(nbins=12, 
                                min_sep=np.min(np.abs(np.diff(ang_lcn.lc_distances.value))), 
                                max_sep=ang_lcn.lc_distances.value[index]*np.max(lon_array)*np.sqrt(2))
    kk.process(cat)
    rnom = kk.rnom
    xi = kk.xi
    return rnom, xi, chunk_dists, kk


def find_lims(simulation_delta, lambda_cdm_delta, data_delta):
    epsilon_list = np.geomspace(1e-12, 1e-4, 10000)
    for epsilon in epsilon_list:
        delta_with_eps = epsilon**4 * simulation_delta + lambda_cdm_delta
        if np.any(delta_with_eps > data_delta):
            return epsilon
    return 1000

sim_mA_list = np.geomspace(1.5e-14, 1e-13, 30)
halo_mA_list = np.geomspace(1e-13, 1e-11, 20) 
# halo_mA_list = np.geomspace(1e-13, 1e-11, 10)
mA_list = np.concatenate((sim_mA_list, halo_mA_list,))
output_array = []
for mA in tqdm.tqdm(mA_list):
    output_array.append(run_for_mA(mA, mode="data"))
output_array = np.array(output_array)
np.save("halo_data/sim_data_epsilon_mA_grid.npy", output_array, allow_pickle=True)


