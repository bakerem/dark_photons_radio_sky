"""
Author: Ethan Baker, Boston University
Description: This file generates lightcones from one of the .toml files in this directory. 
This should be run from the command line with the specified inputs. 
This script is modified from 21cmFISH run script
Github: https://github.com/bakerem
"""
from click import progressbar
from matplotlib.pylab import f
import py21cmfast as py21c
import os
import glob
import numpy as np
import time
from joblib import Parallel, delayed
import argparse
import configparser
import multiprocessing
from astropy import units as un
from scipy.spatial.transform import Rotation
from astropy.cosmology import Planck18
import pickle

import logging

logger = logging.getLogger("21cmFAST")
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)
py21c.config["regenerate"] = False
py21c.config["HALO_CATALOG_MEM_FACTOR"] = 1.22

print(f"21cmFAST version is {py21c.__version__}")
print(py21c.config["regenerate"])
# ==============================================================================
# python make_lightcones_for_fisher.py ../21cmFAST_config_files/Park19.config --dry_run
# TODO =====
# Took ---- Finished making lightcones, took 15.86 hours ---- for ETHOS.
# Took 11 mins to make PS
#
# qdel
# python scripts/make_lightcones_for_fisher.py 21cmFAST_config_files/ETHOS.config --num_cores 2 --h_PEAK 0 --random_seed $r
# ==============================================================================
# ==============================================================================
#
# Script to create set of 21cmFAST simulations for Fisher matrix analysis.
#   Loads a configuration file of default parameters, and parameters to vary
#
# ==============================================================================
# ==============================================================================


# Managing arguments with argparse (see http://docs.python.org/howto/argparse.html)
parser = argparse.ArgumentParser()
# ---- required arguments ---- :
parser.add_argument("output_dir", type=str, help="Path to output directory")
# ---- optional arguments ----

parser.add_argument(
    "--config_file", type=str, default=None, help="Path to config file [default = None]"
)
parser.add_argument(
    "--random_seed", type=int, default=None, help="Random seed [default = 12345]"
)
# ---- flags ------
parser.add_argument(
    "--save_Tb", action="store_true", help="Save BrightnessTemp boxes [default = False]"
)
parser.add_argument(
    "--clobber", action="store_true", help="make new lightcones [default = False]"
)
parser.add_argument(
    "--angular_lightcone",
    action="store_true",
    default=True,
    help="Run code with an angular lightcone instead of linear lightcone [default = True]",
)

args = parser.parse_args()
# ==============================================================================
# Run Parameters
save_Tb = False
if args.save_Tb:
    save_Tb = True
    logger.info(f"Saving BrightnessTemp coeval boxes")

clobber = False
if args.clobber:
    clobber = True
    logger.info(f"Clobber = True - making new lightcones")

print(args.random_seed)
if args.random_seed is None:
    random_seed = np.random.randint(10000, 100000)
else:
    random_seed = args.random_seed
logger.info(f"Using random_seed = {random_seed}")

# ==============================================================================
# Get config
output_dir = args.output_dir + f"_seed_{random_seed}/"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
logger.info(f"Loading from cache at {output_dir}")
py21c.config["direc"] = output_dir

# --------------------------------------
lightcone_quantities = (
    "brightness_temp",
    "density",
    "xray_ionised_fraction",
    "neutral_fraction",
    "spin_temperature",
    "halo_mass",
    "halo_stars",
    "halo_stars_mini",
    "count",
    "halo_sfr",
    "halo_sfr_mini",
    "halo_xray",
    "n_ion",
    "whalo_sfr",
)

global_quantities = ("brightness_temp", "density", "spin_temperature")


# ==================================
# parameters

inputs = py21c.InputParameters.from_template(
    args.config_file,
    random_seed=random_seed,
)

py21c.utils.show_references(inputs)

# ==================================
min_redshift = np.min(inputs.node_redshifts)
max_redshift = np.max(inputs.node_redshifts)
HII_DIM = inputs.simulation_options.HII_DIM
BOX_LEN = inputs.simulation_options.BOX_LEN
dim = inputs.simulation_options.DIM
# Define grid for angular lightcone
cosmo = inputs.cosmo_params.cosmo
box_size_radians = BOX_LEN / cosmo.comoving_distance(min_redshift).value
lon = np.linspace(0, box_size_radians, dim)
lat = np.linspace(0, box_size_radians, dim)[::-1]
LON, LAT = np.meshgrid(lon, lat)
LON = LON.flatten()
LAT = LAT.flatten()
offset = cosmo.comoving_distance(min_redshift).to(
    un.pixel, un.pixel_scale((BOX_LEN / HII_DIM * un.Mpc) / un.pixel)
)
origin = np.array([0, 0, offset.value]) * offset.unit
rot = Rotation.from_euler("Y", -np.pi / 2)


logger.info(f"Making lightcone from z={min_redshift}-{max_redshift}")
logger.info(f"Box HII_DIM={HII_DIM}, BOX_LEN={BOX_LEN}")

# ==================================
# Define cache configuration
cache = py21c.OutputCache(output_dir)
# if save_Tb:
#     cache_config = py21c.CacheConfig.on()
# else:
#     cache_config = py21c.CacheConfig.noloop()
cache_config = py21c.CacheConfig.on()


# ==================================
# Find ICs and perturbed fields
PerturbedField_files = glob.glob(f"{output_dir}PerturbedField*")
IC_files = glob.glob(f"{output_dir}**/InitialConditions*", recursive=True)
print(IC_files)
if not IC_files:
    print("No initial conditions found")
    init_cond = None
else:
    init_cond = py21c.io.h5.read_output_struct(IC_files[0])
logger.info(f"Loaded or made initial conditions")
# Will not write more boxes
# py21c.config['write'] = False

# ==================================
# Create angular lightconer
logger.info(f"Making angular lightconer")
ang_lcn = py21c.AngularLightconer.with_equal_cdist_slices(
    min_redshift=min_redshift,
    max_redshift=max_redshift,
    quantities=(
        "brightness_temp",
        "density",
        "xray_ionised_fraction",
        "spin_temperature",
        "neutral_fraction",
        "halo_mass",
        "halo_stars",
        "halo_stars_mini",
        "count",
        "halo_sfr",
        "halo_sfr_mini",
        "halo_xray",
        "n_ion",
        "whalo_sfr",
    ),
    resolution=inputs.simulation_options.cell_size,
    latitude=LAT,
    longitude=LON,
    origin=-origin,
    rotation=rot,
    get_los_velocity=True,
)
ang_lcn_files = glob.glob(f"{output_dir}lightconer*")
with open(
    f"/projectnb/darkcosmo/dark_photon_project/21cmfast_cache/v4_lightcones/lightconer_seed{random_seed}.pkl",
    "wb",
) as fp:
    pickle.dump(ang_lcn, fp)
    logger.info("lightconer saved successfully to file")
# ==================================
# Run each filter


def make_lin_lightcone(astro_params_key):
    """
    Make lightcone for a given set of astroparams
    """

    # Save output for each parameter to a new directory
    # if save_Tb:
    output_dir = f"{output_dir}_{astro_params_key}"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # put PerturbedFields in output_dir
    if len(PerturbedField_files) > 0:
        for PF in PerturbedField_files:
            PF_file = PF.split("/")[-1]
            linked_file = f"{output_dir}/{PF_file}"
            if not os.path.exists(linked_file):
                os.symlink(PF, linked_file)

    for IC in IC_files:
        IC_file = IC.split("/")[-1]
        linked_file = f"{output_dir}/{IC_file}"
        if not os.path.exists(linked_file):
            os.symlink(IC, linked_file)
    direc = output_dir
    # else:
    #     direc = None

    # Lightcone filename
    suffix = f"HIIDIM={HII_DIM}_BOXLEN={BOX_LEN}"
    lightcone_filename = f"LightCone_z{min_redshift:.1f}_{suffix}_r{random_seed}.h5"
    logger.info(f"Will save lightcone to {lightcone_filename}")

    t1 = time.time()

    if not os.path.exists(f"{output_dir}{lightcone_filename}"):
        idx, z, coeval, lightcone = py21c.run_lightcone(
            redshift=min_redshift,
            max_redshift=max_redshift,
            lightcone_quantities=lightcone_quantities,
            global_quantities=global_quantities,
            inputs=inputs,
            random_seed=random_seed,
            cache=cache,
            write=cache_config,
            progressbar=True,
        )

        # save in main dir
        lightcone_save = lightcone.save(
            fname=lightcone_filename, direc=output_dir, clobber=True
        )
        logger.info(f"Saved lightcone to {lightcone_save}")
    else:
        logger.info(f"{lightcone_filename} already exists, skipping...")

    t2 = time.time()
    logger.info(f"Done with {astro_params_key}, took {(t2-t1)/3600:.2f} hours")

    return


def make_ang_lightcone():
    """
    Make lightcone for a given set of astroparams
    """

    # Save output for each parameter to a new directory
    # if save_Tb:
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # put PerturbedFields in output_dir
    if len(PerturbedField_files) > 0:
        for PF in PerturbedField_files:
            PF_file = PF.split("/")[-1]
            linked_file = f"{output_dir}/{PF_file}"
            if not os.path.exists(linked_file):
                os.symlink(PF, linked_file)

    for IC in IC_files:
        IC_file = IC.split("/")[-1]
        linked_file = f"{output_dir}{IC_file}"
        if not os.path.exists(linked_file):
            os.symlink(IC, linked_file)

    for ALN in ang_lcn_files:
        ALN_file = ALN.split("/")[-1]
        linked_file = f"{output_dir}/{ALN_file}"
        if not os.path.exists(linked_file):
            os.symlink(ALN, linked_file)

    # Lightcone filename
    suffix = f"HIIDIM={HII_DIM}_BOXLEN={BOX_LEN}"
    lightcone_filename = f"LightCone_z{min_redshift:.1f}_{suffix}_r{random_seed}.h5"
    logger.info(f"Will save lightcone to {lightcone_filename}")

    t1 = time.time()

    idx, z, coeval, lightcone = py21c.run_lightcone(
        lightconer=ang_lcn,
        global_quantities=global_quantities,
        inputs=inputs,
        write=cache_config,
        cache=cache,
        lightcone_filename=f"/projectnb/darkcosmo/dark_photon_project/21cmfast_cache/v4_lightcones/{lightcone_filename}",
        progressbar=True,
        initial_conditions=init_cond,
        regenerate=False,
    )
    # save in main dir
    lightcone_save = lightcone.save(
        f"/projectnb/darkcosmo/dark_photon_project/21cmfast_cache/v4_lightcones/{lightcone_filename}",
        clobber=True,
    )
    logger.info(f"Saved lightcone to {lightcone_save}")

    t2 = time.time()
    logger.info(f"Done with lightcone, took {(t2-t1)/3600:.2f} hours")

    return


t1 = time.time()

if args.angular_lightcone:
    make_lightcone = make_ang_lightcone
else:
    make_lightcone = make_lin_lightcone

make_lightcone()


t2 = time.time()
logger.info(f"---- Finished making lightcones, took {(t2-t1)/3600:.2f} hours")
