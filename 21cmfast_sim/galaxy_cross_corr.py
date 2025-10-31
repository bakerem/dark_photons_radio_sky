"""
Author: Ethan Baker, Boston University
Description: This module computes the auto and cross-correlation functions for 
21cmFAST simulations with galaxy surveys, specifically for the Roman telescope using pyILC cleaned maps.
It saves the results for different dark photon mass values (mA) to specified output files.
The user can specify the output directory and mass range. 
This also computes limits, but sometimes this fails because of numerical issues. In that case, just rerun the limits part
with a better numerical solver. No need to regenerate the correlation functions. 
Github: https://github.com/bakerem
"""
import os
import sys

sys.path.append("../")

import numpy as np
import matplotlib.pyplot as plt

from plot_params import params
from galaxy_survey import *

# Accept a default so the module can be imported without command-line args.
# This is useful if you want to rerun this multiple times in a batch script
jobid = sys.argv[1] if len(sys.argv) > 1 else "0"


# output directory
output = (
    "/projectnb/darkcosmo/dark_photon_project/21cmfast_cache/pyilc_alt_pt_src_0.1mJy_new_freq/limits"
)

# mA_list = np.geomspace(5e-15, 1e-13, 15)
mA_list = np.geomspace(1e-15, 1e-14, 10)
np.save(f"{output}/mA_list.npy", mA_list)

exists = [
    os.path.exists(
        f"{output}/21cmfast_roman_pyilc_limits{jobid}_mA_{mA:.3e}_obs_auto_xi.npy"
    )
    for mA in mA_list
]

if np.all(exists):
    print("All files already exist, skipping...")
    quit()

cache_name = f"/projectnb/darkcosmo/dark_photon_project/21cmfast_cache/v4_lightcones/"
lc_name = "LightCone_z5.5_HIIDIM=200_BOXLEN=300.0_r3843290498042390.h5"
# lightconer_name = "lightconer_seed12345.pkl"
lightconer_name = "lightconer_seed3843290498042390.pkl"
roman = Survey(
    "Roman",
    "Lya",
    cache=cache_name,
    lc_name=lc_name,
    lightconer_name=lightconer_name,
    nside=2048,
    foregrounds_path="/projectnb/darkcosmo/dark_photon_project/21cmfast_cache/pyilc_alt_pt_src_0.1mJy_new_freq/",
    clean_map_loc="/projectnb/darkcosmo/dark_photon_project/21cmfast_cache/pyilc_alt_pt_src_0.1mJy_new_freq/clean_needlet_ILC_map_roman_20deg_LOFARthermal_mK.fits",
    use_pixel_ilc=False,
    # include_analytic=True,
    # analytic_path="/projectnb/darkcosmo/dark_photon_project/21cmfast_cache/analytic_data",
)

results = []


def get_limit(mA):
    """Load diagnostics for ``mA`` if present, otherwise compute and save.

    Returns
    -------
    (auto, cross)
        ``auto`` is the value returned by ``roman.find_auto_limits`` when a
        computation was performed; ``cross`` is ``np.nan`` here (cross
        correlations are not computed by this script). If the diagnostics
        were already present, the function returns ``(None, None)`` to signal
        that no computation took place.
    """

    obs_auto_path = f"{output}/21cmfast_roman_pyilc_limits{jobid}_mA_{mA:.3e}_obs_auto_xi.npy"
    if os.path.exists(obs_auto_path):
        # diagnostics already saved for this mA; do not recompute here.
        print(f"Found saved diagnostics for mA={mA:.3e}, skipping computation")
        return None, None

    # compute and save diagnostics
    auto = roman.find_auto_limits(mA=mA)
    cross = np.nan

    np.save(
        f"{output}/21cmfast_roman_pyilc_limits{jobid}_mA_{mA:.3e}.npy",
        [auto, cross],
    )
    np.save(
        f"{output}/21cmfast_roman_pyilc_limits{jobid}_mA_{mA:.3e}_obs_auto_xi.npy",
        [roman.obs_auto_rnom, roman.obs_auto_xi],
    )
    np.save(
        f"{output}/21cmfast_roman_pyilc_limits{jobid}_mA_{mA:.3e}_pred_auto_xi.npy",
        [roman.pred_auto_rnom, roman.pred_auto_xi],
    )
    np.save(
        f"{output}/21cmfast_roman_pyilc_limits{jobid}_mA_{mA:.3e}_obs_auto_cov.npy",
        roman.obs_auto_cov,
    )

    return auto, cross


def _run_all():
    """Run get_limit over the configured mass list and collect results."""
    for mA in mA_list:
        res = get_limit(mA)
        if res is not None:
            results.append([mA, res[0], res[1]])


if __name__ == "__main__":
    _run_all()
