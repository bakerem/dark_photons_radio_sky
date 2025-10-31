import sys, os
import subprocess
import numpy as np
from tqdm import tqdm

input_files = [
    "/projectnb/darkcosmo/dark_photon_project/21cmfast_cache/pyilc_planck_small_beam/foregrounds_nu30.000_Planckthermal_KCMB.fits",
    "/projectnb/darkcosmo/dark_photon_project/21cmfast_cache/pyilc_planck_small_beam/foregrounds_nu44.000_Planckthermal_KCMB.fits",
    "/projectnb/darkcosmo/dark_photon_project/21cmfast_cache/pyilc_planck_small_beam/foregrounds_nu70.000_Planckthermal_KCMB.fits",
    "/projectnb/darkcosmo/dark_photon_project/21cmfast_cache/pyilc_planck_small_beam/foregrounds_nu100.000_Planckthermal_KCMB.fits",
    "/projectnb/darkcosmo/dark_photon_project/21cmfast_cache/pyilc_planck_small_beam/foregrounds_nu143.000_Planckthermal_KCMB.fits",
    "/projectnb/darkcosmo/dark_photon_project/21cmfast_cache/pyilc_planck_small_beam/foregrounds_nu217.000_Planckthermal_KCMB.fits",
    "/projectnb/darkcosmo/dark_photon_project/21cmfast_cache/pyilc_planck_small_beam/foregrounds_nu353.000_Planckthermal_KCMB.fits",
    "/projectnb/darkcosmo/dark_photon_project/21cmfast_cache/pyilc_planck_small_beam/foregrounds_nu545.000_Planckthermal_KCMB.fits",
]


mask = (
    "/usr3/graduate/ebaker/dark_photon_constraints/ilc/pyilc_files/mccarthy_mask.fits"
)

for file in input_files:
    if not os.path.exists(file.replace(".fits", "_inpainted.fits")):
        print("Inpainting ", file)
        subprocess.run(
            [
                "python",
                "/projectnb/darkcosmo/dark_photon_project/pyilc/diffusive_inpaint/diffusive_inpaint_example.py",
                mask,
                file,
                file.replace(".fits", "_inpainted.fits"),
            ]
        )
    else:
        print(
            "File ",
            file.replace(".fits", "_inpainted.fits"),
            " already exists, skipping...",
        )
