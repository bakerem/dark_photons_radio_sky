# 21cm_dark_photon_constraints

The code in this repository was used in arXiv:_____ to compute the sensitivity of radio experiments to $\gamma \to A'$ conversions. 

The code can be broken down into a few different pieces, each of which is relevant for a different part of the analysis.

Throughout the code, we save power spectra and correlation functions for a dark photon signal with default $\epsilon=1$ and $\omega_0 = 1$ eV and do not include an factors of $T_{\gamma,0}$. 

## 21cmFAST Simulations

The files in the `21cmfast_sim` directory are used to run and process 21cmFAST simulations. 

`make_lightconesv4.py` is used to run a 21cmFASTv4 simulation with a provided configuration file. The configuration files used in this work are `dark_phot_params.toml` and `small_dark_photon.toml`. They are the same except `small_dark_photon.toml` runs a smaller simulation and is useful for some basic testing. 

Then, these files are best analyzed with the `Survey` class in `galaxy_survey.py`. This class allows the user to create a mock galaxy survey from the lightcone according to the parameters in our paper. Our fiducial survey is the *Roman* High Latitude Survey, although new surveys can easily be defined. 

The class is instantiated with the path to a lightcone, the name of a galaxy survey, and several details about foreground maps. These are the maps that are used in the analysis. The foreground maps can also be generated from scratch using the parameters provided in an optional dictionary during instantiation. This isn't recommended however, because it can take a very long time to generate foregrounds. There is an option to use either a user-provided post-ilc map.

Once the `Survey` class is instantiated, a galaxy catalog can be created, the dark photon signal computed for a range of masses, and auto- and cross-correlations performed. Several helper functions are also provided for manipulating maps. More information about all of these procedures is provided in the class docstrings. 

An example of using the `Survey` class to compute the expected sensitivity to dark photons is provided in the `galaxy_cross_corr.ipynb` notebook. 

## Halo Model

Code used to generate the auto-correlation of the $\gamma \to A'$ conversions in the halo model and cross-correlations between this signal and the *unWise* galaxy survey is provided in the `halo_model` directory. The most important file is `halo_gal_cross_correlation.ipynb`, which computes the auto- and cross-power spectra for any $m_A'$. The `halo_gal_cross_correlation_mccarthy.ipynb` notebook generates these power spectra to replicate the results in [McCarthy et al., 2024](https://arxiv.org/abs/2406.02546) and [Pîrvu et al., 2023](https://arxiv.org/abs/2307.15124). The python and bash scripts in the directory contain the same functions as the notebooks, but allow the user to more easily run and save the power spectra for different dark photon masses. 

## ILC

The code used to perform the ILC algorithm is in the `ilc` directory. `generate_foregrounds.py` is a script to generate the mock foreground maps that we use and `foreground_generation.sh` is an example bash script of how to run this file. The user can either produce mock SKA-Mid maps, mock *Planck* maps, or maps at custom frequencies. The point source generation takes a long time and generates very large files, so beware! 

The subdirectory `pyilc_files` contains several example configuration files for running `pyilc`. These should be run on a cluster of some sort typically, and modified with the paths where the foreground maps are saved. There are also various masks saved in this directory, including the mask we use in this work, a 20 degree mask on the ELIAS-N1 field, and the mask used in the McCarthy et al. analysis from [here](https://users.flatironinstitute.org/~fmccarthy/dark_photon_screening_maps/) (Note: this link appears to be broken.)

## Analytic Computation

Finally, notebooks and scripts to generate analytic dark photon auto-power spectra and correlation functions are provided in the `analytics` directory. These work pretty much the same way as the halo model code. This code is adapted heavily from [this repository](https://github.com/smsharma/dark-photons-perturbations/). 

## Plots from arXiv:____# 21cm_dark_photon_constraints

The code in this repository was used in arXiv:_____ to compute the sensitivity of radio experiments to $\gamma \to A'$ conversions. 

The code can be broken down into a few different pieces, each of which is relevant for a different part of the analysis.

Throughout the code, we save power spectra and correlation functions for a dark photon signal with default $\epsilon=1$ and $\omega_0 = 1$ eV and do not include an factors of $T_{\gamma,0}$. 

## 21cmFAST Simulations

The files in the `21cmfast_sim` directory are used to run and process 21cmFAST simulations. 

`make_lightconesv4.py` is used to run a 21cmFASTv4 simulation with a provided configuration file. The configuration files used in this work are `dark_phot_params.toml` and `small_dark_photon.toml`. They are the same except `small_dark_photon.toml` runs a smaller simulation and is useful for some basic testing. 

Then, these files are best analyzed with the `Survey` class in `galaxy_survey.py`. This class allows the user to create a mock galaxy survey from the lightcone according to the parameters in our paper. Our fiducial survey is the *Roman* High Latitude Survey, although new surveys can easily be defined. 

The class is instantiated with the path to a lightcone, the name of a galaxy survey, and several details about foreground maps. These are the maps that are used in the analysis. The foreground maps can also be generated from scratch using the parameters provided in an optional dictionary during instantiation. This isn't recommended however, because it can take a very long time to generate foregrounds. There is an option to use either a user-provided post-ilc map.

Once the `Survey` class is instantiated, a galaxy catalog can be created, the dark photon signal computed for a range of masses, and auto- and cross-correlations performed. Several helper functions are also provided for manipulating maps. More information about all of these procedures is provided in the class docstrings. 

An example of using the `Survey` class to compute the expected sensitivity to dark photons is provided in the `galaxy_cross_corr.ipynb` notebook. 

## Halo Model

Code used to generate the auto-correlation of the $\gamma \to A'$ conversions in the halo model and cross-correlations between this signal and the *unWise* galaxy survey is provided in the `halo_model` directory. The most important file is `halo_gal_cross_correlation.ipynb`, which computes the auto- and cross-power spectra for any $m_A'$. The `halo_gal_cross_correlation_mccarthy.ipynb` notebook generates these power spectra to replicate the results in [McCarthy et al., 2024](https://arxiv.org/abs/2406.02546) and [Pîrvu et al., 2023](https://arxiv.org/abs/2307.15124). The python and bash scripts in the directory contain the same functions as the notebooks, but allow the user to more easily run and save the power spectra for different dark photon masses. 

## ILC

The code used to perform the ILC algorithm is in the `ilc` directory. `generate_foregrounds.py` is a script to generate the mock foreground maps that we use and `foreground_generation.sh` is an example bash script of how to run this file. The user can either produce mock SKA-Mid maps, mock *Planck* maps, or maps at custom frequencies. The point source generation takes a long time and generates very large files, so beware! 

The subdirectory `pyilc_files` contains several example configuration files for running `pyilc`. These should be run on a cluster of some sort typically, and modified with the paths where the foreground maps are saved. There are also various masks saved in this directory, including the mask we use in this work, a 20 degree mask on the ELIAS-N1 field, and the mask used in the McCarthy et al. analysis from [here](https://users.flatironinstitute.org/~fmccarthy/dark_photon_screening_maps/) (Note: this link appears to be broken.)

## Analytic Computation

Finally, notebooks and scripts to generate analytic dark photon auto-power spectra and correlation functions are provided in the `analytics` directory. These work pretty much the same way as the halo model code. This code is adapted heavily from [this repository](https://github.com/smsharma/dark-photons-perturbations/). 

## Plots from arXiv:____

The notebooks used to generate the plots in this work are provided in the `notebooks_for_paper` directory. 

The notebooks used to generate the plots in this work are provided in the `notebooks_for_paper` directory. 