#!/bin/bash -l

#$ -P darkcosmo
#$ -N halo_limits
#$ -m bae
#$ -M ebaker@bu.edu
#$ -j y
conda activate 21cmfastv4

python /projectnb/darkcosmo/dark_photon_project/dark_photons_radio_sky/halo_model/gal_cross_corr_halo_model.py