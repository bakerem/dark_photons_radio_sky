#!/bin/bash -l

#$ -P darkcosmo
#$ -N halo_limits
#$ -m bae
#$ -M ebaker@bu.edu
#$ -j y
conda activate 21cmfastv4

python /usr3/graduate/ebaker/dark_photon_constraints/halo_model/gal_cross_corr_correct.py