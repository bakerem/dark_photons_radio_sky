#!/bin/bash -l

#$ -P darkcosmo
#$ -N foreground_generation
#$ -m bae
#$ -M ebaker@bu.edu
#$ -pe omp 8 
#$ -l mem_per_core=16G
#$ -l h_rt=12:00:00
#$ -j y

conda activate 21cmfastv4

TITLE=planck
nside=2048 # nside for the HEALPix map
max_flux=0.1 # maximum flux in Jy for the point sources
njobs=4 # number of jobs to run in parallel
gen_pt_srcs=true # whether to generate point sources or not

# make directory to store results
results_dir=/projectnb/darkcosmo/dark_photon_project/21cmfast_cache/pyilc_${TITLE}/
cache_dir=/net/scc-ca5/scratch/ebaker_results/
mkdir ${results_dir}
if [ "${gen_pt_srcs}" == true ]; then
    echo "Generating foregrounds with point sources..."
    python /usr3/graduate/ebaker/dark_photon_constraints/ilc/generate_foregrounds.py --output_dir=${results_dir} --nside=${nside} --max_flux=${max_flux} --fstart=${fstart} --fstop=${fstop} --n_freqs=${n_freqs} --njobs=${njobs} --fwhm=${FWHM} --generate_pt_srcs --planck 
else
    echo "Generating foregrounds without point sources..."
    python /usr3/graduate/ebaker/dark_photon_constraints/ilc/generate_foregrounds.py --output_dir=${results_dir} --nside=${nside} --max_flux=${max_flux} --fstart=${fstart} --fstop=${fstop} --n_freqs=${n_freqs} --njobs=${njobs} --fwhm=${FWHM} --planck
fi