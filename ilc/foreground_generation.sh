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

TITLE=pt_src_test
nside=2048 # nside for the HEALPix map
max_flux=0.1 # maximum flux in Jy for the point sources
njobs=4 # number of jobs to run in parallel
fstart=140 # starting frequency in MHz
fstop=140 # stopping frequency in MHz
n_freqs=1 # number of frequency channels
gen_pt_srcs=true # whether to generate point sources or not
FWHM=0.5 # FWHM of the Gaussian smoothing in arcmin

# make directory to store results
results_dir=/projectnb/darkcosmo/dark_photon_project/21cmfast_cache/pyilc_${TITLE}/
cache_dir=/net/scc-ca5/scratch/ebaker_results/
mkdir ${results_dir}
if [ "${gen_pt_srcs}" == true ]; then
    echo "Generating foregrounds with point sources..."
    python /usr3/graduate/ebaker/dark_photon_constraints/ilc/generate_foregrounds.py --output_dir=${results_dir} --nside=${nside} --max_flux=${max_flux} --fstart=${fstart} --fstop=${fstop} --n_freqs=${n_freqs} --njobs=${njobs} --fwhm=${FWHM} --generate_pt_srcs 
else
    echo "Generating foregrounds without point sources..."
    python /usr3/graduate/ebaker/dark_photon_constraints/ilc/generate_foregrounds.py --output_dir=${results_dir} --nside=${nside} --max_flux=${max_flux} --fstart=${fstart} --fstop=${fstop} --n_freqs=${n_freqs} --njobs=${njobs} --fwhm=${FWHM}
fi
# Create a directory for cached information if it doesn't exist
# mkdir /net/scc-ca5/scratch/ebaker_results/

# run pyilc
# python /projectnb/darkcosmo/dark_photon_project/pyilc/pyilc/main.py /usr3/graduate/ebaker/dark_photon_constraints/ilc/pyilc_dark_photon_no_pt_srcs.yml


# # turn output map into mK
# cp ${cache_dir}needletILCmap_component_dp.fits ${results_dir}/needletILCmap_component_dp_${TITLE}.fits


# export results_dir

# python <<EOF
# import healpy as hp
# import os
# results_dir = os.environ['results_dir']
# map = hp.read_map(f'{results_dir}/needletILCmap_component_dp_{TITLE}.fits')/1000
# hp.write_map(f'{results_dir}/clean_needlet_ILC_map.fits', map, overwrite=True)
# EOF

# # clean up files
# rm -rf ${results_dir}/needletILCmap_component_dp_${TITLE}.fits
# rm -rf ${cache_dir}