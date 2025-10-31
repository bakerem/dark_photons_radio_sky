#!/bin/bash -l

#$ -P darkcosmo
#$ -N pt_src_gen
#$ -m bae
#$ -M ebaker@bu.edu
#$ -pe omp 2
#$ -l h_rt=12:00:00
#$ -j y

conda activate 21cmfastv4

python gen_pt_srcs.py

