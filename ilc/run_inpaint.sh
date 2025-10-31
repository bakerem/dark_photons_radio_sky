#!/bin/bash -l

#$ -P darkcosmo
#$ -N inpaint
#$ -m bae
#$ -M ebaker@bu.edu
#$ -pe omp 1
#$ -l h_rt=12:00:00
#$ -j y

conda activate 21cmfastv4

python inpaint.py

