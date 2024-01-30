#!/bin/bash
#$ -l mem=20G,time=5:: -S /bin/bash -N eakf_plots
source ~/.bashrc
conda activate epyfilter
python /ifs/scratch/jls106_gp/nhw2114/repos/rt-estimation/src/epyfilter/c2b2/run_eakf.py \
    --in-dir /ifs/scratch/jls106_gp/nhw2114/repos/rt-estimation/src/epyfilter/c2b2/ \
    --pkl-dir /ifs/scratch/jls106_gp/nhw2114/data/20231025_synthetic_data \
    --out-dir /ifs/scratch/jls106_gp/nhw2114/data/example_plots_for_paper \
    --param-list 2 96956 60997 97638 4616 44331 99403 26094 57072 60945 1337 190 54589 32062