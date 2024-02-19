#!/bin/bash
#$ -l mem=10G,time=5:: -S /bin/bash -N pickle -t 1-48
#$ -e /ifs/scratch/jls106_gp/nhw2114/data/20240212_run
#$ -o /ifs/scratch/jls106_gp/nhw2114/data/20240212_run
source ~/.bashrc
conda activate epyfilter
python /ifs/scratch/jls106_gp/nhw2114/repos/rt-estimation/src/epyfilter/c2b2/open_pickle_rt.py \
    --in-dir /ifs/scratch/jls106_gp/nhw2114/repos/rt-estimation/src/epyfilter/c2b2/ \
    --pkl-dir /ifs/scratch/jls106_gp/nhw2114/data/20231025_synthetic_data \
    --out-dir /ifs/scratch/jls106_gp/nhw2114/data/20240212_run \
    --param-file good_param_list.csv