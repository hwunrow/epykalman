#!/bin/bash
#$ -l mem=2G,time=0:30: -S /bin/bash -N eakf -t 1-7
#$ -e /ifs/scratch/jls106_gp/nhw2114/data/20240212_run
#$ -o /ifs/scratch/jls106_gp/nhw2114/data/20240212_run
source ~/.bashrc
conda activate epyfilter
python /ifs/scratch/jls106_gp/nhw2114/repos/rt-estimation/src/epyfilter/c2b2/run_eakf.py \
    --in-dir /ifs/scratch/jls106_gp/nhw2114/repos/rt-estimation/src/epyfilter/c2b2/ \
    --pkl-dir /ifs/scratch/jls106_gp/nhw2114/data/20231025_synthetic_data \
    --out-dir /ifs/scratch/jls106_gp/nhw2114/data/20240212_run \
    --param-file todo_params.csv \
    --save-reliability