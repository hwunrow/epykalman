#!/bin/bash
#$ -l mem=5G,time=0:30: -S /bin/bash -N eakf -t 1-100
#$ -e /ifs/scratch/jls106_gp/nhw2114/data/20240226_metric_date_experiment
#$ -o /ifs/scratch/jls106_gp/nhw2114/data/20240226_metric_date_experiment
source ~/.bashrc
conda activate epyfilter
python /ifs/scratch/jls106_gp/nhw2114/repos/rt-estimation/src/epyfilter/c2b2/run_eakf.py \
    --in-dir /ifs/scratch/jls106_gp/nhw2114/repos/rt-estimation/src/epyfilter/c2b2/ \
    --pkl-dir /ifs/scratch/jls106_gp/nhw2114/data/20231025_synthetic_data \
    --out-dir /ifs/scratch/jls106_gp/nhw2114/data/20240226_metric_date_experiment \
    --param-file 20240226_metric_date_experiment_param_list.csv \
    --save-reliability \
    --n-real 500 \
    --save-data \