#!/bin/bash
#$ -l mem=10G,time=2:30: -S /bin/bash -N eakf -t 1-100
#$ -e /ifs/scratch/jls106_gp/nhw2114/data/20240226_metric_date_experiment/rankings
#$ -o /ifs/scratch/jls106_gp/nhw2114/data/20240226_metric_date_experiment/rankings
source ~/.bashrc
conda activate epyfilter
python /ifs/scratch/jls106_gp/nhw2114/repos/rt-estimation/src/epyfilter/c2b2/compute_metric_day_ranking.py \
    --in-dir /ifs/scratch/jls106_gp/nhw2114/repos/rt-estimation/src/epyfilter/c2b2/ \
    --pkl-dir /ifs/scratch/jls106_gp/nhw2114/data/20240226_metric_date_experiment \
    --out-dir /ifs/scratch/jls106_gp/nhw2114/data/20240226_metric_date_experiment/rankings \
    --param-file 20240226_metric_date_experiment_param_list.csv \
    --n-real 300 \
    --n-ens 300