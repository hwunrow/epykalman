#!/bin/bash
#$ -l mem=20G,time=5:: -S /bin/bash -N eakf_plots -t 1-5
#$ -e /ifs/scratch/jls106_gp/nhw2114/data/example_plots_for_paper
#$ -o /ifs/scratch/jls106_gp/nhw2114/data/example_plots_for_paper
source ~/.bashrc
conda activate epyfilter
python /ifs/scratch/jls106_gp/nhw2114/repos/rt-estimation/src/epyfilter/c2b2/run_eakf.py \
    --in-dir /ifs/scratch/jls106_gp/nhw2114/repos/rt-estimation/src/epyfilter/c2b2/ \
    --pkl-dir /ifs/scratch/jls106_gp/nhw2114/data/20231025_synthetic_data \
    --out-dir /ifs/scratch/jls106_gp/nhw2114/data/example_plots_for_paper \
    --save-data \
    --save-reliability \
    --param-file /ifs/scratch/jls106_gp/nhw2114/data/example_plots_for_paper/plot_params.csv \
    --files-per-task 10