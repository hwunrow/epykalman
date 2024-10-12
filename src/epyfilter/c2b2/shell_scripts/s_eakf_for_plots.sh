#!/bin/bash

#SBATCH --job-name=eakf_plots               # Job name
#SBATCH --array=1-5                         # Array job with tasks 1 through 5
#SBATCH --mem=20G                           # Memory per task
#SBATCH --time=5:00:00                      # Walltime limit (5 hours)
#SBATCH --output=/ifs/scratch/jls106_gp/nhw2114/data/example_plots_for_paper/%A_%a.out  # Output file (with array task ID)
#SBATCH --error=/ifs/scratch/jls106_gp/nhw2114/data/example_plots_for_paper/%A_%a.err   # Error file (with array task ID)

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