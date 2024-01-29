#!/bin/bash
#$ -l mem=20G,time=10:: -S /bin/bash -N eakf
source ~/.bashrc
conda activate epyfilter
python /ifs/scratch/jls106_gp/nhw2114/repos/rt-estimation/src/epyfilter/c2b2/run_eakf.py \
    --in-dir /ifs/scratch/jls106_gp/nhw2114/repos/rt-estimation/src/epyfilter/c2b2/ \
    --pkl-dir /ifs/scratch/jls106_gp/nhw2114/data/example_plots_for_paper \
    --out-dir /ifs/scratch/jls106_gp/nhw2114/data/example_plots_for_paper \
    --param-list 1 3348 3775 5863 8019 8298 27533 28485 36206 40299 43056 43775 45193 46482 49477 55741 57066 66815 84882 89399