#!/bin/bash
#$ -l mem=1G,time=1:: -S /bin/bash -N synthetic_data

source ~/.bashrc
conda activate epyfilter
python /ifs/scratch/jls106_gp/nhw2114/repos/rt-estimation/src/epyfilter/c2b2/generate_data.py \
    --in-dir /ifs/scratch/jls106_gp/nhw2114/repos/rt-estimation/src/epyfilter/c2b2/ \
    --out-dir /ifs/scratch/jls106_gp/nhw2114/data/example_plots_for_paper \
    --param-list 2236 3457 6115 16729 21258 55685 58428 63679 69090  70286  73071  85084  96475  96799  98225  99368 100184 100185  21258  55685  58428  63679  69090  70286 73071  85084  96475  96799  98225  99368 100184 100185 46482  66815  43056   5863      1   3348  45193   8019 8298   3775  27533  28485  40299  84882  89399  57066 49477  21258  43775  55741  36206 \