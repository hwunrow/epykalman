#!/bin/bash
#$ -l mem=20G,time=10:: -S /bin/bash -N epiestim
module load singularity
PASSWORD='ihme123' singularity exec --bind "$HOME"/run:/run,"$HOME"/var-lib-rstudio-server:/var/lib/rstudio-server,"$HOME"/database.conf:/etc/rstudio/database.conf,/ifs/scratch/jls106_gp/nhw2114:/ifs/scratch/jls106_gp/nhw2114 \
    "$HOME"/rstudio.simg Rscript --no-save --vanilla /ifs/scratch/jls106_gp/nhw2114/repos/rt-estimation/src/epyfilter/c2b2/epiestim_parallel.R \
    --in-dir /ifs/scratch/jls106_gp/nhw2114/repos/rt-estimation/src/epyfilter/c2b2/ --data-dir /ifs/scratch/jls106_gp/nhw2114/data/example_plots_for_paper/ --out-dir /ifs/scratch/jls106_gp/nhw2114/data/example_plots_for_paper/ \
    --param-list 1   2236   3348   3457   3775   5863   6115   8019 8298  16729  21258  27533  28485  36206  40299  43056 43775  45193  46482  49477  55685  55741  57066  58428 63679  66815  69090  70286  73071  84882  85084  89399 96475  96799  98225  99368 100184 100185
