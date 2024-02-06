#!/bin/bash
#$ -l mem=20G,time=10:: -S /bin/bash -N epiestim
module load singularity
PASSWORD='ihme123' singularity exec --bind "$HOME"/run:/run,"$HOME"/var-lib-rstudio-server:/var/lib/rstudio-server,"$HOME"/database.conf:/etc/rstudio/database.conf,/ifs/scratch/jls106_gp/nhw2114:/ifs/scratch/jls106_gp/nhw2114 \
    "$HOME"/rstudio.simg Rscript --no-save --vanilla /ifs/scratch/jls106_gp/nhw2114/repos/rt-estimation/src/epyfilter/c2b2/epiestim_parallel.R \
    --in-dir /ifs/scratch/jls106_gp/nhw2114/repos/rt-estimation/src/epyfilter/c2b2/ --data-dir /ifs/scratch/jls106_gp/nhw2114/data/example_plots_for_paper/ --out-dir /ifs/scratch/jls106_gp/nhw2114/data/example_plots_for_paper/ \
    --param-list 2 96956 60997 97638 4616 44331 99403 26094 57072 60945 1337 190 54589 32062