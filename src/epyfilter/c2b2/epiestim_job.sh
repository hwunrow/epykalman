#!/bin/bash
#$ -l mem=5G,time=1:: -S /bin/bash -N pickle
run_file="$1"; shift
module load R/4.2.2
R CMD BATCH --no-save --vanilla $run_file routput --args $@
# qsub -t 1-2 epiestim_job.sh /ifs/scratch/jls106_gp/nhw2114/repos/rt-estimation/src/epyfilter/c2b2/epiestim_parallel.R --in-dir /ifs/scratch/jls106_gp/nhw2114/repos/rt-estimation/src/epyfilter/c2b2/ --out-dir /ifs/scratch/jls106_gp/nhw2114/data/20231025_synthetic_data