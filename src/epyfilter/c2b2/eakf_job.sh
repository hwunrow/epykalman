#!/bin/bash
#$ -l mem=5G,time=5:: -S /bin/bash -N eakf
source ~/.bashrc
conda activate epyfilter
python /ifs/scratch/jls106_gp/nhw2114/repos/rt-estimation/src/epyfilter/c2b2/run_eakf.py \
    --in-dir /ifs/scratch/jls106_gp/nhw2114/repos/rt-estimation/src/epyfilter/c2b2/ \
    --out-dir /ifs/scratch/jls106_gp/nhw2114/data/20231025_synthetic_data \
    --param-list 688 1465 1553 18031 27165 29653