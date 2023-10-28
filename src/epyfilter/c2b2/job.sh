#!/bin/bash
#$ -l mem=1G,time=1:: -S /bin/bash -N synthetic_data -cwd
source ~/.bashrc
conda activate epyfilter
python /ifs/scratch/jls106_gp/nhw2114/repos/rt-estimation/src/epyfilter/c2b2/generate_data.py \
    --in-dir /ifs/scratch/jls106_gp/nhw2114/repos/rt-estimation/src/epyfilter/c2b2/ \
    --out-dir /ifs/scratch/jls106_gp/nhw2114/data/20231025_synthetic_data