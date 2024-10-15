"""
This script calculates and aggregates various metrics related to EpiEstim model performance.

It reads in data from multiple CSV files, specifically:
* Files containing RMSE, CRPS, and reliability metrics for different parameters.
* Files containing KL divergence and W2 distance metrics for different parameters.
* A file defining the first and last epidemic days for each parameter.

For each parameter, it calculates mean RMSE and CRPS up to the first and last epidemic days, as well as mean KL divergence and W2 distance up to those days.

The results are then compiled into a single DataFrame and saved to a new CSV file.

Key functionalities:

* Reads and processes multiple CSV files containing model performance metrics.
* Calculates mean metrics for specific time windows (up to first and last epidemic days).
* Aggregates results across different parameters.
* Saves the final aggregated metrics to epiEstim_metrics_all.csv.
"""

import pandas as pd

import os
import glob

from tqdm.auto import tqdm

in_dir = "/ifs/scratch/jls106_gp/nhw2114/data/20240818_run_rmse_crps_inci_fix"
in_dir_kl_w2 = "/ifs/scratch/jls106_gp/nhw2114/data/20240727_run_kl_w2_day"

last_epi_days_df = pd.read_csv("/ifs/scratch/jls106_gp/nhw2114/repos/rt-estimation/src/epyfilter/c2b2/compute_days.csv")

files = glob.glob(f"{in_dir}/*_eakf_metrics.csv")

df_list = []

for file in tqdm(files):
    df = pd.read_csv(file)
    param = int(os.path.basename(file).split("_")[0])
    kl_w2_df = pd.read_csv(f"{in_dir_kl_w2}/{param}_eakf_metrics.csv") 
    
    first_epi_day = last_epi_days_df.loc[last_epi_days_df.param == param, "first_epi_day"].values[0]
    last_epi_day = last_epi_days_df.loc[last_epi_days_df.param == param, "last_epi_day"].values[0]
    
    first_epi_df = df.loc[df.day <= first_epi_day].groupby("method").mean()
    first_rmse = first_epi_df['rt_rmse']
    first_crps = first_epi_df['crps']
    
    last_epi_df = df.loc[df.day <= last_epi_day].groupby("method").mean()
    last_rmse = last_epi_df['rt_rmse']
    last_crps = last_epi_df['crps']
    
    first_kl_w2_df = kl_w2_df.loc[kl_w2_df.day <= first_epi_day].groupby("method").mean()
    first_kl = first_kl_w2_df['avg_kl']
    first_w2 = first_kl_w2_df['avg_w2']
    
    last_kl_w2_df = kl_w2_df.loc[kl_w2_df.day <= last_epi_day].groupby("method").mean()
    last_kl = last_kl_w2_df['avg_kl']
    last_w2 = last_kl_w2_df['avg_w2']
    
    metric_df = pd.concat([first_rmse, last_rmse, first_crps, last_crps, first_kl, last_kl, first_w2, last_w2], axis=1).reset_index()
    metric_df.columns = ['method', 'rt_rmse_up_to_first_epi_day', 'rt_rmse_up_to_last_epi_day', 'crps_up_to_first_epi_day', 'crps_up_to_last_epi_day', 'avg_kl_up_to_first_epi_day', 'avg_kl_up_to_last_epi_day', 'avg_w2_up_to_first_epi_day', 'avg_w2_up_to_last_epi_day']
    metric_df['param'] = param
    
    df_list.append(metric_df)

eakf_metric_df = pd.concat(df_list)

eakf_metric_df.to_csv(f"{in_dir}/eakf_metrics_all.csv", index=False)


