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

in_dir = "/ifs/scratch/jls106_gp/nhw2114/data/20240827_epiestim_rmse_crps_reliability_day_rerun2"
in_dir_kl_w2 = "/ifs/scratch/jls106_gp/nhw2114/data/20240731_epiestim_run_kl_w2_day"

last_epi_days_df = pd.read_csv("/ifs/scratch/jls106_gp/nhw2114/repos/rt-estimation/src/epyfilter/c2b2/compute_days.csv")

files = glob.glob(f"{in_dir}/*_epiEstim_rmse_crps_reliability.csv")
len(files)

df_list = []

for file in tqdm(files):
    df = pd.read_csv(file)
    param = int(os.path.basename(file).split("_")[0])
    kl_w2_df = pd.read_csv(f"{in_dir_kl_w2}/{param}_epiEstim_kl_w2.csv") 
    
    first_epi_day = last_epi_days_df.loc[last_epi_days_df.param == param, "first_epi_day"].values[0]
    last_epi_day = last_epi_days_df.loc[last_epi_days_df.param == param, "last_epi_day"].values[0]
    
    first_epi_df = df.loc[df.day <= first_epi_day].mean()
    first_rmse = first_epi_df['rmse']
    first_crps = first_epi_df['crps']
    
    last_epi_df = df.loc[df.day <= last_epi_day].mean()
    last_rmse = last_epi_df['rmse']
    last_crps = last_epi_df['crps']
    
    first_kl_w2_df = kl_w2_df.loc[kl_w2_df.day <= first_epi_day].mean()
    first_kl = first_kl_w2_df['avg_kl']
    first_w2 = first_kl_w2_df['avg_w2']
    
    last_kl_w2_df = kl_w2_df.loc[kl_w2_df.day <= last_epi_day].mean()
    last_kl = last_kl_w2_df['avg_kl']
    last_w2 = last_kl_w2_df['avg_w2']
    
    metric_df = pd.DataFrame(data={
        'window'                      : [df.iloc[0]['window']],
        'rt_rmse_up_to_first_epi_day' : [first_rmse],
        'rt_rmse_up_to_last_epi_day'  : [last_rmse],
        'crps_up_to_first_epi_day'    : [first_crps],
        'crps_up_to_last_epi_day'     : [last_crps],
        'avg_kl_up_to_first_epi_day'  : [first_kl],
        'avg_kl_up_to_last_epi_day'   : [last_kl],
        'avg_w2_up_to_first_epi_day'  : [first_w2],
        'avg_w2_up_to_last_epi_day'   : [last_w2],
        'param'                       : [param],
    })
    
    df_list.append(metric_df)

epiestim_metric_df = pd.concat(df_list)

epiestim_metric_df.to_csv("/ifs/scratch/jls106_gp/nhw2114/data/20240827_epiestim_rmse_crps_reliability_day_rerun2/epiEstim_metrics_all.csv", index=False)
print("DONE!")