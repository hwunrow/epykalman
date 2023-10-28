import os
import argparse
import numpy as np
import pandas as pd
import pickle
from epyfilter import model_da, eakf, enks, posterior_checks, simulate_data

from numpy.random import uniform


if __name__ == '__main__':
    sge_task_id = int(os.environ.get("SGE_TASK_ID"))

    parser = argparse.ArgumentParser(
        description="Run EAKF with adaptive, fixed, and no inflation for 1000 different synthetic data sets",
    )
    parser.add_argument(
        "--n-real", type=int, required=False,
        default=100,
        help="Number of realizations to run EAKF.")
    parser.add_argument(
        "--n-ens", type=int, required=False,
        default=300,
        help="Number of ensemble members.")
    parser.add_argument(
        "--fixed-inf", type=float, required=False,
        default=1.05,
        help="Number of ensemble members.")
    parser.add_argument(
        "--in-dir", type=str, required=True,
        help="Directory for inputs.")
    parser.add_argument(
        "--out-dir", type=str, required=True,
        help="Directory to save plots and files.")
    args = parser.parse_args()

    np.random.seed(1994)

    files_per_task = 1000
    df = pd.read_csv(os.path.join(args.in_dir, "pickle_list.csv"))
    start_row = (sge_task_id - 1) * files_per_task
    end_row = sge_task_id * files_per_task
    if end_row < len(df):
        pickle_files = df.iloc[start_row:end_row, 0]
    else:
        pickle_files = df.iloc[start_row:, 0]

    for i, pickle_file in enumerate(pickle_files):
        param_num = os.path.basename(pickle_file).split("_")[0]

        with open(pickle_file, 'rb') as file:
            data = pickle.load(file)

        model = model_da.SIR_model(data)

        prior = {
            'beta':{
                'dist': uniform,
                'args':{
                    'low': 0.1,
                    'high': 2.,
                }
            },
            't_I':{
                'dist': "constant",
            },
        }

        beta_1 = data.rt_1 / data.t_I
        beta_0 = data.rt_0 / data.t_I
        late_day = -1/data.k * np.log((beta_1 - beta_0)/(0.99*beta_1 - beta_0)-1) + data.midpoint
        late_day = int(late_day)

        det_data = simulate_data.simulate_data(**data.true_params, run_deterministic=True)
        peak_days, = np.where(np.diff(np.sign(np.diff(det_data.i_true))) == -2)

        columns=["method", "rt_peak_rmse", "rt_rmse", "data_rmse","avg_w2","avg_kl","in_ci"]
        check_df = pd.DataFrame(columns=columns)

        for r in range(args.n_real):

            # adaptive inflation 
            kf = eakf.EnsembleAdjustmentKalmanFilter(model, m=args.n_ens)
            kf.filter(prior)
            
            if r == 0:
                kf.plot_posterior(path=args.out_dir, name=f"{param_num}_eakf_posterior_adaptive_inflation")
                kf.plot_reliability(path=args.out_dir, name=f"{param_num}_eakf_reliability_adaptive_inflation")
                kf.plot_ppc(path=args.out_dir, name=f"{param_num}_eakf_ppc_adaptive_inflation")

            kf_checks = pd.DataFrame([
                ["adaptive inflation",
                 posterior_checks.rt_rmse(kf, peaks=peak_days),
                 posterior_checks.rt_rmse(kf),
                 posterior_checks.data_rmse(kf),
                 posterior_checks.avg_wasserstein2(kf),
                 posterior_checks.avg_kl_divergence(kf),
                 posterior_checks.check_param_in_ci(kf,late_day)]
                 ], columns=columns)
            check_df = pd.concat([check_df, kf_checks], ignore_index=True)

            # no inflation
            kf_no = eakf.EnsembleAdjustmentKalmanFilter(model, m=args.n_ens)
            kf_no.filter(prior, inf_method="none")

            if r == 0:
                kf_no.plot_posterior(path=args.out_dir, name=f"{param_num}_eakf_posterior_no_inflation")
                kf_no.plot_reliability(path=args.out_dir, name=f"{param_num}_eakf_reliability_no_inflation")
                kf_no.plot_ppc(path=args.out_dir, name=f"{param_num}_eakf_ppc_no_inflation")

            kf_no_checks = pd.DataFrame([
                ["no inflation",
                 posterior_checks.rt_rmse(kf_no, peaks=peak_days),
                 posterior_checks.rt_rmse(kf_no),
                 posterior_checks.data_rmse(kf_no),
                 posterior_checks.avg_wasserstein2(kf_no),
                 posterior_checks.avg_kl_divergence(kf_no),
                 posterior_checks.check_param_in_ci(kf_no,late_day)]
                 ], columns=columns)
            check_df = pd.concat([check_df, kf_no_checks], ignore_index=True)

            # fixed inflation 
            kf_fixed = eakf.EnsembleAdjustmentKalmanFilter(model, m=args.n_ens)
            kf_fixed.filter(prior, inf_method="constant", lam_fixed=args.fixed_inf)
            
            if r == 0:
                kf_fixed.plot_posterior(path=args.out_dir, name=f"{param_num}_eakf_posterior_fixed_inflation")
                kf_fixed.plot_reliability(path=args.out_dir, name=f"{param_num}_eakf_reliability_fixed_inflation")
                kf_fixed.plot_ppc(path=args.out_dir, name=f"{param_num}_eakf_ppc_fixed_inflation")

            kf_fixed_checks = pd.DataFrame([
                ["fixed inflation",
                posterior_checks.rt_rmse(kf_fixed, peaks=peak_days),
                posterior_checks.rt_rmse(kf_fixed),
                posterior_checks.data_rmse(kf_fixed),
                posterior_checks.avg_wasserstein2(kf_fixed),
                posterior_checks.avg_kl_divergence(kf_fixed),
                posterior_checks.check_param_in_ci(kf_fixed, late_day)]
                ], columns=columns)
            check_df = pd.concat([check_df, kf_fixed_checks], ignore_index=True)

            ks = enks.EnsembleSquareRootSmoother(kf)
            ks.smooth(window_size=10, plot=False)

            ks_checks = pd.DataFrame([
                ["smooth",
                 posterior_checks.rt_rmse(ks, peaks=peak_days),
                 posterior_checks.rt_rmse(ks),
                 posterior_checks.data_rmse(ks),
                 posterior_checks.avg_wasserstein2_ks(ks),
                 posterior_checks.avg_kl_divergence_ks(ks),
                 posterior_checks.check_param_in_ci_ks(ks, late_day)]
                ], columns=columns)
            check_df = pd.concat([check_df, ks_checks], ignore_index=True)
        
        check_df = check_df.groupby("method").mean()
        check_df.to_csv(f"{args.out_dir}/{param_num}_eakf_metrics.csv", index=True)
