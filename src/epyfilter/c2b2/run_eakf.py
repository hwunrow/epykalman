import os
import argparse
import numpy as np
import pandas as pd
import pickle
from epyfilter import model_da, eakf, enks, posterior_checks

from numpy.random import uniform


if __name__ == '__main__':
    # sge_task_id = int(os.environ.get("SGE_TASK_ID"))
    sge_task_id = 1

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

        check_array = np.empty((0, 5))

        for r in range(args.n_real):

            # adaptive inflation 
            kf = eakf.EnsembleAdjustmentKalmanFilter(model, m=args.n_ens)
            kf.filter(prior)
            
            if r == 0:
                kf.plot_posterior(path=args.out_dir, name=f"{sge_task_id}_{i}_eakf_posterior_adaptive_inflation")
                kf.plot_reliability(path=args.out_dir, name=f"{sge_task_id}_{i}_eakf_reliability_adaptive_inflation")
                kf.plot_ppc(path=args.out_dir, name=f"{sge_task_id}_{i}_eakf_ppc_adaptive_inflation")

            kf_checks = np.array(["adaptive inflation",
                                  posterior_checks.rmse(kf),
                                  posterior_checks.avg_wasserstein2(kf),
                                  posterior_checks.avg_kl_divergence(kf),
                                  posterior_checks.check_param_in_ci(kf, late_day)])
            check_array = np.vstack((check_array, kf_checks))

            # no inflation
            kf_no = eakf.EnsembleAdjustmentKalmanFilter(model, m=args.n_ens)
            kf_no.filter(prior, inf_method="none")

            if r == 0:
                kf_no.plot_posterior(path=args.out_dir, name=f"{sge_task_id}_{i}_eakf_posterior_no_inflation")
                kf_no.plot_reliability(path=args.out_dir, name=f"{sge_task_id}_{i}_eakf_reliability_no_inflation")
                kf_no.plot_ppc(path=args.out_dir, name=f"{sge_task_id}_{i}_eakf_ppc_no_inflation")

            kf_no_checks = np.array(["no inflation",
                                     posterior_checks.rmse(kf_no),
                                     posterior_checks.avg_wasserstein2(kf_no),
                                     posterior_checks.avg_kl_divergence(kf_no),
                                     posterior_checks.check_param_in_ci(kf_no, late_day)])
            check_array = np.vstack((check_array, kf_no_checks))

            # fixed inflation 
            kf_fixed = eakf.EnsembleAdjustmentKalmanFilter(model, m=args.n_ens)
            kf_fixed.filter(prior, inf_method="constant", lam_fixed=args.fixed_inf)
            
            if r == 0:
                kf_fixed.plot_posterior(path=args.out_dir, name=f"{sge_task_id}_{i}_eakf_posterior_fixed_inflation")
                kf_fixed.plot_reliability(path=args.out_dir, name=f"{sge_task_id}_{i}_eakf_reliability_fixed_inflation")
                kf_fixed.plot_ppc(path=args.out_dir, name=f"{sge_task_id}_{i}_eakf_ppc_fixed_inflation")

            kf_fixed_checks = np.array(["fixed inflation",
                                        posterior_checks.rmse(kf_fixed),
                                        posterior_checks.avg_wasserstein2(kf_fixed),
                                        posterior_checks.avg_kl_divergence(kf_fixed),
                                        posterior_checks.check_param_in_ci(kf_fixed, late_day)])
            check_array = np.vstack((check_array, kf_fixed_checks))

            ks = enks.EnsembleSquareRootSmoother(kf)
            ks.smooth(window_size=10, plot=False)

            ks_checks = np.array(["smooth",
                                  posterior_checks.rmse(ks),
                                  posterior_checks.avg_wasserstein2_ks(ks),
                                  posterior_checks.avg_kl_divergence_ks(ks),
                                  posterior_checks.check_param_in_ci_ks(ks, late_day)])
            check_array = np.vstack((check_array, ks_checks))
        
        check_df = pd.DataFrame(check_array, columns=["method","rmse","avg_w2","avg_kl","in_ci"])
        import pdb; pdb.set_trace()
        check_df = check_df.groupby("method").mean()

        check_df.to_csv(f"args.out_dir/{i}_{sge_task_id}.csv", index=False)
