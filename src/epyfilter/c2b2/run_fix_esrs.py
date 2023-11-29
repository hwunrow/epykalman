import os
import argparse
import numpy as np
import pandas as pd
import pickle
import logging
from tqdm import tqdm
from epyfilter import model_da, eakf, enks, posterior_checks, simulate_data
from matplotlib.backends.backend_pdf import PdfPages


from numpy.random import uniform


if __name__ == '__main__':
    try:
        sge_task_id = int(os.environ.get("SGE_TASK_ID"))
        sge_outputs_file = os.environ.get('SGE_STDOUT_PATH')
    except:
        sge_task_id = 1
        sge_outputs_file = "test.log"
    
    logger = logging.getLogger('my_logger')
    logger.setLevel(logging.INFO)

    sg_outputs_handler = logging.FileHandler(sge_outputs_file)
    sg_outputs_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    sg_outputs_handler.setFormatter(formatter)
    logger.addHandler(sg_outputs_handler)

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
    parser.add_argument(
        "--param-list", type=int, nargs="+",
        help="Rerunning for specific sge_task_ids"
    )
    args = parser.parse_args()

    np.random.seed(1994)

    files_per_task = 10
    df = pd.read_csv(os.path.join(args.in_dir, "good_param_list.csv"))
    start_row = (sge_task_id - 1) * files_per_task
    end_row = sge_task_id * files_per_task
    if end_row < len(df):
        pickle_files = df.iloc[start_row:end_row, 0]
    else:
        pickle_files = df.iloc[start_row:, 0]
    
    # override if supplied param list
    if args.param_list:
        pickle_files = args.param_list

    pickle_files = [f"/ifs/scratch/jls106_gp/nhw2114/data/20231025_synthetic_data/{p}_synthetic_data.pkl" for p in pickle_files]
    last_epidemic_days_df = pd.read_csv(os.path.join(args.in_dir, "last_epidemic_day.csv"))

    for i, pickle_file in enumerate(tqdm(pickle_files)):
        param_num = os.path.basename(pickle_file).split("_")[0]
        last_epi_day = int(last_epidemic_days_df.loc[last_epidemic_days_df.param==int(param_num), 'last_epidemic_day'].values)

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
        peak_days, = np.where(np.diff(np.sign(np.diff(det_data.i_true))) == -2)  # days where it increases before then decreases
        peak_days = peak_days[:2]  # just take first two days

        columns=["method", "in_ci","ens_var","in_ci_last_day","ens_var_last_day","in_ci_last_epi_day","ens_var_last_epi_day"]
        check_df = pd.DataFrame(columns=columns)

        def compute_posterior_checks(kf, method_name, is_ks=False):
            kf_checks = pd.DataFrame({
                'method' : method_name,
                'in_ci'                  : (posterior_checks.check_param_in_ci_ks(kf,late_day) if is_ks else posterior_checks.check_param_in_ci(kf,late_day)),
                'ens_var'                : (posterior_checks.compute_ens_var_ks(kf,late_day) if is_ks else posterior_checks.compute_ens_var(kf,late_day)),
                'in_ci_last_day'         : (posterior_checks.check_param_in_ci_ks(kf,"last") if is_ks else posterior_checks.check_param_in_ci(kf,"last")),
                'ens_var_last_day'       : (posterior_checks.compute_ens_var_ks(kf,"last") if is_ks else posterior_checks.compute_ens_var(kf,"last")),
                'in_ci_last_epi_day'     : (posterior_checks.check_param_in_ci_ks(kf,last_epi_day) if is_ks else posterior_checks.check_param_in_ci(kf,last_epi_day)),
                'ens_var_last_epi_day'   : (posterior_checks.compute_ens_var_ks(kf,last_epi_day) if is_ks else posterior_checks.compute_ens_var(kf,last_epi_day)),
            }, index=[0])
            return kf_checks

        for r in range(args.n_real):
            # adaptive inflation 
            kf = eakf.EnsembleAdjustmentKalmanFilter(model, m=args.n_ens)
            kf.filter(prior)
            percentiles = np.arange(2.5, 100, 2.5)
            kf.compute_reliability(percentiles)
            kf.compute_beta_reliability(percentiles)

            kf_checks = compute_posterior_checks(kf, "adaptive inflation")
            check_df = pd.concat([check_df, kf_checks], ignore_index=True)

            ks = enks.EnsembleSquareRootSmoother(kf)
            ks.smooth(window_size=10, plot=False)
            ks.compute_reliability(percentiles)
            ks.compute_beta_reliability(percentiles)

            ks_checks = compute_posterior_checks(ks, "smooth inflation", is_ks=True)
            check_df = pd.concat([check_df, ks_checks], ignore_index=True)

        check_df = check_df.groupby("method").mean()
        check_df.to_csv(f"{args.out_dir}/{param_num}_esrs_metrics.csv", index=True)
        logger.info(f"{param_num} saved csv")

logger.info("DONE")