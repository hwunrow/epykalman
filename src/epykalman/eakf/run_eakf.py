import os
import argparse
import numpy as np
import pandas as pd
import pickle
import logging
from tqdm import tqdm
from epyfilter import model_da, eakf, enks, posterior_checks
from matplotlib.backends.backend_pdf import PdfPages


from numpy.random import uniform


def compute_posterior_checks(kf, method_name, is_ks=False):
    if args.compute_kl_w2_time_series:
        kf_checks = np.zeros((len(kf.x_list), 2))
        if is_ks:
            for day in range(len(kf.x_list)):
                kf_checks[day,0] = posterior_checks.avg_kl_divergence_ks(kf, day, evaluate_on=True)
                kf_checks[day,1] = posterior_checks.avg_wasserstein2_ks(kf, day, evaluate_on=True)
        else:
            for day in range(len(kf.x_list)):
                kf_checks[day,0] = posterior_checks.avg_kl_divergence(kf, day, evaluate_on=True)
                kf_checks[day,1] = posterior_checks.avg_wasserstein2(kf, day, evaluate_on=True)

        kf_checks = pd.DataFrame(kf_checks, columns=["avg_kl", "avg_w2"])
        kf_checks["method"] = method_name
        kf_checks["day"] = range(len(kf.x_list))
    elif args.compute_time_series:
        kf_checks = np.zeros((len(kf.x_list), 3))
        for day in range(len(kf.x_list)):
            kf_checks[day,0] = posterior_checks.rt_rmse(kf, False, day)
            kf_checks[day,1] = posterior_checks.crps(kf, day)
            if is_ks:
                kf_checks[day,2] = posterior_checks.check_param_in_ci_ks(kf, day)
            else:
                kf_checks[day,2] = posterior_checks.check_param_in_ci(kf, day)

        kf_checks = pd.DataFrame(kf_checks, columns=["rt_rmse", "crps", "in_ci"])
        kf_checks["method"] = method_name
        kf_checks["day"] = range(len(kf.x_list))
    else:
        kf_checks = pd.DataFrame(
            {
                "method": method_name,
                "rt_rmse_up_to_last_epi_day": posterior_checks.rt_rmse(kf, True, last_epi_day),
                "rt_rmse_last_epi_day": posterior_checks.rt_rmse(kf, False, last_epi_day),
                "avg_w2_up_to_last_epi_day": (
                    posterior_checks.avg_wasserstein2_ks(kf, last_epi_day)
                    if is_ks
                    else posterior_checks.avg_wasserstein2(kf, last_epi_day)
                ),
                "avg_w2_last_epi_day": (
                    posterior_checks.avg_wasserstein2_ks(kf, last_epi_day, evaluate_on=True)
                    if is_ks
                    else posterior_checks.avg_wasserstein2(kf, last_epi_day, evaluate_on=True)
                ),
                "avg_kl_up_to_last_epi_day": (
                    posterior_checks.avg_kl_divergence_ks(kf, last_epi_day)
                    if is_ks
                    else posterior_checks.avg_kl_divergence(kf, last_epi_day)
                ),
                "avg_kl_last_epi_day": (
                    posterior_checks.avg_kl_divergence_ks(kf, last_epi_day, evaluate_on=True)
                    if is_ks
                    else posterior_checks.avg_kl_divergence(kf, last_epi_day, evaluate_on=True)
                ),
                "in_ci_last_epi_day": (
                    posterior_checks.check_param_in_ci_ks(kf, last_epi_day)
                    if is_ks
                    else posterior_checks.check_param_in_ci(kf, last_epi_day)
                ),
                "crps_last_epi_day": posterior_checks.crps(kf, last_epi_day),
                "rt_rmse_up_to_first_epi_day": posterior_checks.rt_rmse(kf, True, first_epi_day),
                "rt_rmse_first_epi_day": posterior_checks.rt_rmse(kf, False, first_epi_day),
                "avg_w2_up_to_first_epi_day": (
                    posterior_checks.avg_wasserstein2_ks(kf, first_epi_day)
                    if is_ks
                    else posterior_checks.avg_wasserstein2(kf, first_epi_day)
                ),
                "avg_w2_first_epi_day": (
                    posterior_checks.avg_wasserstein2_ks(kf, first_epi_day, evaluate_on=True)
                    if is_ks
                    else posterior_checks.avg_wasserstein2(kf, first_epi_day, evaluate_on=True)
                ),
                "avg_kl_up_to_first_epi_day": (
                    posterior_checks.avg_kl_divergence_ks(kf, first_epi_day)
                    if is_ks
                    else posterior_checks.avg_kl_divergence(kf, first_epi_day)
                ),
                "avg_kl_first_epi_day": (
                    posterior_checks.avg_kl_divergence_ks(kf, first_epi_day, evaluate_on=True)
                    if is_ks
                    else posterior_checks.avg_kl_divergence(kf, first_epi_day, evaluate_on=True)
                ),
                "in_ci_first_epi_day": (
                    posterior_checks.check_param_in_ci_ks(kf, first_epi_day)
                    if is_ks
                    else posterior_checks.check_param_in_ci(kf, first_epi_day)
                ),
                "crps_first_epi_day": posterior_checks.crps(kf, first_epi_day),
            },
            index=[0],
        )
    return kf_checks


def save_plots(kf, kf_no, kf_fixed, param_num, args):
    """Plot posterior, reliability, and ppc for kf methods."""
    pdf_file = args.out_dir + f"/{param_num}_eakf_plots.pdf"

    f1 = kf.plot_posterior(
        path=args.out_dir,
        name=f"{param_num}_eakf_posterior_adaptive_inflation",
    )
    f2 = kf.plot_reliability(
        path=args.out_dir,
        name=f"{param_num}_eakf_reliability_adaptive_inflation",
    )
    f3 = kf.plot_ppc(path=args.out_dir, name=f"{param_num}_eakf_ppc_adaptive_inflation")

    f4 = kf_no.plot_posterior(
        path=args.out_dir, name=f"{param_num}_eakf_posterior_no_inflation"
    )
    f5 = kf_no.plot_reliability(
        path=args.out_dir, name=f"{param_num}_eakf_reliability_no_inflation"
    )
    f6 = kf_no.plot_ppc(path=args.out_dir, name=f"{param_num}_eakf_ppc_no_inflation")

    f7 = kf_fixed.plot_posterior(
        path=args.out_dir,
        name=f"{param_num}_eakf_posterior_fixed_inflation",
    )
    f8 = kf_fixed.plot_reliability(
        path=args.out_dir,
        name=f"{param_num}_eakf_reliability_fixed_inflation",
    )
    f9 = kf_fixed.plot_ppc(
        path=args.out_dir, name=f"{param_num}_eakf_ppc_fixed_inflation"
    )

    with PdfPages(pdf_file) as pdf:
        pdf.savefig(f1)
        pdf.savefig(f2)
        pdf.savefig(f3)
        pdf.savefig(f4)
        pdf.savefig(f5)
        pdf.savefig(f6)
        pdf.savefig(f7)
        pdf.savefig(f8)
        pdf.savefig(f9)


if __name__ == "__main__":
    if any(var in os.environ for var in ["SLURM_JOB_ID", "SLURM_ARRAY_TASK_ID"]):
        sge_task_id = int(os.environ.get("SLURM_ARRAY_TASK_ID"))
        slurm_submit_dir = os.environ.get("SLURM_SUBMIT_DIR")
        slurm_job_id = os.environ.get("SLURM_JOB_ID")
        sge_outputs_file = os.path.join(slurm_submit_dir, f"{slurm_job_id}.{sge_task_id}.out")
    elif any(var in os.environ for var in ["JOB_ID", "SGE_TASK_ID"]): 
        sge_task_id = int(os.environ.get("SGE_TASK_ID"))
        sge_outputs_file = os.environ.get("SGE_STDOUT_PATH")
    else:
        sge_task_id = 1
        sge_outputs_file = "test.log"

    logger = logging.getLogger("my_logger")
    logger.setLevel(logging.INFO)

    sg_outputs_handler = logging.FileHandler(sge_outputs_file)
    sg_outputs_handler.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    sg_outputs_handler.setFormatter(formatter)
    logger.addHandler(sg_outputs_handler)
    print(sge_task_id)

    parser = argparse.ArgumentParser(
        description="Run EAKF with adaptive, fixed, and no inflation for 1000 different synthetic data sets",
    )
    parser.add_argument(
        "--n-real",
        type=int,
        required=False,
        default=100,
        help="Number of realizations to run EAKF.",
    )
    parser.add_argument(
        "--n-ens",
        type=int,
        required=False,
        default=300,
        help="Number of ensemble members.",
    )
    parser.add_argument(
        "--fixed-inf",
        type=float,
        required=False,
        default=1.05,
        help="Number of ensemble members.",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Plot results.",
    )
    parser.add_argument(
        "--save-data",
        action="store_true",
        help="Save pkl for each realization.",
    )
    parser.add_argument(
        "--save-reliability",
        action="store_true",
        help="Save reliablity for data and beta.",
    )
    parser.add_argument(
        "--in-dir",
        type=str,
        required=True,
        help="Directory for inputs."
    )
    parser.add_argument(
        "--pkl-dir",
        type=str,
        required=True,
        help="Directory for synthetic data inputs.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        required=True,
        help="Directory to save plots and files."
    )
    parser.add_argument(
        "--param-file", type=str, required=False, help="CSV filename with params to run."
    )
    parser.add_argument(
        "--param-list", type=int, nargs="+", help="Rerunning for specific sge_task_ids"
    )
    parser.add_argument(
        "--files-per-task",
        type=int,
        default=50,
        help="Number of files per array subjob"
    )
    parser.add_argument(
        "--compute-kl-w2-time-series",
        action="store_true",
        help="Special run to compute time series of kl and w2 metrics"
    )
    parser.add_argument(
        "--compute-time-series",
        action="store_true",
        help="Special run to compute time series of rmse, crps, and reliability metrics"
    )
    args = parser.parse_args()

    np.random.seed(1994)

    if not args.param_file and not args.param_list:
        raise ValueError("Must supply either param_file or param list")

    if args.param_file:
        files_per_task = args.files_per_task
        df = pd.read_csv(os.path.join(args.in_dir, args.param_file))
        start_row = (sge_task_id - 1) * files_per_task
        end_row = sge_task_id * files_per_task
        if end_row < len(df):
            pickle_files = df.iloc[start_row:end_row, 0]
        else:
            pickle_files = df.iloc[start_row:, 0]

        pickle_files = [
            os.path.join(args.pkl_dir, f"{p}_synthetic_data.pkl") for p in pickle_files
        ]
    # override if supplied param list
    if args.param_list:
        pickle_files = [
            os.path.join(args.pkl_dir, f"{pp}_synthetic_data.pkl")
            for pp in args.param_list
        ]

    last_epidemic_days_df = pd.read_csv(os.path.join(args.in_dir, "compute_days.csv"))

    for i, pickle_file in enumerate(tqdm(pickle_files)):
        param_num = os.path.basename(pickle_file).split("_")[0]
        last_epi_day = int(
            last_epidemic_days_df.loc[
                last_epidemic_days_df.param == int(param_num), "last_epi_day"
            ].values[0]
        )
        first_epi_day = int(
            last_epidemic_days_df.loc[
                last_epidemic_days_df.param == int(param_num), "first_epi_day"
            ].values[0]
        )

        with open(pickle_file, "rb") as file:
            data = pickle.load(file)

        model = model_da.SIR_model(data)

        prior = {
            "beta": {
                "dist": uniform,
                "args": {
                    "low": 0.1,
                    "high": 2.0,
                },
            },
            "t_I": {
                "dist": "constant",
            },
        }

        # beta_1 = data.rt_1 / data.t_I
        # beta_0 = data.rt_0 / data.t_I
        # late_day = (
        #     -1 / data.k * np.log((beta_1 - beta_0) / (0.99 * beta_1 - beta_0) - 1)
        #     + data.midpoint
        # )
        # late_day = int(late_day)

        # det_data = simulate_data.simulate_data(
        #     **data.true_params, run_deterministic=True
        # )
        # (peak_days,) = np.where(
        #     np.diff(np.sign(np.diff(det_data.i_true))) == -2
        # )  # days where it increases before then decreases
        # peak_days = peak_days[:2]  # just take first two days

        reliability_df = pd.DataFrame()
        check_df = pd.DataFrame()

        for r in range(args.n_real):
            # adaptive inflation
            kf = eakf.EnsembleAdjustmentKalmanFilter(model, m=args.n_ens)
            kf.filter(prior)
            percentiles = np.arange(2.5, 100, 2.5)
            kf.compute_reliability(percentiles)
            kf.compute_beta_reliability(percentiles)

            kf_checks = compute_posterior_checks(kf, "adaptive inflation")
            check_df = pd.concat([check_df, kf_checks], ignore_index=True)
            if args.save_data:
                kf.save_data(
                    path=args.out_dir, name=f"{param_num}_adaptive_inflation_run_{r}"
                )

            # no inflation
            kf_no = eakf.EnsembleAdjustmentKalmanFilter(model, m=args.n_ens)
            kf_no.filter(prior, inf_method="none")
            kf_no.compute_reliability(percentiles)
            kf_no.compute_beta_reliability(percentiles)

            kf_no_checks = compute_posterior_checks(kf_no, "no inflation")
            check_df = pd.concat([check_df, kf_no_checks], ignore_index=True)
            if args.save_data:
                kf_no.save_data(
                    path=args.out_dir, name=f"{param_num}_no_inflation_run_{r}"
                )

            # fixed inflation
            kf_fixed = eakf.EnsembleAdjustmentKalmanFilter(model, m=args.n_ens)
            kf_fixed.filter(prior, inf_method="constant", lam_fixed=args.fixed_inf)
            kf_fixed.compute_reliability(percentiles)
            kf_fixed.compute_beta_reliability(percentiles)

            kf_fixed_checks = compute_posterior_checks(kf_fixed, "fixed inflation")
            check_df = pd.concat([check_df, kf_fixed_checks], ignore_index=True)
            if args.save_data:
                kf_fixed.save_data(
                    path=args.out_dir, name=f"{param_num}_fixed_inflation_run_{r}"
                )

            ks = enks.EnsembleSquareRootSmoother(kf)
            ks.smooth(window_size=10, plot=False)
            ks.compute_reliability(percentiles)
            ks.compute_beta_reliability(percentiles)

            ks_checks = compute_posterior_checks(ks, "smooth inflation", is_ks=True)
            check_df = pd.concat([check_df, ks_checks], ignore_index=True)
            if args.save_data:
                ks.save_data(
                    path=args.out_dir, name=f"{param_num}_smooth_inflation_run_{r}"
                )

            if args.plot and r == 0:
                # only plot for first realization and if plot flag toggled
                save_plots(kf, kf_no, kf_fixed, param_num, args)

            tmp_df = pd.DataFrame.from_dict(
                {
                    "percentile": percentiles,
                    "adaptive": kf.prop_list,
                    "adaptive_beta": kf.beta_prop_list,
                    "no": kf_no.prop_list,
                    "no_beta": kf_no.beta_prop_list,
                    "fixed": kf_fixed.prop_list,
                    "fixed_beta": kf_fixed.beta_prop_list,
                    "smooth": ks.prop_list,
                    "smooth_beta": ks.beta_prop_list,
                }
            )

            reliability_df = pd.concat([reliability_df, tmp_df], ignore_index=True)

        if args.save_reliability:
            reliability_df = reliability_df.groupby("percentile").mean()
            reliability_df["param"] = param_num
            reliability_df.to_csv(
                f"{args.out_dir}/{param_num}_reliability.csv", index=True
            )
        if args.compute_kl_w2_time_series or args.compute_time_series:
            check_df = check_df.groupby(["method","day"]).mean().reset_index()
        else:
            check_df = check_df.groupby("method").mean()
        check_df["param"] = param_num
        check_df.to_csv(f"{args.out_dir}/{param_num}_eakf_metrics.csv", index=True)
        logger.info(f"{param_num} saved csv")

    logger.info("DONE")
