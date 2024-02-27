import os
import argparse
import numpy as np
import pandas as pd
import pickle
import logging
from tqdm import tqdm
from matplotlib.backends.backend_pdf import PdfPages


def open_pickle(pickle_file):
    with open(f"{pickle_file}", 'rb') as file:
        data = pickle.load(file)
    return data


def score_rank_hist(file_prefix, n_real, n_ens):
    scores = []
    rank = np.zeros((n_real, 365))
    percentiles = np.zeros((n_real, 365))
    for run in tqdm(range(args.n_real)):
        print(file_prefix)
        file = args.pkl_dir + f"/{file_prefix}_{run}.pkl"
        eakf = open_pickle(file)

        truth_rt = eakf.data.beta * eakf.data.t_I
        post_rt = np.sort(np.array([θ.beta * θ.t_I for θ in eakf.θ_list]), axis=1)

        # find which index in ensemble posterior the truth lies
        idx = (post_rt.T < truth_rt).sum(axis=0)
        rank[run, :] = idx

        # map from index to percentile p (smallest one where truth lives within p% CrI)
        # each ensemble member has mass 1 / eakf.m
        # for cases where; the truth does not lie within idx = 0 or idx = n_ens
        assert n_ens == eakf.m
        percentiles[run, :] = np.maximum(1 - 2*idx/n_ens, 2*idx/n_ens - 1)

    for i in range(eakf.data.n_t):
        hist = np.histogram(rank[:, i], bins=np.arange(0, n_ens + 1))[0]
        scores.append(np.sum(np.abs(hist[0] - n_real/(n_ens + 1))))

    scores = np.argsort(np.array(scores))

    return percentiles, scores


def score_reliability(percentiles, n_real, n_ens):
    scores = []
    p_list = np.arange(0, 1, 2/n_ens)
    for i in range(percentiles.shape[1]):
        cdf = np.cumsum(np.histogram(percentiles[:, i], bins=p_list)[0]/n_real)
        cdf = np.insert(cdf, 0, 0.)
        scores.append(np.mean((p_list - cdf) ** 2))

    scores = np.argsort(np.array(scores))

    return scores


def crps(observation, ensembles):
    """
    Calculate the Continuous Ranked Probability Score (CRPS).

    Parameters:
        observation (float): The observed value.
        ensembles (array_like): The posterior of the prediction. It should be a
            1D array representing the ensemble members of the filter on one day

    Returns:
        crps_score (float): The CRPS score.
    """
    hist, bins = np.histogram(ensembles, bins=len(np.unique(ensembles)))
    cdf = np.cumsum(hist/ensembles.shape[0])
    crps_score = 0
    for i, x in enumerate(bins[1:]):
        crps_score += (cdf[i] - np.heaviside(x-observation, 0.5))**2 * (bins[i] - bins[i-1])

    return crps_score


def crps_all_runs(file_prefix):
    """
    Calculate the average Continuous Ranked Probability Score (CRPS) across all runs.

    Parameters:
        file_prefix (str): The prefix of the file name.

    Returns:
        crps_all_runs (array_like): The average CRPS scores.
    """
    crps_all_runs = []
    for run in tqdm(range(100)):
        file = args.pkl_dir + f"/{file_prefix}_{run}.pkl"
        eakf = open_pickle(file)

        truth = eakf.data.beta * eakf.data.t_I
        post_rts = np.sort(np.array([θ.beta * θ.t_I for θ in eakf.θ_list]), axis=1)

        scores = []
        for i, obs in enumerate(truth):
            scores.append(crps(obs, post_rts[i,:]))

        crps_all_runs.append(scores)

    crps_all_runs = np.argsort(np.mean(crps_all_runs, axis=0))
    return crps_all_runs


def rmse_all_runs(file_prefix):
    """
    Calculate the average root mean square error (RMSE) across all runs.

    Parameters:
        file_prefix (str): The prefix of the file name.

    Returns:
        rmse_all_runs (array_like): The average RMSE.
    """
    rmse_all_runs = []
    for run in tqdm(range(100)):
        file = args.pkl_dir + f"/{file_prefix}_{run}.pkl"
        eakf = open_pickle(file)

        truth = eakf.data.beta * eakf.data.t_I
        post_rts = np.sort(np.array([θ.beta * θ.t_I for θ in eakf.θ_list]), axis=1)

        rmse = np.mean(np.square((post_rts - truth[:, np.newaxis])), axis=1)

        rmse_all_runs.append(rmse)

    rmse_all_runs = np.mean(np.array(rmse_all_runs), axis=0)
    rmse_all_runs = np.argsort(rmse_all_runs)

    return rmse_all_runs


if __name__ == "__main__":
    try:
        sge_task_id = int(os.environ.get("SGE_TASK_ID"))
        sge_outputs_file = os.environ.get("SGE_STDOUT_PATH")
    except Exception:
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

    parser = argparse.ArgumentParser(
        description="Computes ranking of the days on which the filter performs best.",
    )
    parser.add_argument(
        "--in-dir", type=str, required=True, help="Directory for SGE inputs."
    )
    parser.add_argument(
        "--n-real", type=int, required=True, help="Number of realizations ran."
    )
    parser.add_argument(
        "--n-ens", type=int, required=True, help="Number of ensemble members."
    )
    parser.add_argument(
        "--pkl-dir",
        type=str,
        required=True,
        help="Directory for eakf realization data.",
    )
    parser.add_argument(
        "--out-dir", type=str, required=True, help="Directory to save plots and files."
    )
    parser.add_argument(
        "--param-file", type=str, required=True, help="CSV filename with params to run."
    )
    parser.add_argument(
        "--param-list", type=int, nargs="+", help="Rerunning for specific sge_task_ids"
    )
    parser.add_argument(
        "--files-per-task", type=int, default=1, help="Number of files per task."
    )
    args = parser.parse_args()

    df = pd.read_csv(os.path.join(args.in_dir, args.param_file))
    start_row = (sge_task_id - 1) * args.files_per_task
    end_row = sge_task_id * args.files_per_task
    if end_row < len(df):
        params = df.iloc[start_row:end_row, 0]
    else:
        params = df.iloc[start_row:, 0]

    # override if supplied param list
    if args.param_list:
        params = args.param_list

    for i, pp in enumerate(tqdm(params)):
        # compute rank histogram and associated scores
        adapt_percentiles, adapt_rank_hist_score = \
            score_rank_hist(f"{pp}_adaptive_inflation_run", args.n_real, args.n_ens)
        no_percentiles, no_rank_hist_score = \
            score_rank_hist(f"{pp}_no_inflation_run", args.n_real, args.n_ens)
        fixed_percentiles, fixed_rank_hist_score = \
            score_rank_hist(f"{pp}_fixed_inflation_run", args.n_real, args.n_ens)
        smooth_percentiles, smooth_rank_hist_score = \
            score_rank_hist(f"{pp}_smooth_inflation_run", args.n_real, args.n_ens)

        # compute reliability scores
        adapt_reliability_score = score_reliability(adapt_percentiles, args.n_real, args.n_ens)
        no_reliability_score = score_reliability(no_percentiles, args.n_real, args.n_ens)
        fixed_reliability_score = score_reliability(fixed_percentiles, args.n_real, args.n_ens)
        smooth_reliability_score = score_reliability(smooth_percentiles, args.n_real, args.n_ens)

        # compute crps
        adapt_crps_scores = crps_all_runs(f"{pp}_adaptive_inflation_run")
        no_crps_scores = crps_all_runs(f"{pp}_no_inflation_run")
        fixed_crps_scores = crps_all_runs(f"{pp}_fixed_inflation_run")
        smooth_crps_scores = crps_all_runs(f"{pp}_smooth_inflation_run")

        # compute rmse
        adapt_rmse_scores = rmse_all_runs(f"{pp}_adaptive_inflation_run")
        no_rmse_scores = rmse_all_runs(f"{pp}_no_inflation_run")
        fixed_rmse_scores = rmse_all_runs(f"{pp}_fixed_inflation_run")
        smooth_rmse_scores = rmse_all_runs(f"{pp}_smooth_inflation_run")

        # save results
        adapt_df = pd.DataFrame(
            np.array([
                np.arange(len(adapt_rank_hist_score)),
                adapt_rank_hist_score,
                adapt_reliability_score,
                adapt_crps_scores,
                adapt_rmse_scores]).T,
            columns=["day", "rank_hist", "reliability", "crps", "rmse"]
        )
        adapt_df["type"] = "adaptive"
        adapt_df['average'] = np.argsort(adapt_df[["rank_hist", "reliability", "crps", "rmse"]].mean(axis=1))

        no_df = pd.DataFrame(
            np.array([
                np.arange(len(no_rank_hist_score)),
                no_rank_hist_score,
                no_reliability_score,
                no_crps_scores,
                no_rmse_scores]).T,
            columns=["day", "rank_hist", "reliability", "crps", "rmse"]
        )
        no_df["type"] = "no"
        no_df['average'] = np.argsort(no_df[["rank_hist", "reliability", "crps", "rmse"]].mean(axis=1))

        fixed_df = pd.DataFrame(
            np.array([
                np.arange(len(fixed_rank_hist_score)),
                fixed_rank_hist_score,
                fixed_reliability_score,
                fixed_crps_scores,
                fixed_rmse_scores]).T,
            columns=["day", "rank_hist", "reliability", "crps", "rmse"]
        )
        fixed_df["type"] = "fixed"
        fixed_df['average'] = np.argsort(fixed_df[["rank_hist", "reliability", "crps", "rmse"]].mean(axis=1))

        smooth_df = pd.DataFrame(
            np.array([
                np.arange(len(smooth_rank_hist_score)),
                smooth_rank_hist_score,
                smooth_reliability_score,
                smooth_crps_scores,
                smooth_rmse_scores]).T,
            columns=["day", "rank_hist", "reliability", "crps", "rmse"]
        )
        smooth_df["type"] = "smooth"
        smooth_df['average'] = np.argsort(smooth_df[["rank_hist", "reliability", "crps", "rmse"]].mean(axis=1))

        rank_df = pd.concat([adapt_df, no_df, fixed_df, smooth_df])
        rank_df.to_csv(args.out_dir + f"{pp}_rank_df.csv", index=False)
        logger.info(f"{pp} saved csv")

logger.info("DONE")
