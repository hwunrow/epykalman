import pickle
from epyfilter import simulate_data
from tqdm import tqdm
import pandas as pd
import numpy as np
import os
import logging
import argparse


def compute_late_day(data):
    beta_1 = data.rt_1 / data.t_I
    beta_0 = data.rt_0 / data.t_I
    late_day = (
        -1 / data.k * np.log((beta_1 - beta_0) / (0.99 * beta_1 - beta_0) - 1)
        + data.midpoint
    )
    late_day = int(late_day)
    return late_day


def compute_peaks(data):
    det_data = simulate_data.simulate_data(**data.true_params, run_deterministic=True)
    (peak_days,) = np.where(
        np.diff(np.sign(np.diff(det_data.i_true))) == -2
    )  # days where it increases before then decreases
    peak_days = peak_days[:2]  # just take first two days
    return peak_days


def compute_last_epi_day(data):
    if len(np.where(data.i_true == 0)[0]) == 1:
        last_epi_day = len(data.i_true)
    else:
        last_epi_day = np.where(data.i_true == 0)[0][1]
    return last_epi_day


def open_pickle(pickle_file):
    with open(pickle_file, "rb") as file:
        data = pickle.load(file)
    return data


if __name__ == "__main__":
    sge_outputs_file = os.environ.get("SGE_STDOUT_PATH")
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
        description="calculate late day, last epi day, peaks, and save data distribution csvs.",
    )
    parser.add_argument(
        "--in-dir", type=str, required=True, help="Directory for inputs."
    )
    parser.add_argument(
        "--synthetic-dir", type=str, required=True, help="Directory for synthetic data."
    )
    parser.add_argument(
        "--out-dir", type=str, required=True, help="Directory to save plots and files."
    )
    parser.add_argument(
        "--param-list", type=int, nargs="+", help="Rerunning for specific sge_task_ids"
    )
    args = parser.parse_args()

    if not args.param_list:  # if no param list is provided, do all params
        df = pd.read_csv(os.path.join(args.in_dir, "good_param_list.csv"))
        args.param_list = df["param"].values

    dates = []
    for pp in tqdm(args.param_list):
        # load data
        file = f"{args.synthetic_dir}/{pp}_synthetic_data.pkl"
        data = open_pickle(file)

        # compute late day, peaks, and last epi day
        late_day = compute_late_day(data)
        peaks = compute_peaks(data)
        last_epi_day = compute_last_epi_day(data)

        dates.append(
            {
                "param": pp,
                "late_day": late_day,
                "peak1": peaks[0],
                "peak2": peaks[1],
                "last_epi_day": last_epi_day,
            }
        )

        # save data distribution csv
        data_distribution_df = pd.DataFrame(
            data.data_distribution,
            columns=[
                f"sample{x}" for x in range(1, data.data_distribution.shape[1] + 1)
            ],
        )
        data_distribution_df["day"] = range(len(data_distribution_df))
        data_distribution_df["late_day"] = compute_late_day(data)
        data_distribution_df["peak1"] = peaks[0]
        data_distribution_df["peak2"] = peaks[1]
        data_distribution_df["last_epi_day"] = last_epi_day
        data_distribution_df.to_csv(
            f"{args.out_dir}/{pp}_data_distribution.csv", index=False
        )
        logger.info(f"{pp}")
    dates_df = pd.DataFrame(dates)
    dates_df.to_csv(f"{args.out_dir}/compute_days.csv", index=False)

    logger.info("Done")
