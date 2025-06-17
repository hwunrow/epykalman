import pickle
import os
import argparse
import logging
from tqdm import tqdm
import pandas as pd
import numpy as np


def open_pickle(pickle_file):
    with open(pickle_file, "rb") as file:
        data = pickle.load(file)

    return data


if __name__ == "__main__":
    try:
        sge_task_id = int(os.environ.get("SGE_TASK_ID"))
        sge_outputs_file = os.environ.get("SGE_STDOUT_PATH")
        files_per_task = 1000
    except:
        sge_task_id = 1
        sge_outputs_file = "test.log"
        files_per_task = 100_185

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
        description="Run EAKF with adaptive, fixed, and no inflation for 1000 different synthetic data sets",
    )
    parser.add_argument(
        "--in-dir", type=str, required=True, help="Directory for inputs."
    )
    parser.add_argument(
        "--pkl-dir",
        type=str,
        required=True,
        help="Directory for synthetic data inputs.",
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
    args = parser.parse_args()

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
            f"{args.pkl_dir}/{pp}_synthetic_data.pkl" for pp in args.param_list
        ]

    for i, pickle_file in enumerate(tqdm(pickle_files)):
        data = open_pickle(pickle_file)
        param_num = os.path.basename(pickle_file).split("_")[0]
        tmp_df = pd.DataFrame(
            np.stack([data.rt, data.i[1:], data.S[1:] / data.N]).T,
            columns=["rt", "i", "prop_S"],
        )
        tmp_df.to_csv(f"{args.out_dir}/{param_num}_for_epiestim.csv", index=False)
        logger.info(param_num)

    logger.info("Done!")
