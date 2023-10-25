import os
import argparse
import numpy as np
import pandas as pd
from epyfilter import simulate_data


if __name__ == '__main__':
    sge_task_id = os.environ.get("SGE_TASK_ID")

    parser = argparse.ArgumentParser(
        description="Generate synthetic data.",
    )
    parser.add_argument(
        "--in-dir", type=str, required=True,
        help="The version the migration data are saved under.")
    parser.add_argument(
        "--out-dir", type=str, required=True,
        help="The version the model results are saved under.")
    parser.add_argument(
        "--model-name", type=str, required=True,
        help="The name of the model.")
    args = parser.parse_args()

    np.random.seed(1994)

    df = pd.read_csv(os.path.join(args.in_dir, "param_list.csv"))
    params = df.iloc[sge_task_id - 1].to_dict()
    data = simulate_data.simulate_data(
        **params, add_noise=True, noise_param=1/50)

    data.plot_all(path=args.out_dir, name=f"{sge_task_id}_synthetic_plots")
    data.compute_data_distribution()
    data.save_data(path=args.out_dir, name=f"{sge_task_id}_synthetic_data")
