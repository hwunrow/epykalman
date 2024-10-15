import os
import logging
import argparse
import numpy as np
import pandas as pd
from epyfilter import simulate_data


if __name__ == "__main__":
    try:
        sge_task_id = int(os.environ.get("SGE_TASK_ID"))
        sge_outputs_file = os.environ.get("SGE_STDOUT_PATH")
    except:
        sge_task_id = 1
        sge_outputs_file = "test.log"

    parser = argparse.ArgumentParser(
        description="Generate synthetic data.",
    )
    parser.add_argument(
        "--in-dir",
        type=str,
        required=True,
        help="The version the migration data are saved under.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        required=True,
        help="The version the model results are saved under.",
    )
    parser.add_argument(
        "--param-list", type=int, nargs="+", help="Rerunning for specific sge_task_ids"
    )
    args = parser.parse_args()

    logger = logging.getLogger("my_logger")
    logger.setLevel(logging.INFO)

    sg_outputs_handler = logging.FileHandler(sge_outputs_file)
    sg_outputs_handler.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    sg_outputs_handler.setFormatter(formatter)
    logger.addHandler(sg_outputs_handler)

    np.random.seed(1994)

    df = pd.read_csv(os.path.join(args.in_dir, "param_list.csv"))
    if args.param_list is not None:
        logger.info(args.param_list)
        for pp in args.param_list:
            params = df.iloc[pp - 1].to_dict()
            data = simulate_data.simulate_data(
                add_noise=True, noise_param=1 / 50, **params
            )
            data.plot_all(path=args.out_dir, name=f"{pp}_synthetic_plots")
            data.compute_data_distribution()
            data.save_data(path=args.out_dir, name=f"{pp}_synthetic_data")
    else:
        params = df.iloc[sge_task_id - 1].to_dict()
        data = simulate_data.simulate_data(add_noise=True, noise_param=1 / 50, **params)
        data.plot_all(path=args.out_dir, name=f"{sge_task_id}_synthetic_plots")
        data.compute_data_distribution()
        data.save_data(path=args.out_dir, name=f"{sge_task_id}_synthetic_data")

    logger.info("Done!")
