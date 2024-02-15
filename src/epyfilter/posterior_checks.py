import numpy as np


def check_param_in_ci(kf, day, percentile=95):
    """
    Check if the parameter estimate falls within the credible interval.

    Args:
        kf (object): The EnsembleAdjustmentKalmanFilter object.
        day (int or str): The day for which to check the parameter estimate.
                          If "last", checks the last day.
        percentile (float, optional): The percentile for credible interval. Default is 95.

    Returns:
        bool: True if the parameter estimate is within the credible interval, False otherwise.
    """
    post_betas = np.asarray([θ.beta * θ.t_I for θ in kf.θ_list])
    quantiles = [(1 - percentile / 100) / 2, 1 - (1 - percentile / 100) / 2]
    quantiles_beta = np.quantile(post_betas, q=quantiles, axis=1)
    lower = quantiles_beta[0, :]
    upper = quantiles_beta[1, :]

    if day == "last":
        return lower[-1] <= kf.data.beta[-1] * kf.data.t_I <= upper[-1]
    else:
        assert isinstance(day, int), "day must be integer"
        if day >= len(kf.data.beta):
            # day is past the end of the time series
            return lower[-1] <= kf.data.beta[-1] * kf.data.t_I <= upper[-1]
        else:
            return lower[day] <= kf.data.beta[day] * kf.data.t_I <= upper[day]


def compute_ens_var(kf, day):
    """
    Compute the ensemble variance of parameter estimates.

    Args:
        kf (object): The EnsembleAdjustmentKalmanFilter object.
        day (int or str): The day for which to compute the ensemble variance.
                          If "last", computes for the last day.

    Returns:
        float: The ensemble variance.
    """
    if day == "last":
        return np.var(np.asarray([θ.beta * θ.t_I for θ in kf.θ_list])[-1])
    else:
        assert isinstance(day, int), "day must be integer"
        if day >= len(kf.θ_list):
            # day is past the end of the time series
            return np.var(np.asarray([θ.beta * θ.t_I for θ in kf.θ_list])[-1])
        else:
            return np.var(np.asarray([θ.beta * θ.t_I for θ in kf.θ_list])[day])


def avg_kl_divergence(kf, last_epi=False, last_epi_day=None, min_i=100, num_bins=10):
    """
    Compute the average Kullback-Leibler divergence.

    Args:
        kf (object): The EnsembleAdjustmentKalmanFilter object.
        last_epi (bool, optional): If True, computes only for days up to last_epi_day.
        last_epi_day (int, optional): Last day of the second epidemic. Defaults to None.
        min_i (int, optional): Minimum number of infected individuals to consider. Defaults to 100.
        num_bins (int, optional): Number of bins for histogram. Defaults to 10.

    Returns:
        float: The average Kullback-Leibler divergence.
    """
    if not hasattr(kf, "i_ppc"):
        betas = np.asarray([θ.beta for θ in kf.θ_list])
        _, _, _, _ = kf.free_sim(betas)

    kl_list = []
    days = np.where(kf.data.i_true >= min_i)[0]
    if last_epi:
        # compute only for days less than the day when the second epidemic ends
        assert isinstance(last_epi_day, int), "peaks must be integer"
        last_epi_day = min(last_epi_day, kf.data.n_t - 1)
        days = days[days <= last_epi_day]

    for t in days:
        # Calculate KL divergence
        kl = kl_divergence(
            kf.data.data_distribution[t, :], kf.i_ppc[t, :], num_bins=num_bins
        )
        kl_list.append(kl)
    return np.mean(kl_list)


def avg_wasserstein2(kf, last_epi=False, last_epi_day=None, min_i=100, num_bins=10):
    """
    Compute the average Wasserstein-2 distance.

    Args:
        kf (object): The EnsembleAdjustmentKalmanFilter object.
        last_epi (bool, optional): If True, computes only for days up to last_epi_day.
        last_epi_day (int, optional): Last day of the second epidemic. Defaults to None.
        min_i (int, optional): Minimum number of infected individuals to consider. Defaults to 100.
        num_bins (int, optional): Number of bins for histogram. Defaults to 10.

    Returns:
        float: The average Wasserstein-2 distance.
    """
    if not hasattr(kf, "i_ppc"):
        betas = np.asarray([θ.beta for θ in kf.θ_list])
        _, _, _, _ = kf.free_sim(betas)
    w2_list = []
    days = np.where(kf.data.i_true >= min_i)[0]
    if last_epi:
        # compute only for days less than the day when the second epidemic ends
        assert isinstance(last_epi_day, int), "peaks must be integer"
        last_epi_day = min(last_epi_day, kf.data.n_t - 1)
        days = days[days <= last_epi_day]

    for t in days:
        # Calculate w2
        w2 = wasserstein2(
            kf.data.data_distribution[t, :], kf.i_ppc[t, :], num_bins=num_bins
        )
        w2_list.append(w2)
    return np.mean(w2_list)


def avg_kl_divergence_ks(ks, last_epi=False, last_epi_day=None, min_i=100, num_bins=10):
    """
    Compute the average Kullback-Leibler divergence for EnsembleSquareRootSmoother.

    Args:
        ks (object): The EnsembleSquareRootSmoother object.
        last_epi (bool, optional): If True, computes only for days up to last_epi_day.
        last_epi_day (int, optional): Last day of the second epidemic. Defaults to None.
        min_i (int, optional): Minimum number of infected individuals to consider. Defaults to 100.
        num_bins (int, optional): Number of bins for histogram. Defaults to 10.

    Returns:
        float: The average Kullback-Leibler divergence.
    """
    if not hasattr(ks, "i_ppc"):
        betas = np.asarray([θ.beta for θ in ks.θ_lag_list])
        _, _, _, _ = ks.free_sim(betas)

    kl_list = []
    days = np.where(ks.data.i_true >= min_i)[0]
    days = days[days < len(ks.θ_lag_list)]
    if last_epi:
        # compute only for days less than the day when the second epidemic ends
        assert isinstance(last_epi_day, int), "peaks must be integer"
        last_epi_day = min(last_epi_day, ks.data.n_t - 1)
        days = days[days <= last_epi_day]
    for t in days:
        # Calculate KL divergence
        kl = kl_divergence(
            ks.data.data_distribution[t, :], ks.i_ppc[t, :], num_bins=num_bins
        )
        kl_list.append(kl)
    return np.mean(kl_list)


def avg_wasserstein2_ks(ks, last_epi=False, last_epi_day=None, min_i=100, num_bins=10):
    """
    Compute the average Wasserstein-2 distance for EnsembleSquareRootSmoother.

    Args:
        ks (object): The EnsembleSquareRootSmoother object.
        last_epi (bool, optional): If True, computes only for days up to last_epi_day.
        last_epi_day (int, optional): Last day of the second epidemic. Defaults to None.
        min_i (int, optional): Minimum number of infected individuals to consider. Defaults to 100.
        num_bins (int, optional): Number of bins for histogram. Defaults to 10.

    Returns:
        float: The average Wasserstein-2 distance.
    """

    if not hasattr(ks, "i_ppc"):
        betas = np.asarray([θ.beta for θ in ks.θ_lag_list])
        _, _, _, _ = ks.free_sim(betas)
    w2_list = []
    days = np.where(ks.data.i_true >= min_i)[0]
    days = days[days < len(ks.θ_lag_list)]
    if last_epi:
        # compute only for days less than the day when the second epidemic ends
        assert isinstance(last_epi_day, int), "peaks must be integer"
        last_epi_day = min(last_epi_day, ks.data.n_t - 1)
        days = days[days <= last_epi_day]
    for t in days:
        # Calculate w2
        w2 = wasserstein2(
            ks.data.data_distribution[t, :], ks.i_ppc[t, :], num_bins=num_bins
        )
        w2_list.append(w2)
    return np.mean(w2_list)


def compute_ens_var_ks(ks, day):
    """
    Compute the ensemble variance of parameter estimates for EnsembleSquareRootSmoother.

    Args:
        ks (object): The EnsembleSquareRootSmoother object.
        day (int or str): The day for which to compute the ensemble variance.
                          If "last", computes for the last day.

    Returns:
        float: The ensemble variance.
    """
    if day == "last":
        return np.var(np.asarray([θ.beta * θ.t_I for θ in ks.θ_lag_list])[-1])
    else:
        assert isinstance(day, int), "day must be integer"
        if day > len(ks.θ_lag_list):
            day = len(ks.θ_lag_list) - 1
        return np.var(np.asarray([θ.beta * θ.t_I for θ in ks.θ_lag_list])[day])


def check_param_in_ci_ks(ks, day, percentile=95):
    """
    Check if the parameter estimate falls within the confidence interval for EnsembleSquareRootSmoother.

    Args:
        ks (object): The EnsembleSquareRootSmoother object.
        day (int or str): The day for which to check the parameter estimate.
                          If "last", checks the last day.
        percentile (float, optional): The percentile for confidence interval. Default is 95.

    Returns:
        bool: True if the parameter estimate is within the confidence interval, False otherwise.
    """
    post_betas = np.asarray([θ.beta * θ.t_I for θ in ks.θ_lag_list])
    quantiles = [(1 - percentile / 100) / 2, 1 - (1 - percentile / 100) / 2]
    quantiles_beta = np.quantile(post_betas, q=quantiles, axis=1)
    lower = quantiles_beta[0, :]
    upper = quantiles_beta[1, :]
    if day == "last":
        return lower[-1] <= ks.data.beta[-1] * ks.data.t_I <= upper[-1]
    else:
        assert isinstance(day, int), "day must be integer"
        if day > len(ks.θ_lag_list):
            day = len(ks.θ_lag_list) - 1
        return lower[day] <= ks.data.beta[day] * ks.data.t_I <= upper[day]


def bin_data(d, d_pp, num_bins, bins=None):
    data_min = min(np.min(d), np.min(d_pp))
    data_max = max(np.max(d), np.max(d_pp))
    if bins is None:
        bins = np.linspace(data_min, data_max, num_bins + 1)
    digitized = np.digitize(d, bins)
    digitized_pp = np.digitize(d_pp, bins)

    # Adjust bin indices to start from 0
    return digitized - 1, digitized_pp - 1, bins


def compute_probs(digitized_data, num_bins):
    counts = np.bincount(digitized_data, minlength=num_bins)
    probs = counts / len(digitized_data)
    return probs


def kl_divergence(p_sample, q_sample, epsilon=1e-10, num_bins=10):
    p_bins, q_bins, bins = bin_data(p_sample, q_sample, num_bins)

    p_probs = compute_probs(p_bins, num_bins + 1)
    q_probs = compute_probs(q_bins, num_bins + 1)
    assert len(p_probs) == len(q_probs)

    p_safe = p_probs + epsilon
    q_safe = q_probs + epsilon

    return np.sum(p_safe * np.log(p_safe / q_safe))


def wasserstein2(p_sample, q_sample, num_bins=10):
    assert len(p_sample) == len(q_sample)
    p_bins, q_bins, bins = bin_data(p_sample, q_sample, num_bins)
    p_probs = compute_probs(p_bins, num_bins + 1)
    q_probs = compute_probs(q_bins, num_bins + 1)
    assert len(p_probs) == len(q_probs)
    assert np.isclose(np.sum(p_probs), 1.0), "Dist p must sum to 1"
    assert np.isclose(np.sum(q_probs), 1.0), "Dist q must sum to 1"

    # Compute the cumulative sums
    p_cdf = np.cumsum(p_probs)
    q_cdf = np.cumsum(q_probs)

    # Calculate the squared distances between the cumulative sums
    squared_distances = (p_cdf - q_cdf) ** 2

    # Compute the W-2 metric
    w2 = np.sqrt(np.sum(squared_distances))

    return w2


def data_rmse(kf, last_epi_day=None, last_epi=False):
    """
    Compute the root mean square error (RMSE) between the data and case counts posterior.

    Args:
        kf (object): The EnsembleAdjustmentKalmanFilter object.
        last_epi_day (int, optional): Last day of the second epidemic. Defaults to None.
        last_epi (bool, optional): If True, compute RMSE up to last_epi_day.

    Returns:
        float: The RMSE.
    """
    if last_epi:
        i_kf = np.array([x.i for x in kf.x_list])
        i_kf = i_kf[:last_epi_day]
        rmse = np.sqrt(np.mean((i_kf.T - kf.data.i[1 : len(i_kf) + 1]) ** 2, axis=0))
    else:
        i_kf = np.array([x.i for x in kf.x_list])
        rmse = np.sqrt(np.mean((i_kf.T - kf.data.i[1 : len(i_kf) + 1]) ** 2, axis=0))
        spread = np.sqrt(np.var(i_kf, axis=1))

    return np.mean(rmse)


def rt_rmse(kf, last_epi=False, peaks=None):
    """
    Compute the root mean square error (RMSE) between the truth and Rt posterior.

    Args:
        kf (object): The EnsembleAdjustmentKalmanFilter object.
        last_epi (bool, optional): If True, compute RMSE up to last_epi_day.
        peaks (int or list of int, optional): Peak day(s) of epidemic(s). Defaults to None.

    Returns:
        float: The RMSE.
    """
    rt_kf = np.array([θ.beta * θ.t_I for θ in kf.θ_list])

    if last_epi:
        # compute rmse from day 1 until the last day of the second epidemic
        assert isinstance(peaks, int), "peaks must be integer"
        peaks = min(peaks, kf.data.n_t - 1)

        rmse = np.sqrt(np.mean((rt_kf[:peaks].T - kf.data.rt[:peaks]) ** 2, axis=0))
        return np.mean(rmse)
    elif peaks is not None:
        # computes rmse on specific day(s)
        if isinstance(peaks, list):
            peaks = [min(p, kf.data.n_t - 1) for p in peaks]
        elif isinstance(peaks, int):
            peaks = min(peaks, kf.data.n_t - 1)
        else:
            raise ValueError("peaks must be a list or an integer")

        rmse = np.sqrt(np.mean((rt_kf[peaks].T - kf.data.rt[peaks]) ** 2, axis=0))
        return np.mean(rmse)
    else:
        # computes rmse for entire time series
        rmse = np.sqrt(np.mean((rt_kf.T - kf.data.rt[: len(rt_kf)]) ** 2, axis=0))
    return np.mean(rmse)
