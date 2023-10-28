import numpy as np

def check_param_in_ci(kf, day, percentile=95):
    assert isinstance(day, int), "day must be integer"
    post_betas = np.asarray([θ.beta for θ in kf.θ_list])
    quantiles = [(1-percentile/100)/2, 1-(1-percentile/100)/2]
    quantiles_beta = np.quantile(post_betas, q=quantiles, axis=1)
    lower = quantiles_beta[0, :]
    upper = quantiles_beta[1, :]

    return lower[day] <= kf.data.beta[day] <= upper[day]


def avg_kl_divergence(kf, min_i=100, num_bins=10):
    if not hasattr(kf, "i_ppc"):
        betas = np.asarray([θ.beta for θ in kf.θ_list])
        _, _, _, _ = kf.free_sim(betas)

    kl_list = []
    days = np.where(kf.data.i_true >= min_i)[0]
    for t in days:
        # Calculate KL divergence
        kl = kl_divergence(kf.data.data_distribution[t, :],
                           kf.i_ppc[t, :],
                           num_bins=num_bins)
        kl_list.append(kl)
    return np.mean(kl_list)


def avg_wasserstein2(kf, min_i=100, num_bins=10):
    if not hasattr(kf, "i_ppc"):
        betas = np.asarray([θ.beta for θ in kf.θ_list])
        _, _, _, _ = kf.free_sim(betas)
    w2_list = []
    days = np.where(kf.data.i_true >= min_i)[0]
    for t in days:
        # Calculate w2
        w2 = wasserstein2(kf.data.data_distribution[t, :],
                          kf.i_ppc[t, :],
                          num_bins=num_bins)
        w2_list.append(w2)
    return np.mean(w2_list)


def avg_kl_divergence_ks(ks, min_i=100, num_bins=10):
    if not hasattr(ks, "i_ppc"):
        betas = np.asarray([θ.beta for θ in ks.θ_lag_list])
        _, _, _, _ = ks.free_sim(betas)

    kl_list = []
    days = np.where(ks.data.i_true >= min_i)[0]
    days = days[days < len(ks.θ_lag_list)]
    for t in days:
        # Calculate KL divergence
        kl = kl_divergence(ks.data.data_distribution[t, :],
                           ks.i_ppc[t, :],
                           num_bins=num_bins)
        kl_list.append(kl)
    return np.mean(kl_list)


def avg_wasserstein2_ks(ks, min_i=100, num_bins=10):
    if not hasattr(ks, "i_ppc"):
        betas = np.asarray([θ.beta for θ in ks.θ_lag_list])
        _, _, _, _ = ks.free_sim(betas)
    w2_list = []
    days = np.where(ks.data.i_true >= min_i)[0]
    days = days[days < len(ks.θ_lag_list)]
    for t in days:
        # Calculate w2
        w2 = wasserstein2(ks.data.data_distribution[t, :],
                          ks.i_ppc[t, :],
                          num_bins=num_bins)
        w2_list.append(w2)
    return np.mean(w2_list)


def check_param_in_ci_ks(ks, day, percentile=95):
    assert isinstance(day, int), "day must be integer"
    assert day < len(ks.θ_lag_list)
    post_betas = np.asarray([θ.beta for θ in ks.θ_lag_list])
    quantiles = [(1-percentile/100)/2, 1-(1-percentile/100)/2]
    quantiles_beta = np.quantile(post_betas, q=quantiles, axis=1)
    lower = quantiles_beta[0, :]
    upper = quantiles_beta[1, :]

    return lower[day] <= ks.data.beta[day] <= upper[day]


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

    p_probs = compute_probs(p_bins, num_bins+1)
    q_probs = compute_probs(q_bins, num_bins+1)
    assert len(p_probs) == len(q_probs)

    p_safe = p_probs + epsilon
    q_safe = q_probs + epsilon
    np.sum(p_safe * np.log(p_safe / q_safe))
    return np.sum(p_safe * np.log(p_safe / q_safe))


def wasserstein2(p_sample, q_sample, num_bins=10):
    assert len(p_sample) == len(q_sample)
    p_bins, q_bins, bins = bin_data(p_sample, q_sample, num_bins)
    p_probs = compute_probs(p_bins, num_bins+1)
    q_probs = compute_probs(q_bins, num_bins+1)
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

def data_rmse(kf):
    i_kf = np.array([x.i for x in kf.x_list])
    rmse = np.sqrt(np.mean((i_kf.T - kf.data.i[1:len(i_kf)+1])**2, axis=0))
    spread = np.sqrt(np.var(i_kf, axis=1))

    return np.mean(rmse)

def rt_rmse(kf, peaks=None):
    if peaks is not None:
        rt_kf = np.array([θ.beta * θ.t_I for θ in kf.θ_list])
        rmse = np.sqrt(np.mean((rt_kf[peaks].T - kf.data.rt[peaks])**2, axis=0))
    else:
        rt_kf = np.array([θ.beta * θ.t_I for θ in kf.θ_list])
        rmse = np.sqrt(np.mean((rt_kf.T - kf.data.rt[:len(rt_kf)])**2, axis=0))

    return np.mean(rmse)