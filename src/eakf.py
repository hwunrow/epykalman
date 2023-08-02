import numpy as np
from tqdm import tqdm
import inflation
import matplotlib.pyplot as plt
import simulate_data


class EnsembleAdjustmentKalmanFilter():
    def __init__(self, model, m):
        """
        model: SIR_model
        m: number of ensemble members
        """
        self.f = model.f
        self.h = model.h
        self.f0 = model.f0
        self.θ0 = model.θ0
        self.data = model.data
        self.m = m
        self.SIR = model.SIR
        self.ModelParams = model.ModelParams

    def oev(self, z):
        return np.maximum(10, z**2/50)

    def eakf(self, x, y, z, oev):
        """
        Args:
            x: latenet state rv or latent parameter rv
            y: observed rv
            z: observation
            oev: observational error variance
        """
        x = np.array(x)
        y = np.array(y)

        p, m = x.shape

        mu_prior = y.mean()
        var_prior = y.var()

        # degenerate prior.
        var_prior = np.where(var_prior == 0., 1e-3, var_prior)

        var_post = var_prior * oev / (var_prior + oev)
        mu_post = var_post * (mu_prior/var_prior + z/oev)
        alpha = (oev / (oev + var_prior)) ** (0.5)
        dy = (mu_post - y) + alpha * (y - mu_prior)

        # compute covariance
        rr = np.full((p, 1), np.nan)

        for ip in range(p):
            A = np.cov(x[ip, :], y)
            rr[ip, :] = A[1, 0] / var_prior
        dx = np.dot(rr, dy.reshape((1, self.m)))
        # rr = np.cov(x, y)[:-1,-1] / var_prior
        # dx_new = np.dot(rr.reshape((4,1)), dy.reshape((1,300)))

        xpost = x + dx
        ypost = y + dy

        return xpost, ypost, alpha

    def filter(self, prior, inf_method="adaptive", lam_fixed=1.01):
        x_list = []
        xhat_list = []
        θ_list = []
        lam_list = []
        alpha_list = []

        lam = lam_S = lam_I = lam_R = lam_i = 1.01

        for t in tqdm(range(self.data.n_t)):
            if t == 0:
                x = self.f0(self.data.N, m=self.m)
                θ = self.θ0(prior, m=self.m)
                xhat_list.append(x)
            else:
                x = self.f(t, x, θ, self.data.N)
                xhat_list.append(x)
                y = self.h(x)
                z = self.data.i[t]
                oev = self.oev(z)
                if t > 50:  # inflate after 50 days
                    if inf_method == "adaptive":
                        lam = inflation.adaptive_inflation(
                            θ.beta, y, z, oev, lambar_prior=lam)
                        lam_S = inflation.adaptive_inflation(x.S, y, z, oev)
                        lam_I = inflation.adaptive_inflation(x.I, y, z, oev)
                        lam_R = inflation.adaptive_inflation(x.R, y, z, oev)
                        lam_i = inflation.adaptive_inflation(x.i, y, z, oev)
                    elif inf_method == "constant":
                        if t < 250:
                            lam = lam_fixed
                            lam_S = lam_I = lam_R = lam_i = 1.01
                        else:
                            lam = lam_S = lam_I = lam_R = lam_i = 1.
                    else:
                        lam = lam_S = lam_I = lam_R = lam_i = 1.
                    # if z > 0:
                    #     if np.abs(np.mean(y) - z) / z > 1:
                    #         turn_off = True
                    #         lam = 1.
                    #         lam_S = 1.
                    #         lam_I = 1.
                    #         lam_R = 1.
                    #         lam_i = 1.
                    if np.isnan(lam):
                        lam = 1.
                    if np.isnan(lam_S):
                        lam_S = 1.
                    if np.isnan(lam_I):
                        lam_I = 1.
                    if np.isnan(lam_R):
                        lam_R = 1.
                    if np.isnan(lam_i):
                        lam_i = 1.
                    # if turn_off:
                    #     lam = 1.
                    #     lam_S = 1.
                    #     lam_I = 1.
                    #     lam_R = 1.
                    #     lam_i = 1.
                    lam_list.append(lam)
                    θ = inflation.inflate_ensemble(
                        θ, inflation_value=lam, params=True)
                    θ = np.clip(θ, 0, 10)
                    θ = self.ModelParams(*θ)

                    S = inflation.inflate_ensemble(x.S, inflation_value=lam_S)
                    Ir = inflation.inflate_ensemble(x.I, inflation_value=lam_I)
                    R = inflation.inflate_ensemble(x.R, inflation_value=lam_R)
                    i = inflation.inflate_ensemble(x.i, inflation_value=lam_i)

                    x = self.SIR(
                        S=np.clip(S, 0, self.data.N),
                        I=np.clip(Ir, 0, self.data.N),
                        R=np.clip(R, 0, self.data.N),
                        i=np.clip(i, 0, self.data.N)
                    )

                x, new_i, alpha = self.eakf(x, y, z, oev=oev)
                x = self.SIR(*x)
                x = x._replace(i=new_i)
                x = self.SIR(
                    S=np.clip(x.S, 0, self.data.N),
                    I=np.clip(x.I, 0, self.data.N),
                    R=np.clip(x.R, 0, self.data.N),
                    i=np.clip(x.i, 0, self.data.N)
                )
                θ, _, alpha_θ = self.eakf(θ, y, z, oev=oev)
                θ = np.clip(θ, 0, None)
                θ = self.ModelParams(*θ)

                alpha_list.append([alpha, alpha_θ])

            x_list.append(x)
            θ_list.append(θ)

        self.xhat_list = np.array(xhat_list)
        self.x_list = np.array(x_list)
        self.θ_list = np.array(θ_list)
        self.lam_list = np.array(lam_list)
        self.alpha_list = np.array(alpha_list)

    def free_sim(self, beta):
        S = np.array([self.data.S0 * np.ones(300)])
        Ir = np.array([self.data.I0 * np.ones(300)])
        R = np.array([np.zeros(300)])
        i = np.array([np.zeros(300)])

        for t in range(self.data.beta.shape[0]):
            if t < 10:
                dSI = np.random.poisson(self.data.rt_0/self.data.t_I * Ir[t] *
                                        S[t] / self.data.N)
            else:
                dSI = np.random.poisson(beta[t]*Ir[t]*S[t]/self.data.N)
            dIR = np.random.poisson(Ir[t]/self.data.t_I)

            S_new = np.clip(S[t]-dSI, 0, self.data.N)
            I_new = np.clip(Ir[t]+dSI-dIR, 0, self.data.N)
            R_new = np.clip(R[t]+dIR, 0, self.data.N)

            S = np.append(S, [S_new], axis=0)
            Ir = np.append(Ir, [I_new], axis=0)
            R = np.append(R, [R_new], axis=0)
            i = np.append(i, [dSI], axis=0)

        self.i_ppc = i

        return S, Ir, R, i

    def plot_reliability(self):
        betas = np.asarray([θ.beta for θ in self.θ_list])
        S, Ir, R, i = self.free_sim(betas)

        fig, ax = plt.subplots(1)
        ax.plot(i, color='gray', alpha=0.1)
        ax.plot(self.data.i_true, 'x', color='black', label="truth")
        ax.set_title("free simulation - all ensembles")

        ci = np.quantile(i, q=[0.025, 0.975], axis=1)
        ci_50 = np.quantile(i, q=[0.25, 0.75], axis=1)

        fig, ax = plt.subplots(1)

        ax.fill_between(np.arange(0, 366), ci[0], ci[1], facecolor='gray',
                        alpha=0.5, label='95% CI')
        ax.fill_between(np.arange(0, 366), ci_50[0], ci_50[1],
                        facecolor='gray', alpha=0.75, label='50% CI')
        ax.plot(self.data.i_true, 'x', color='black', label="truth")
        ax.legend(loc='upper left')

        ax.set_title("free simulation")

        prop_95 = np.mean((ci[0] <= self.data.i) & (self.data.i <= ci[1]))
        prop_50 = np.mean(
            (ci_50[0] <= self.data.i) & (self.data.i <= ci_50[1]))

        print(f"Percent of observations in 95% CI {round(prop_95*100, 2)}%")
        print(f"Percent of observations in 50% CI {round(prop_50*100, 2)}%")

        prop_list = []
        percentiles = np.arange(2.5, 100, 2.5)
        for p in percentiles:
            lower = np.quantile(
                i, q=[(1-p/100)/2, 1-(1-p/100)/2], axis=1)[0, :]
            upper = np.quantile(
                i, q=[(1-p/100)/2, 1-(1-p/100)/2], axis=1)[1, :]
            prop_list.append(np.mean(
                (lower <= self.data.i) & (self.data.i <= upper)))

        fig, ax = plt.subplots(1)
        ax.plot(percentiles/100, prop_list, '-.')
        ax.axline((0, 0), (1, 1), color='r')
        ax.set_xlabel('CI')
        ax.set_ylabel('% of obs within CI')
        ax.set_title("observation reliability plot")

        betas = np.asarray([θ.beta for θ in self.θ_list])
        betas_skip = betas[15:, :]
        beta_true = self.data.beta
        beta_true = beta_true[15:]

        prop_list = []
        for p in percentiles:
            lower = np.quantile(
                betas_skip, q=[(1-p/100)/2, 1-(1-p/100)/2], axis=1)[0, :]
            upper = np.quantile(
                betas_skip, q=[(1-p/100)/2, 1-(1-p/100)/2], axis=1)[1, :]
            prop_list.append(
                np.mean((lower <= beta_true) & (beta_true <= upper)))

        fig, ax = plt.subplots(1)
        ax.plot(percentiles/100, prop_list, '-.')
        ax.axline((0, 0), (1, 1), color='r')
        ax.set_xlabel('CI')
        ax.set_ylabel(r'% of $\beta$ within CI')
        ax.set_title(r"$\beta$ reliability plot")

        # for p in percentiles:
        #     S, Ir, R, i = self.free_sim(np.quantile(betas_skip, q=p/100))

    def plot_posterior(self):
        fig, ax = plt.subplots(3)
        ax[0].plot([x.S for x in self.x_list], color='gray', alpha=0.1)
        ax[0].plot(np.mean([x.S for x in self.x_list], axis=1), color='black')
        ax[0].plot(self.data.S, '.')

        ax[1].plot([x.I for x in self.x_list], color='gray', alpha=0.1)
        ax[1].plot(np.mean([x.I for x in self.x_list], axis=1), color='black')
        ax[1].plot(self.data.I, '.')

        ax[2].plot([x.R for x in self.x_list], color='gray', alpha=0.1)
        ax[2].plot(np.mean([x.R for x in self.x_list], axis=1), color='black')
        ax[2].plot(self.data.R, '.')

        fig, ax = plt.subplots(3)
        ax[0].plot([x.i for x in self.x_list], color='gray', alpha=0.1)
        ax[0].plot(np.mean([x.i for x in self.x_list], axis=1), color='black')
        ax[0].plot(self.data.i, '.')
        ax[0].set_ylabel('daily case counts')

        ax[1].plot([θ.beta for θ in self.θ_list], color="gray", alpha=0.1)
        ax[1].plot(
            np.mean([θ.beta for θ in self.θ_list], axis=1), color="black")
        ax[1].plot(self.data.beta, color="red")
        ax[1].set_xlabel('day')
        ax[1].set_ylabel(r'$\beta(t)$')

        ax[2].plot(
            np.linspace(0, 365, 364), np.append(np.ones(50), self.lam_list))
        ax[2].set_xlabel("day")
        ax[2].set_ylabel(r"$\lambda$")

        fig.suptitle('EAKF full time series adaptive inflation')

    def compute_data_distribution(self, num_real=300):
        data_distribution = np.zeros(shape=self.data.i.shape)

        for _ in range(num_real):
            a = simulate_data.simulate_data(
                **self.data.true_params, add_noise=True, noise_param=1/50)
            data_distribution = np.vstack((data_distribution, a.i))

        data_distribution = np.delete(data_distribution, (0), axis=0)
        self.data_distribution = data_distribution.T

    def plot_ppc(self):
        fig, ax = plt.subplots()
        ax.plot(self.data_distribution, color="gray", alpha=0.01)
        ax.plot(self.i_ppc, color="blue", alpha=0.01)

        blue_line = plt.Line2D(
            [], [], color='blue', label='Posterior Predictive')
        grey_line = plt.Line2D(
            [], [], color='grey', label='Data Distribution')
        ax.legend(handles=[blue_line, grey_line])

        ax.axvspan(10, 105, color="red", alpha=0.1)
        ax.axvspan(190, 250, color="red", alpha=0.1)

        ax.set_title(
            "Data Distribution vs EAKF Posterior Predictive Distribution")

    def bin_data(self, d, d_pp, num_bins, bins=None):
        data_min = min(np.min(d), np.min(d_pp))
        data_max = max(np.max(d), np.max(d_pp))
        if bins is None:
            bins = np.linspace(data_min, data_max, num_bins + 1)
        digitized = np.digitize(d, bins)
        digitized_pp = np.digitize(d_pp, bins)

        # Adjust bin indices to start from 0
        return digitized - 1, digitized_pp - 1, bins

    def compute_probs(self, digitized_data, num_bins):
        counts = np.bincount(digitized_data, minlength=num_bins)
        probs = counts / len(digitized_data)
        return probs

    def kl_divergence(self, p_sample, q_sample, epsilon=1e-10, num_bins=10):
        p_bins, q_bins, bins = self.bin_data(p_sample, q_sample, num_bins)

        p_probs = self.compute_probs(p_bins, num_bins+1)
        q_probs = self.compute_probs(q_bins, num_bins+1)
        assert len(p_probs) == len(q_probs)

        p_safe = p_probs + epsilon
        q_safe = q_probs + epsilon
        np.sum(p_safe * np.log(p_safe / q_safe))
        return np.sum(p_safe * np.log(p_safe / q_safe))

    def wasserstein2(self, p_sample, q_sample, num_bins=10):
        assert len(p_sample) == len(q_sample)
        p_bins, q_bins, bins = self.bin_data(p_sample, q_sample, num_bins)
        p_probs = self.compute_probs(p_bins, num_bins+1)
        q_probs = self.compute_probs(q_bins, num_bins+1)
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
