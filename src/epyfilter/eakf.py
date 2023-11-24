import numpy as np
from epyfilter import inflation
import matplotlib.pyplot as plt

tol = 1e-16


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
        self.inf_method = inf_method
        x_list = []
        xhat_list = []
        θ_list = []
        lam_list = []
        lam_var_list = []
        alpha_list = []

        lam = lam_S = lam_I = lam_R = lam_i = 1.01
        lam_var = lam_var_S = lam_var_I = lam_var_R = lam_var_i = 0.1

        try:
            beta_0 = self.data.rt_0 / self.data.t_I
            beta_1 = self.data.rt_1 / self.data.t_I
            late_day = -1/self.data.k * np.log(
                (beta_1 - beta_0)/(0.95*beta_1 - beta_0)-1)\
                + self.data.midpoint
        except:
            beta_0 = self.data.rt_m[1] / self.data.t_I
            beta_1 = self.data.rt_m[2] / self.data.t_I
            late_day = -1/self.data.k[1] * np.log(
                (beta_1 - beta_0)/(0.95*beta_1 - beta_0)-1)\
                + self.data.midpoint[1]

        for t in range(self.data.n_t):
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

                if t > 10:  # inflate after 10 days
                    if inf_method == "adaptive":
                        lam, lam_var = inflation.adaptive_inflation(
                            θ.beta, y, z, oev, lam, lam_var)
                        # if t > 100:
                        lam = 0.98 * (lam - 1) + 1
                        if t > late_day + 7:
                            lam = 1.005
                        # lam_max = 1.15
                        # if lam > lam_max:
                        # #     lam = lam_max
                        lam_S, lam_var_S = inflation.adaptive_inflation(
                            x.S, y, z, oev)
                        lam_I, lam_var_I = inflation.adaptive_inflation(
                            x.I, y, z, oev)
                        lam_R, lam_var_R = inflation.adaptive_inflation(
                            x.R, y, z, oev)
                        lam_i, lam_var_i = inflation.adaptive_inflation(
                            x.i, y, z, oev)
                    elif inf_method == "constant":
                        if t < late_day:
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
                    # if turn_off:
                    #     lam = 1.
                    #     lam_S = 1.
                    #     lam_I = 1.
                    #     lam_R = 1.
                    #     lam_i = 1.
                    lam_list.append(lam)
                    lam_var_list.append(lam_var)
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

        self.xhat_list = xhat_list
        self.x_list = x_list
        self.θ_list = θ_list
        self.lam_list = np.array(lam_list)
        self.lam_var_list = np.array(lam_var_list)
        self.alpha_list = np.array(alpha_list)

    def free_sim(self, beta):
        S = np.array([self.data.S0 * np.ones(self.m)])
        Ir = np.array([self.data.I0 * np.ones(self.m)])
        R = np.array([np.zeros(self.m)])
        i = np.array([np.zeros(self.m)])

        for t in range(self.data.beta.shape[0]):
            if t < 10:
                dSI = np.random.poisson(self.data.beta[t] * Ir[t] *
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
    
    def compute_reliability(self, percentiles):
        prop_list = []
        betas = np.asarray([θ.beta for θ in self.θ_list])
        _, _, _, i = self.free_sim(betas)
        for p in percentiles:
            lower = np.quantile(
                i, q=[(1-p/100)/2, 1-(1-p/100)/2], axis=1)[0, :]
            upper = np.quantile(
                i, q=[(1-p/100)/2, 1-(1-p/100)/2], axis=1)[1, :]
            pp = (lower <= self.data.i) & (self.data.i <= upper)
            prop_list.append(np.mean(pp[np.where(self.data.i > 5)]))
        self.prop_list = prop_list
    
    def compute_beta_reliability(self, percentiles):
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
        self.beta_prop_list = prop_list

    def plot_reliability(self, path=None, name='eakf_reliability'):
        betas = np.asarray([θ.beta for θ in self.θ_list])
        S, Ir, R, i = self.free_sim(betas)

        fig, ax = plt.subplots(1)
        ax.plot(i, color='gray', alpha=0.1)
        ax.plot(self.data.i_true, 'x', color='black', label="truth")
        ax.set_title(f"free simulation - all ensembles {name}")

        ci = np.quantile(i, q=[0.025, 0.975], axis=1)
        ci_50 = np.quantile(i, q=[0.25, 0.75], axis=1)

        fig, ax = plt.subplots(1)

        ax.fill_between(np.arange(0, self.data.n_t+1), ci[0], ci[1],
                        facecolor='gray', alpha=0.5, label='95% CI')
        ax.fill_between(np.arange(0, self.data.n_t+1), ci_50[0], ci_50[1],
                        facecolor='gray', alpha=0.75, label='50% CI')
        ax.plot(self.data.i_true, 'x', color='black', label="truth")
        ax.legend(loc='upper left')

        ax.set_title(f"free simulation {name}")

        prop_95 = np.mean((ci[0] <= self.data.i) & (self.data.i <= ci[1]))
        prop_50 = np.mean(
            (ci_50[0] <= self.data.i) & (self.data.i <= ci_50[1]))

        print(f"Percent of observations in 95% CI {round(prop_95*100, 2)}%")
        print(f"Percent of observations in 50% CI {round(prop_50*100, 2)}%")

        percentiles = np.arange(2.5, 100, 2.5)
        if not hasattr(self, "prop_list"):
            self.compute_reliability(percentiles)

        fig, ax = plt.subplots(1)
        ax.plot(percentiles/100, self.prop_list, '-.')
        ax.axline((0, 0), (1, 1), color='r')
        ax.set_xlabel('CI')
        ax.set_ylabel('% of obs within CI')
        ax.set_title(f"observation reliability plot {name}")

        betas = np.asarray([θ.beta for θ in self.θ_list])
        betas_skip = betas[15:, :]
        beta_true = self.data.beta
        beta_true = beta_true[15:]

        if not hasattr(self, "beta_prop_list"):
            self.compute_beta_reliability(percentiles)

        fig, ax = plt.subplots(1)
        ax.plot(percentiles/100, self.beta_prop_list, '-.')
        ax.axline((0, 0), (1, 1), color='r')
        ax.set_xlabel('CI')
        ax.set_ylabel(r'% of $\beta$ within CI')
        ax.set_title(rf"$\beta$ reliability plot {name}")

        # for p in percentiles:
        #     S, Ir, R, i = self.free_sim(np.quantile(betas_skip, q=p/100))
        return fig

    def plot_posterior(self, path=None, name='eakf_posterior'):
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

        fig, ax = plt.subplots(4)
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

        ax[2].plot(np.linspace(0, self.data.n_t, self.data.n_t-1),
                   np.append(np.ones(self.data.n_t - len(self.lam_list) - 1),
                             self.lam_list))
        ax[2].set_xlabel("day")
        ax[2].set_ylabel(r"$\lambda$")

        ax[3].plot(np.concatenate(
            (np.zeros(self.data.n_t - len(self.lam_var_list)-1),
             self.lam_var_list)))
        ax[3].set_xlabel("day")
        ax[3].set_ylabel(r"$\sigma^2_{\lambda}$")

        fig.suptitle(f'EAKF full time series {name}')

        return fig

    def plot_ppc(self, path=None, name='eakf_ppc'):
        fig, ax = plt.subplots(3)
        ax[0].plot(self.data.data_distribution, color="gray", alpha=0.01)
        ax[0].plot(self.i_ppc, color="blue", alpha=0.01)

        blue_line = plt.Line2D(
            [], [], color='blue', label='Posterior Predictive')
        grey_line = plt.Line2D(
            [], [], color='grey', label='Data Distribution')
        ax[0].legend(handles=[blue_line, grey_line])
        ax[0].set_title(
            f"Data Distribution vs EAKF Posterior Predictive Distribution {name}")
        ax[1].plot(self.data.data_distribution, color="gray", alpha=0.01)
        ax[2].plot(self.i_ppc, color="blue", alpha=0.01)

        return fig
