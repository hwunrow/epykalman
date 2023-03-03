import arviz as az
import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np
import logging


def plot_likelihood(self):
    fig, ax = plt.subplots()
    for x in range(4):
        ax.plot(
            self.trace.log_likelihood.sel(chain=x).mean(
                "draw").to_array().values.ravel(), label=f"chain {x+1}")
        ax.legend()
    return fig


def plot_trace(self, vars):
    lines = [(var, {}, getattr(self.data, var)) for var in vars]
    plot_vars = vars+['sigma']
    trace_plot = az.plot_trace(
        self.trace, var_names=plot_vars, lines=lines, figsize=(18, 20))
    return trace_plot


def plot_posterior(self, vars):
    ref_val = {var: [{"ref_val": getattr(self.data, var)}] for var in vars}
    post_plot = az.plot_posterior(
        self.trace,
        var_names=vars+['sigma'],
        ref_val=ref_val,
        ref_val_color='red',
        figsize=(20, 5),
        )
    return post_plot


def plot_prior_posterior(self, vars):
    self.trace.extend(self.prior)
    prior_post_plot = az.plot_dist_comparison(
        self.trace, var_names=vars)
    return prior_post_plot


def plot_cov_corr(self, vars, ax=None):
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))

    def _flat_t(var):
        x = self.trace.posterior[var].data
        x = x.reshape((x.shape[0], np.prod(x.shape[1:], dtype=int)))
        return x.mean(axis=0).flatten()

    cov_matrix = np.cov(np.stack(list(map(_flat_t, vars))))
    sns.heatmap(
        cov_matrix, annot=True, xticklabels=vars,
        yticklabels=vars, ax=axs[0])

    corr = np.corrcoef(np.stack(list(map(_flat_t, vars))))
    sns.heatmap(
        corr, annot=True, xticklabels=vars,
        yticklabels=vars, ax=axs[1])

    return fig


def plot_parallel_coord(self, vars):
    if self.method == self.method == "NUTS":
        par_plot = az.plot_parallel(
            self.trace, var_names=vars, norm_method="normal")
        return par_plot.figure

    _posterior = self.trace.posterior[vars].stack(
        chain_draw=['chain', 'draw']).to_array().data

    fig, axs = plt.subplots(2, figsize=(10, 10))

    axs[0].plot(_posterior[:], color='black', alpha=0.1)
    axs[0].tick_params(labelsize=10)
    axs[0].set_xticks(range(len(vars)))
    axs[0].set_xticklabels(vars)

    # normalize
    mean = self.trace.posterior[vars].mean().to_array().data
    sd = self.trace.posterior[vars].std().to_array().data
    for i in range(0, np.shape(mean)[0]):
        _posterior[i, :] = (_posterior[i, :] - mean[i]) / sd[i]

    axs[1].plot(_posterior[:], color='black', alpha=0.05)
    axs[1].tick_params(labelsize=10)
    axs[1].set_xticks(range(len(vars)))
    axs[1].set_xticklabels(vars)

    return fig


def plot_sir(self, plot_chain=False):
    fig, ax = plt.subplots(2, 2, sharex=True, figsize=(15, 15))

    if plot_chain:  # plot mean of each chain
        for x in range(4):
            ax[0, 0].plot(
                self.trace.posterior["i"].data.mean(axis=1)[x],
                '.', label=f'chain {x+1}')
            ax[0, 1].plot(
                self.trace.posterior["S"].data.mean(axis=1)[x],
                '.', label=f'chain {x+1}')
            ax[1, 0].plot(
                self.trace.posterior["I"].data.mean(axis=1)[x],
                '.', label=f'chain {x+1}')
    else:  # plot 95% HDI
        az.plot_hdi(
            x=range(self.n_t),
            y=self.trace.posterior["i"],
            hdi_prob=0.5,
            color="gray",
            smooth=False,
            fill_kwargs={"label": "HDI 50%", "alpha": 0.3},
            ax=ax[0, 0],
        )
        az.plot_hdi(
            x=range(self.n_t),
            y=self.trace.posterior["i"],
            hdi_prob=0.95,
            color="gray",
            smooth=False,
            fill_kwargs={"label": "HDI 95%", "alpha": 0.5},
            ax=ax[0, 0],
        )

        az.plot_hdi(
            x=range(self.n_t),
            y=self.trace.posterior["S"],
            hdi_prob=0.5,
            color="gray",
            smooth=False,
            fill_kwargs={"label": "HDI 50%", "alpha": 0.3},
            ax=ax[0, 1],
        )
        az.plot_hdi(
            x=range(self.n_t),
            y=self.trace.posterior["S"],
            hdi_prob=0.95,
            color="gray",
            smooth=False,
            fill_kwargs={"label": "HDI 95%", "alpha": 0.5},
            ax=ax[0, 1],
        )

        az.plot_hdi(
            x=range(self.n_t),
            y=self.trace.posterior["I"],
            hdi_prob=0.5,
            color="gray",
            smooth=False,
            fill_kwargs={"label": "HDI 50%", "alpha": 0.3},
            ax=ax[1, 0],
        )
        az.plot_hdi(
            x=range(self.n_t),
            y=self.trace.posterior["I"],
            hdi_prob=0.95,
            color="gray",
            smooth=False,
            fill_kwargs={"label": "HDI 95%", "alpha": 0.5},
            ax=ax[1, 0],
        )

    ax[0, 0].plot(self.data.i_true, '.', label="truth", color='black')
    ax[0, 0].plot(self.data.i, 'x', label="obs", color='blue')
    ax[0, 0].set_xlabel('day')
    ax[0, 0].set_ylabel('Daily case counts')
    ax[0, 0].legend()

    ax[0, 1].plot(self.data.S, '.', label="truth", color='black')
    ax[0, 1].set_xlabel('day')
    ax[0, 1].set_ylabel('Susceptible')
    ax[0, 1].legend()

    ax[1, 0].plot(self.data.I, '.', label="truth", color='black')
    ax[1, 0].set_xlabel('day')
    ax[1, 0].set_ylabel('Infected')
    ax[1, 0].legend()

    fig.delaxes(ax[1, 1])

    return fig


def plot_rt(self):
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']

    fig, ax = plt.subplots(2, figsize=(10, 15))

    for x in range(4):
        ax[0].plot(
            self.trace.posterior["Rt"].data.mean(axis=1)[x],
            label=f"chain {x+1}", color=colors[x])

    ax[0].plot(self.data.rt, label="truth", color='black')
    ax[0].set_xlabel('day')
    ax[0].set_ylabel('Rt')
    ax[0].legend()

    rt_ci = az.hdi(
        self.trace, var_names="Rt", hdi_prob=0.95
        ).to_array().squeeze().data
    mean = self.trace.posterior['Rt'].mean(dim=["chain", "draw"]).data

    ax[1].fill_between(
        range(self.n_t), rt_ci[:, 0], rt_ci[:, 1],
        facecolor='gray', alpha=0.3, label='95% HDI')
    ax[1].plot(mean, '--', label="posterior mean", color='gray')
    ax[1].plot(self.data.rt, label="truth", color='black')
    ax[1].set_xlabel('day')
    ax[1].set_ylabel('Rt')
    ax[1].legend()

    return fig


def plot_ppc(self):
    fig, ax = plt.subplots()
    t = range(self.n_t)

    ax.plot(t, self.data.i_true[1:], '.', label="truth", color='black')
    ax.plot(t, self.i, 'x', label="obs", color='blue')

    az.plot_hdi(
        x=t,
        y=self.posterior_predictive.posterior_predictive["i_est"],
        hdi_prob=0.95,
        color="gray",
        smooth=False,
        fill_kwargs={"label": "HDI 95%", "alpha": 0.3},
        ax=ax,
    )
    az.plot_hdi(
        x=t,
        y=self.posterior_predictive.posterior_predictive["i_est"],
        hdi_prob=0.5,
        color="gray",
        smooth=False,
        fill_kwargs={"label": "HDI 50%", "alpha": 0.5},
        ax=ax,
    )
    ax.legend(loc="upper left")
    ax.set(title="Posterior Predictive HDI SIR Model")

    ci_95 = az.hdi(
        self.posterior_predictive.posterior_predictive,
        var_names=["i_est"], hdi_prob=0.95).i_est.values
    ci_50 = az.hdi(
        self.posterior_predictive.posterior_predictive,
        var_names=["i_est"], hdi_prob=0.50).i_est.values
    prop_95 = np.mean((ci_95[:, 0] <= self.i) & (self.i <= ci_95[:, 1]))
    prop_50 = np.mean((ci_50[:, 0] <= self.i) & (self.i <= ci_50[:, 1]))
    logging.info(f"Percent of observations in 95% CI {prop_95}")
    logging.info(f"Percent of observations in 50% CI {prop_50}")
    print(f"Percent of observations in 95% CI {prop_95}")
    print(f"Percent of observations in 50% CI {prop_50}")

    return (fig)
