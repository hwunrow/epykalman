import pymc as pm
import pytensor.tensor as pt
from pytensor import scan

import logging
import os

import numpy as np

import arviz as az
import matplotlib.pyplot as plt
import seaborn as sns

import pickle
import cloudpickle


class SIR_model():

    def __init__(self, data):
        self.data = data
        self.setup_SIR_model()

    def setup_SIR_model(self):
        self.I0 = pm.floatX(self.data.I[0])
        self.S0 = pm.floatX(self.data.N - self.I0)
        self.i = self.data.i[1:]
        self.n_t = self.data.n_t

    def run_SIR_model(
      self, n_samples, n_tune, likelihood, prior, method, path
      ):
        self.likelihood = likelihood
        self.n_samples = n_samples
        self.n_tune = n_tune
        self.likelihood = likelihood
        self.prior = prior

        if not os.path.exists(path):
            os.makedirs(path)

        logging.basicConfig(
            filename=f'{path}/example.log', encoding='utf-8',
            level=logging.DEBUG)
        logging.info(
            f'true values rt_0: {self.data.rt_0},rt_1: {self.data.rt_1}, \
                midpoint: {self.data.midpoint}, k: {self.data.k}')
        logging.info(f'fixed parameters t_I: {self.data.t_I}, N: \
            {self.data.N}, S_init: {self.data.S0}, I_init: {self.data.I0}')

        logging.info(f'likelihood: {likelihood}')
        logging.info(f'prior parameters: {prior}')
        logging.info(f'MCMC method: {method}')
        logging.info(f'Number of draws: {n_samples} Burn in: {n_tune}')

        with pm.Model() as model:
            rt_0 = pm.Uniform(
                "rt_0", self.prior['rt_0_a'], self.prior['rt_0_b'])
            rt_1 = pm.Uniform(
                "rt_1", self.prior['rt_1_a'], self.prior['rt_1_b'])
            k = pm.Uniform("k", self.prior['k_a'], self.prior['k_b'])
            midpoint = pm.DiscreteUniform(
                "midpoint", self.prior['m_a'], self.prior['m_b'])

            I0 = pm.Poisson("I0", self.prior["I0_lambda"])
            S0 = pm.Deterministic("S0", self.data.N - I0)

            t = np.arange(self.n_t)
            Rt = pm.Deterministic(
                "Rt", rt_0 + (rt_1 - rt_0) / (1. + np.exp(-k*(t - midpoint))))
            beta_t = pm.Deterministic("beta_t", Rt / self.data.t_I)

            def next_day(beta_t, S_t, I_t, _, t_I, N, dt=1):
                dSI = (beta_t * I_t * S_t / N) * dt
                dIR = (I_t / t_I) * dt
                S_t = S_t - dSI
                I_t = I_t + dSI - dIR
                S_t = pt.clip(S_t, 0., N)
                I_t = pt.clip(I_t, 0., N)

                return S_t, I_t, dSI

            new_I_0 = pt.zeros_like(self.I0, dtype="float64")

            results, _ = scan(
                fn=next_day,
                sequences=[beta_t],
                outputs_info=[S0, pm.floatX(I0), new_I_0],
                non_sequences=[self.data.t_I, self.data.N],
                n_steps=self.n_t,
            )

            S, I, i = results
            pm.Deterministic("S", S)
            pm.Deterministic("I", I)
            pm.Deterministic("i", i)
            # print(model.initial_point())
            # print(model.point_logps())
            pm.StudentT(
                "i_est",
                nu=likelihood['nu'],
                mu=i,
                sigma=np.maximum(
                    likelihood['min_sigma'],
                    likelihood['sigma']*self.i),
                observed=self.i
            )

            if method == 'metropolis':
                step = pm.Metropolis()
            elif method == 'NUTS':
                step = pm.NUTS()
            else:
                raise Exception("Method must be either 'metropolis' or 'NUTS'")
            trace = pm.sample(
                n_samples, tune=n_tune, chains=4, cores=4, step=step)

        with model:
            pm.compute_log_likelihood(trace)
            prior_checks = pm.sample_prior_predictive(n_samples)

        self.model = model
        self.trace = trace
        self.prior = prior_checks

    def plot_likelihood(self, ax=None):
        if not ax:
            fig, ax = plt.subplots()
        for x in range(4):
            ax.plot(
                self.model.trace.log_likelihood.sel(chain=x).mean(
                    "draw").to_array().values.ravel(), label=f"chain {x+1}")
            ax.legend()

    def plot_trace(self, vars, ax=None):
        if not ax:
            fig, ax = plt.subplots()
        lines = (
            ('rt_0', {}, self.data.rt_0),
            ('rt_1', {}, self.data.rt_1),
            ('k', {}, self.data.k),
            ('midpoint', {}, self.data.midpoint),
            ('I0', {}, self.data.I0))
        trace_plot = az.plot_trace(
            self.model.trace, var_names=vars, lines=lines, figsize=(18, 20))
        return(trace_plot)

    def plot_posterior(self, vars, ax=None):
        if not ax:
            fig, ax = plt.subplots()
        ref_val = {
            "rt_0": [{"ref_val": self.data.rt_0}],
            "rt_1": [{"ref_val": self.data.rt_1}],
            "k": [{"ref_val": self.data.k}],
            "midpoint": [{"ref_val": self.data.midpoint}],
            "I0": [{"ref_val": self.data.I0}],
            }
        post_plot = az.plot_posterior(
            self.model.trace,
            var_names=vars,
            ref_val=ref_val,
            ref_val_color='red',
            figsize=(20, 5),
            )
        return(post_plot)

    def plot_prior_posterior(self, vars, ax=None):
        if not ax:
            fig, ax = plt.subplots()
        self.model.trace.extend(self.model.prior)
        prior_post_plot = az.plot_dist_comparison(
            self.model.trace, var_names=vars)
        return(prior_post_plot)

    def plot_cov_corr(self, vars, ax=None):
        if not ax:
            fig, ax = plt.subplots()

        def _flat_t(var):
            x = self.model.trace.posterior[var].data
            x = x.reshape((x.shape[0], np.prod(x.shape[1:], dtype=int)))
            return x.mean(axis=0).flatten()

        cov_matrix = np.cov(np.stack(list(map(_flat_t, vars))))
        cov_plot = sns.heatmap(
            cov_matrix, annot=True, xticklabels=vars, yticklabels=vars)

        corr = np.corrcoef(np.stack(list(map(_flat_t, vars))))
        corr_plot = sns.heatmap(
            corr, annot=True, xticklabels=vars, yticklabels=vars)

        return(cov_plot, corr_plot)

    def parallel_coord(self, vars, ax=None):
        if not ax:
            fig, axs = plt.subplots(2)
        # az.plot_parallel(
        # sir_model.trace, var_names=vars, norm_method="normal")
        _posterior = self.model.trace.posterior[vars].mean(
            dim="chain").to_array().data

        fig_parallel, axs = plt.subplots(2, figsize=(10, 10))

        axs[0].plot(_posterior[:], color='black', alpha=0.1)
        axs[0].tick_params(labelsize=10)
        axs[0].set_xticks(range(len(vars)))
        axs[0].set_xticklabels(vars)

        # normalize
        mean = np.mean(_posterior, axis=1)
        sd = np.std(_posterior, axis=1)
        for i in range(0, np.shape(mean)[0]):
            _posterior[i, :] = (_posterior[i, :] - mean[i]) / sd[i]

        axs[1].plot(_posterior[:], color='black', alpha=0.1)
        axs[1].tick_params(labelsize=10)
        axs[1].set_xticks(range(len(vars)))
        axs[1].set_xticklabels(vars)

    def save_model(self, path=None):
        with open(f'{path}/model.pkl', 'wb') as file:
            cloudpickle.dump(self, file, pickle.HIGHEST_PROTOCOL)
