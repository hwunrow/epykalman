import pymc as pm
import pytensor.tensor as pt
from pytensor import scan

import logging
import os
from pprint import pformat

import numpy as np
import pandas as pd

import arviz as az
from matplotlib.backends.backend_pdf import PdfPages

import pickle
import cloudpickle
import inspect


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
        self.method = method

        if not os.path.exists(path):
            os.makedirs(path)

        logging.basicConfig(
            filename=f'{path}/example.log', encoding='utf-8',
            level=logging.INFO)
        true_params = {
            'rt_0': self.data.rt_0,
            'rt_1': self.data.rt_1,
            'k': self.data.k,
            'midpoint': self.data.midpoint,
            'I0': self.data.I0,
            'N': self.data.N
        }
        fixed_vars = []
        for k, v in prior.items():
            if v['dist'] == "constant":
                fixed_vars.append(k)
        fixed_params = {v: true_params[v] for v in fixed_vars} | {
            't_I': self.data.t_I,
            'N': self.data.N,
        }
        logging.info(f'fixed parameters {pformat(fixed_params)}')
        logging.info(f'likelihood: {pformat(likelihood)}')
        logging.info(f'prior parameters: {pformat(prior)}')
        logging.info(f'MCMC method: {method}')
        logging.info(f'Number of draws: {n_samples} Burn in: {n_tune}')

        with pm.Model() as model:
            if prior['rt_0']['dist'] == "constant":
                rt_0 = prior['rt_0']['args']['value']
            else:
                rt_0 = prior['rt_0']['dist']("rt_0", **prior['rt_0']['args'])

            if prior['rt_1']['dist'] == "constant":
                rt_1 = prior['rt_1']['args']['value']
            else:
                rt_1 = prior['rt_1']['dist']("rt_1", **prior['rt_1']['args'])

            if prior['k']['dist'] == "constant":
                k = prior['k']['args']['value']
            else:
                k = prior['k']['dist']("k", **prior['k']['args'])

            if prior['midpoint']['dist'] == "constant":
                midpoint = prior['midpoint']['args']['value']
            else:
                midpoint = prior['midpoint']['dist'](
                    "midpoint", **prior['midpoint']['args'])

            I0 = prior['I0']['dist']("I0", **prior['I0']['args'])
            S0 = pm.Deterministic("S0", self.data.N - I0)

            t = np.arange(self.n_t)
            Rt = pm.Deterministic(
                "Rt", rt_0 + (rt_1 - rt_0) / (1. + np.exp(-k*(t - midpoint))))
            beta_t = pm.Deterministic("beta_t", Rt / self.data.t_I)

            def next_day(b, S_t, I_t, _, t_I, N, dt=1):
                dSI = (b * I_t * S_t / N) * dt
                dIR = (I_t / t_I) * dt

                S_t = S_t - dSI
                I_t = I_t + dSI - dIR

                S_t = pt.clip(S_t, 0., N)
                I_t = pt.clip(I_t, 0., N)

                if not self.data.run_deterministic:
                    obs_error_var = pm.math.maximum(
                        1., dSI**2 * self.data.noise_param)
                    obs_error_sample = np.random.normal(0, 1)
                    dSI += obs_error_sample * np.sqrt(obs_error_var)
                    dSI = pt.clip(dSI, 0., N)

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
            if likelihood['dist'] == 'students-t':
                sigma = pm.HalfCauchy("sigma", 10)
                like = pm.StudentT(
                    "i_est",
                    nu=likelihood['nu'],
                    mu=i,
                    sigma=pt.abs(1+sigma*i),
                    observed=self.i
                )
            elif likelihood['dist'] == 'normal':
                sigma = pm.HalfCauchy("sigma", 10)
                like = pm.Normal(
                    "i_est",
                    mu=i,
                    sigma=pt.abs(1+sigma*i),
                    observed=self.i
                )
            elif likelihood['dist'] == 'negbin':
                alpha_inv = pm.Normal(name="alpha_inv", mu=0, sigma=0.5)
                alpha = pm.Deterministic(
                    name="alpha", var=1 / pm.math.sqr(alpha_inv))
                like = pm.NegativeBinomial(
                    "i_est",
                    alpha=pt.abs(alpha+0.01),
                    mu=i,
                    observed=self.i
                )
            else:
                raise Exception("Likelihood dist must be studens-t or normal")

            pm.Deterministic("likelihood", like)

            if method == 'metropolis':
                step = pm.Metropolis()
            elif method == 'NUTS':
                step = pm.NUTS(adapt_step_size=True)
                # pm.init_nuts(init='advi', n_init=500_000)
            else:
                raise Exception("Method must be either 'metropolis' or 'NUTS'")
            trace = pm.sample(
                n_samples, tune=n_tune, chains=4, cores=4, step=step)

        with model:
            pm.compute_log_likelihood(trace)
            prior_checks = pm.sample_prior_predictive(n_samples)
            posterior_predictive = pm.sample_posterior_predictive(trace)

        self.model = model
        self.trace = trace
        self.prior = prior_checks
        self.posterior_predictive = posterior_predictive

        gv = pm.model_to_graphviz(model)
        gv.render(filename=f'{path}/model', format='pdf')

    def sample_stats(self, vars):
        if self.method == "metropolis":
            acc_rate = self.trace.sample_stats['accepted'].sum(axis=1).data / \
                self.n_samples
            acc_rate = pd.DataFrame(acc_rate)
        elif self.method == "NUTS":
            acc_rate = self.trace.sample_stats['accepted'].sum(dim="draw").\
                data / self.n_samples

        summary_df = az.summary(self.trace, var_names=vars).round(2)
        true_params = {
            'rt_0': self.data.rt_0,
            'rt_1': self.data.rt_1,
            'k': self.data.k,
            'midpoint': self.data.midpoint,
            'I0': self.data.I0,
            'sigma': None,
        }
        summary_df['truth'] = [true_params[v] for v in vars]

        logging.info(f'trace summary:\n {summary_df}')
        logging.info(f'acceptance_rate:\n {acc_rate}')

        return (summary_df, acc_rate)

    # import plotting functions
    from pymc_plot import (plot_trace, plot_posterior, plot_prior_posterior,
                           plot_cov_corr, plot_parallel_coord, plot_sir,
                           plot_rt, plot_ppc)

    def plot_all(self, vars, path=None):
        trace_plot = self.plot_trace(vars)
        post_plot = self.plot_posterior(vars)
        prior_post_plot = self.plot_prior_posterior(vars)
        cov_corr_plot = self.plot_cov_corr(vars)
        fig_parallel = self.plot_parallel_coord(vars)
        fig_sir = self.plot_sir(plot_chain=True)
        fig_sir_ci = self.plot_sir()
        fig_Rt = self.plot_rt()
        fig_ppc = self.plot_ppc()

        with PdfPages(f'{path}/mcmc_plots.pdf') as pdf:
            pdf.savefig(trace_plot.ravel()[0].figure)
            pdf.savefig(post_plot.ravel()[0].figure)
            pdf.savefig(prior_post_plot.ravel()[0].figure)
            pdf.savefig(cov_corr_plot)
            pdf.savefig(fig_parallel)
            pdf.savefig(fig_sir)
            pdf.savefig(fig_sir_ci)
            pdf.savefig(fig_Rt)
            pdf.savefig(fig_ppc)

    def save_model(self, path=None):
        # log source code
        lines = inspect.getsource(self.run_SIR_model)
        logging.info(lines)
        with open(f'{path}/model.pkl', 'wb') as file:
            cloudpickle.dump(self, file, pickle.HIGHEST_PROTOCOL)
