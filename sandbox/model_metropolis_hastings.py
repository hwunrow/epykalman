import numpy as np
import pandas as pd

import logging
import os
from pprint import pformat
from tqdm import tqdm

import arviz as az
import matplotlib.pyplot as plt

from scipy.stats import norm

from random import choices

import pickle
import inspect


class SIR_model():

    def __init__(self, data):
        self.data = data
        self.setup_SIR_model()
        self.index_map = {
            'rt_0': 0,
            'rt_1': 1,
            'k': 2,
            'midpoint': 3,
            'I0': 4,
        }

    def setup_SIR_model(self):
        self.I0 = self.data.I0
        self.t_I = self.data.t_I
        self.N = self.data.N
        self.S0 = self.data.N - self.I0
        self.i = self.data.i[1:]
        self.n_t = self.data.n_t

    def check_bounds(self, new_draw):
        """Returns True if draw is within bounds, False otherwise
        """
        bounds_bool = True
        for k, v in self.prior.items():
            if 'lower' in v.keys():
                bounds_bool = bounds_bool and new_draw[k] > v['lower'] and \
                    new_draw[k] < v['upper']

            if 'mu' in v.keys():
                bounds_bool = bounds_bool and new_draw[k] > 0

        return bounds_bool

    def check_one_bound(self, new_draw, var):
        index = self.index_map[var]
        if index == 4:
            return new_draw > 0
        else:
            return new_draw > self.prior[var]['args']['lower'] and \
                new_draw < self.prior[var]['args']['upper']

    def forward(self, x, a, b):
        u = (x - a)/(b-a)
        return np.log(u / (1 - u))

    def backward(self, y, a, b):
        return a + (b-a)/(1 + np.exp(-y))

    def draw_sample_elemwise(self, draw, var, scales):
        index = self.index_map[var]
        scale = scales[index]

        bounds_good = False

        while not bounds_good:
            val = norm(loc=draw[var], scale=scale).rvs(1)[0]
            bounds_good = self.check_one_bound(val, var)

        if index == 3 or index == 4:
            val = int(val)

        return val

    def log_random_coin(self, logp):
        """
        Returns true with prob exp(logp)
        """
        unif = np.random.uniform(0, 1)
        if np.log(unif) >= logp:
            return False
        else:
            return True

    def integrate(self, rt_0, rt_1, k, midpoint, I0, t_I, N, S0, n_t):
        t = np.arange(365)
        rt = rt_0 + (rt_1 - rt_0) / (1. + np.exp(-k*(t - midpoint)))
        beta = rt / t_I
        S = np.array([S0])
        Ir = np.array([I0])
        R = np.array([0])
        i = np.array([0])
        for t in range(n_t):
            dSI = np.random.poisson(beta[t]*Ir[t]*S[t]/N)
            dIR = np.random.poisson(Ir[t]/t_I)

            S_new = np.clip(S[t]-dSI, 0, N)
            I_new = np.clip(Ir[t]+dSI-dIR, 0, N)
            R_new = np.clip(R[t]+dIR, 0, N)

            S = np.append(S, S_new)
            Ir = np.append(Ir, I_new)
            R = np.append(R, R_new)
            i = np.append(i, dSI)

        if not self.data.run_deterministic:
            i = i.astype('float64')
            obs_error_var = np.maximum(1., i[1:]**2 * self.data.noise_param)
            obs_error_sample = np.random.normal(0, 1, size=n_t)
            i[1:] += obs_error_sample * np.sqrt(obs_error_var)
            i = np.minimum(np.maximum(i, 0), N)

        return S, Ir, R, i

    def logprior_prob(self, prior, rt_0, rt_1, k, midpoint, I0):
        if prior['rt_0']['dist'] == "constant":
            logp_rt_0 = 0
        else:
            # logp_rt_0 = prior['rt_0']['dist'](
            #     loc=prior['rt_0']['lower'],
            #     scale=(prior['rt_0']['upper'] - prior['rt_0']['lower'])
            # ).logpdf(rt_0)
            logp_rt_0 = np.log(
                1./(prior['rt_0']['args']['upper'] - prior['rt_0']['args']['lower']))

        if prior['rt_1']['dist'] == "constant":
            logp_rt_1 = 0
        else:
            # logp_rt_1 = prior['rt_1']['dist'](
            #     loc=prior['rt_1']['lower'],
            #     scale=(prior['rt_1']['upper'] - prior['rt_1']['lower'])
            # ).logpdf(rt_1)
            logp_rt_1 = np.log(
                1./(prior['rt_1']['args']['upper'] - prior['rt_1']['args']['lower']))

        if prior['k']['dist'] == "constant":
            logp_k = 0
        else:
            # logp_k = prior['k']['dist'](
            #     loc=prior['k']['lower'],
            #     scale=(prior['k']['upper'] - prior['k']['lower'])
            # ).logpdf(k)
            logp_k = np.log(
                1./(prior['k']['args']['upper'] - prior['k']['args']['lower']))

        if prior['midpoint']['dist'] == "constant":
            logp_midpoint = 0
        else:
            # logp_midpoint = prior['midpoint']['dist'](
            #     low=prior['midpoint']['lower'],
            #     high=prior['midpoint']['upper']
            # ).logpmf(midpoint)
            logp_midpoint = np.log(
                1./(prior['midpoint']['args']['upper']-prior['midpoint']['args']['lower'] + 1)
            )

        if prior['I0']['dist'] == "constant":
            logp_I0 = 0
        else:
            logp_I0 = prior['I0']['dist'](prior['I0']['args']['mu']).logpmf(I0)

        logp = np.sum([logp_rt_0, logp_rt_1, logp_k, logp_midpoint, logp_I0])

        return logp

    def loglike_prob(self, i):
        data = self.data.i[1:]
        prob = norm(loc=i[1:], scale=np.abs(1+0.2*i[1:])).logpdf(data)
        prob = np.sum(prob)
        return prob

    def initialize_params(self, prior):
        if prior['rt_0']['dist'] == "constant":
            rt_0 = self.data.rt_0
        else:
            rt_0 = prior['rt_0']['dist'](
                loc=prior['rt_0']['args']['lower'],
                scale=(prior['rt_0']['args']['upper'] - prior['rt_0']['args']['lower'])
            ).rvs(1)[0]

        if prior['rt_1']['dist'] == "constant":
            rt_1 = self.data.rt_1
        else:
            rt_1 = prior['rt_1']['dist'](
                loc=prior['rt_1']['args']['lower'],
                scale=(prior['rt_1']['args']['upper'] - prior['rt_1']['args']['lower'])
            ).rvs(1)[0]

        if prior['k']['dist'] == "constant":
            k = self.data.k
        else:
            k = prior['k']['dist'](
                loc=prior['k']['args']['lower'],
                scale=(prior['k']['args']['upper'] - prior['k']['args']['lower'])
            ).rvs(1)[0]

        if prior['midpoint']['dist'] == "constant":
            m = self.data.midpoint
        else:
            m = prior['midpoint']['dist'](
                low=prior['midpoint']['args']['lower'],
                high=prior['midpoint']['args']['upper']
            ).rvs(1)[0]

        if prior['I0']['dist'] == "constant":
            I0 = self.data.I0
        else:
            I0 = prior['I0']['dist'](prior['I0']['args']['mu']).rvs(1)[0]

        return {'rt_0': rt_0, 'rt_1': rt_1, 'k': k, 'midpoint': m, 'I0': I0}

    def tune(self, scale, acc_rate):
        """
        Tunes the scaling parameter for the proposal distribution
        according to the acceptance rate over the last tune_interval:

        Rate    Variance adaptation
        ----    -------------------
        <0.001        x 0.1
        <0.05         x 0.5
        <0.2          x 0.9
        >0.5          x 1.1
        >0.75         x 2
        >0.95         x 10
        """
        return scale * np.where(
            acc_rate < 0.001,
            # reduce by 90 percent
            0.1,
            np.where(
                acc_rate < 0.05,
                # reduce by 50 percent
                0.5,
                np.where(
                    acc_rate < 0.2,
                    # reduce by ten percent
                    0.9,
                    np.where(
                        acc_rate > 0.95,
                        # increase by factor of ten
                        10.0,
                        np.where(
                            acc_rate > 0.75,
                            # increase by double
                            2.0,
                            np.where(
                                acc_rate > 0.5,
                                # increase by ten percent
                                1.1,
                                # Do not change
                                1.0,
                            ),
                        ),
                    ),
                ),
            ),
        )

    def run_SIR_model(
      self, n_samples, n_tune, tune_interval, prior, path,
      scales=np.array([0.1, 0.1, 0.1, 5, 10])
      ):
        self.n_samples = n_samples
        self.n_tune = n_tune
        self.prior = prior

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
        self.true_params = true_params
        fixed_vars = []
        for k, v in prior.items():
            if v['dist'] == "constant":
                fixed_vars.append(k)
        fixed_params = {v: true_params[v] for v in fixed_vars} | {
            't_I': self.data.t_I,
            'N': self.data.N,
        }
        logging.info(f'fixed parameters: {pformat(fixed_params)}')
        logging.info(f'prior parameters: {pformat(prior)}')
        logging.info(f'Number of draws: {n_samples} Burn in: {n_tune}')

        i_list = []
        chain = []
        like_list = []
        prior_list = []
        params = ['rt_0', 'rt_1', 'k', 'midpoint', 'I0']
        params = [p for p in params if p not in fixed_vars]
        accept_list = {p: [] for p in params}
        self.params = params
        accept_prob_list = []
        scale_list = []
        acc_rate_list = []

        params_t = self.initialize_params(prior)
        chain.append(params_t)

        accept = 1
        scales = [scales[self.index_map[p]] for p in params]
        scale_list.append(scales)

        for j in tqdm(range(n_tune+n_samples)):
            if j > 0 and len(params)*j % tune_interval == 0:
                if j < int(n_tune/len(params)) + 1:
                    acc_rate = np.array(list(accept_list.values()))
                    acc_rate = acc_rate[:, j-int(tune_interval/len(params)):]
                    acc_rate = np.mean(acc_rate, axis=1)
                    scales = self.tune(scales, acc_rate)
                    scale_list.append(scales)
                    acc_rate_list.append(acc_rate)

            np.random.shuffle(params)
            for param in params:
                if accept == 1:
                    _, _, _, i = self.integrate(
                        t_I=self.t_I, N=self.N, S0=self.N - params_t['I0'],
                        n_t=self.n_t, **params_t.copy())
                i_list.append(i)

                logprior = self.logprior_prob(prior, **params_t)
                loglike = self.loglike_prob(i)
                curr_prob = logprior + loglike

                new_draw = params_t.copy()
                val = self.draw_sample_elemwise(new_draw, param, scales)
                new_draw[param] = val

                S0_new = self.data.N - new_draw['I0']
                _, _, _, i_new = self.integrate(
                    t_I=self.t_I, N=self.N, S0=S0_new, n_t=self.n_t,
                    **new_draw.copy())

                logprior_new = self.logprior_prob(prior, **new_draw.copy())
                loglike_new = self.loglike_prob(i_new)
                move_prob = logprior_new + loglike_new

                logacceptance = np.minimum(move_prob - curr_prob, 0.)
                accept_prob_list.append(logacceptance)

                if self.log_random_coin(logacceptance):
                    params_t[param] = val
                    accept = 1
                    loglike = loglike_new
                    logprior = logprior_new
                else:
                    accept = 0

                accept_list[param].append(accept)
                like_list.append(loglike)
                prior_list.append(logprior)
            chain.append(params_t.copy())

        self.burn_in = pd.DataFrame(chain[:n_tune])
        self.trace = pd.DataFrame(chain[n_tune:])
        self.trace_list = chain[n_tune:]
        self.accept_prob = np.mean(
            np.array(list(accept_list.values())), axis=1)
        self.accept_prob_list = accept_prob_list
        self.i_chain = np.array(i_list)
        self.like_list = like_list
        self.prior_list = prior_list
        self.scale_list = scale_list
        self.acc_rate_list = acc_rate_list

        print(self.summary(self.trace, params))
        print(self.accept_prob)

    def summary(self, df, vars):
        mean_df = pd.DataFrame(df[vars].mean())
        ci_df = []
        for var in vars:
            ci = az.hdi(np.array(df[var]), alpha=0.94)
            ci_df.append(ci)
        ci_df = pd.DataFrame(ci_df)

        summary_df = pd.DataFrame()
        summary_df['mean'] = mean_df
        summary_df[['HDI 3%', 'HDI 94%']] = ci_df.values
        summary_df['truth'] = [self.true_params[var] for var in vars]

        return summary_df

    def plot_trace(self):
        def compute_bins(data):
            d = np.diff(np.unique(data)).min()
            left_of_first_bin = data.min() - float(d)/2
            right_of_last_bin = data.max() + float(d)/2

            bins = np.arange(left_of_first_bin, right_of_last_bin + d, d)

            return bins

        def compute_ci(data):
            return az.hdi(np.array(data), alpha=0.94)

        def plot_post(var, ax, bins=False):
            ci = compute_ci(self.trace[var])
            if bins:
                bins = compute_bins(self.trace[var])
                self.trace[var].plot.hist(
                    ax=ax, bins=bins, density=True, color='lightblue')
            else:
                self.trace[var].plot.hist(
                    ax=ax, density=True, color='lightblue')
            ax.axvline(
                x=self.true_params[var], color='black', alpha=1, linestyle='-',
                label='truth')
            ax.hlines(
                y=0.001, xmin=ci[0], xmax=ci[1], color='r', linewidth=4,
                label='94% HDI')
            ax.set_title(f'{var} Posterior')
            ax.legend()

        def plot_trace(var, ax):
            ax.plot(np.arange(0, self.n_tune),
                    self.burn_in[var], color='grey', label='burn in')
            ax.plot(np.arange(self.n_tune-1, self.n_tune+self.n_samples),
                    self.trace[var], color='lightblue', label='trace')
            ax.axhline(
                y=self.true_params[var], color='black', alpha=1,
                linestyle='-', label='truth')
            ax.set_title(f'{var} Trace Plot')
            ax.legend()

        fig, ax = plt.subplots(
            nrows=len(self.params), ncols=2, figsize=(15, 25))

        for i, p in enumerate(self.params):
            plot_post(p, ax=ax[i][0])
            plot_trace(p, ax=ax[i][1])

    def calc_ppc(self):
        ppc = choices(self.trace_list, k=1000)

        i_ppc_list = []
        for theta in ppc:
            S0 = self.N - theta['I0']
            _, _, _, i = self.integrate(
                t_I=self.t_I, N=self.N, S0=S0, n_t=self.n_t, **theta)
            i_ppc_list.append(i)

        i_ppc = np.array(i_ppc_list)
        i_ppc = i_ppc[:, 1:]

        self.i_ppc = i_ppc

        return i_ppc

    def plot_ppc(self):
        fig, ax = plt.subplots()
        t = range(self.n_t)

        ax.plot(t, self.data.i_true[1:], '.', label="truth", color='black')
        ax.plot(t, self.data.i[1:], 'x', label="obs", color='blue')

        az.plot_hdi(
            x=t,
            y=self.i_ppc,
            hdi_prob=0.95,
            color="gray",
            smooth=False,
            fill_kwargs={"label": "HDI 95%", "alpha": 0.3},
            ax=ax,
        )
        az.plot_hdi(
            x=t,
            y=self.i_ppc,
            hdi_prob=0.5,
            color="gray",
            smooth=False,
            fill_kwargs={"label": "HDI 50%", "alpha": 0.5},
            ax=ax,
        )
        ax.legend(loc="upper left")
        ax.set(title="Posterior Predictive HDI SIR Model")

    def calc_ppc_data(self):
        ci_95 = az.hdi(self.i_ppc, hdi_prob=0.95)
        ci_50 = az.hdi(self.i_ppc, hdi_prob=0.5)
        prop_95 = np.mean(
            (ci_95[:, 0] <= self.data.i[1:]) &
            (self.data.i[1:] <= ci_95[:, 1]))
        prop_50 = np.mean(
            (ci_50[:, 0] <= self.data.i[1:]) &
            (self.data.i[1:] <= ci_50[:, 1]))
        print(f"Percent of observations in 95% CI {round(prop_95*100, 2)}%")
        print(f"Percent of observations in 50% CI {round(prop_50*100, 2)}%")

    def save_model(self, path=None):
        # log source code
        lines = inspect.getsource(self.run_SIR_model)
        logging.info(lines)
        with open(f'{path}/model.pkl', 'wb') as file:
            pickle.dump(self, file, pickle.HIGHEST_PROTOCOL)
