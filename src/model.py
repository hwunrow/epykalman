import pymc as pm
import pytensor.tensor as pt
from pytensor import scan
import pytensor

import numpy as np


class SIR_model():

    def __init__(self, data):
        self.data = data
        self.setup_SIR_model()

    def setup_SIR_model(self):
        self.I0 = pm.floatX(self.data.I[0])
        self.S0 = pm.floatX(self.data.N - self.I0)
        self.i = self.data.i[1:]
        self.n_t = len(self.i)

    def run_SIR_model(self, n_samples, n_tune, likelihood, prior, method):
        self.likelihood = likelihood
        self.n_samples = n_samples
        self.n_tune = n_tune
        self.likelihood = likelihood
        self.prior = prior

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
                    likelihood['sigma']*np.sqrt(self.i)),
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
