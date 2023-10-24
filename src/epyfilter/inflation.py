import numpy as np

tol = 1e-16
ratio_tol = 0.99


def adaptive_inflation(x, y, z, oev, lambar_prior=1.01, siglam2=0.0001):
    sig2p = np.var(y)
    # if degenerate forecast distribution, do nothing
    if sig2p < tol:
        return lambar_prior, siglam2
    new_lam = linear_bayes(x, y, z, oev, lambar_prior, siglam2)

    siglam2_lower_bound = 0.0001
    if np.abs(siglam2 - siglam2_lower_bound) < tol:
        return new_lam, siglam2
    else:
        new_max = compute_new_density(
            x, y, z, oev, lambar_prior, siglam2, new_lam)
        new_1_sd = compute_new_density(
            x, y, z, oev, lambar_prior, siglam2, new_lam + np.sqrt(siglam2))

        if np.abs(new_max) < tol or np.abs(new_1_sd) < tol:
            return new_lam, siglam2
        else:
            ratio = new_1_sd / new_max

            if ratio > ratio_tol:
                return new_lam, siglam2
            else:
                new_siglam2 = -1. * siglam2 / (2. * np.log(ratio))

                if new_siglam2 > siglam2:
                    return new_lam, siglam2
                else:
                    return new_lam, new_siglam2


def compute_new_density(x, y, z, oev, lambar_prior, siglam2, lam):
    sig2p = np.var(y)
    ybarp = np.mean(y)
    D = np.abs(ybarp - z) * z / 50
    # D = np.abs(ybarp - z)
    r = np.corrcoef(x, y)[0, 1]  # i
    lam0 = (1.0 + r * (np.sqrt(lambar_prior) - 1.0)) ** 2  # iii
    theta = np.sqrt(lam0 * sig2p + oev)  # iv

    exp_prior = (lam-lambar_prior)**2 / (-2. * siglam2)
    exp_like = D**2 / (-2. * theta**2)

    p = np.exp(exp_like + exp_prior) / (2. * np.pi * np.sqrt(siglam2) * theta)

    return p


def linear_bayes(x, y, z, oev, lambar_prior, siglam2):
    # step 3b
    sig2p = np.var(y)
    # degenerate observation posterior
    if sig2p < tol:
        return lambar_prior
    ybarp = np.mean(y)
    D = np.abs(ybarp - z)

    # step 3c
    r = np.corrcoef(x, y)[0, 1]  # i
    lam0 = (1.0 + r * (np.sqrt(lambar_prior) - 1.0)) ** 2  # iii
    theta = np.sqrt(lam0 * sig2p + oev)  # iv

    # v/ Appendix A
    lbar = np.exp(-(D**2) / (2. * theta**2)) / (np.sqrt(2. * np.pi) * theta)
    # if lbar goes to 0, can't do anything so just keep current value
    if lbar < tol:
        return lambar_prior

    dtheta_dlam = (
        sig2p * r * (1.0 - r + r * np.sqrt(lambar_prior))
        / (2.0 * theta * np.sqrt(lambar_prior))
    )
    lprime = lbar * (D**2 / theta**2 - 1.0) / theta * dtheta_dlam

    # if lprime goes to 0, can't do anything so just keep current value
    if lprime < tol:
        return lambar_prior

    b = lbar / lprime - 2. * lambar_prior
    c = lambar_prior**2 - siglam2 - lbar * lambar_prior / lprime

    lam1 = np.abs((-b + np.sqrt(b**2 - 4. * c)) / 2.)
    lam2 = np.abs((-b - np.sqrt(b**2 - 4. * c)) / 2.)

    # pick closest root
    if np.abs(lam1 - lambar_prior) < np.abs(lam2 - lambar_prior):
        return lam1
    else:
        return lam2


def inflate_ensemble(ens, inflation_value, params=False):
    if params:
        _, m = np.asarray(ens).shape
        ens_mean = np.mean(ens, 1, keepdims=True)
        ens_inflated = ens_mean * np.ones((1, m)) + inflation_value * (
            ens - ens_mean * np.ones((1, m))
        )
    else:
        m = len(ens)
        ens_mean = np.mean(ens)
        ens_inflated = ens_mean * np.ones(m) + inflation_value * (
            ens - ens_mean * np.ones(m)
        )

    return ens_inflated
