import pyro
import torch
import numpy as np
from torch.distributions import constraints
from pyro.infer import SVI, JitTraceEnum_ELBO, TraceEnum_ELBO, config_enumerate
import pyro.distributions as dist
import pyro.poutine as poutine
from pyro.params import param_store
from pyro.infer.autoguide.guides import AutoDelta
from pyro.optim import Adam, ClippedAdam
import pyro.contrib.gp as gp
from scipy.special import logsumexp
from scipy.stats import norm, invgamma
import warnings
warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)

@config_enumerate
def gmm_simple(data, n_params=3, n_mod_status=2):
    n_reads, n_bases = data.shape
    n_bases = int(n_bases / n_params)

    is_observed = (~torch.isnan(data))
    valid_data = torch.nan_to_num(data, nan=0.0)

    # modification params:
    with pyro.plate("mod", n_mod_status, dim=-2):
        # weight for modified/unmodified
        ω = pyro.sample(
            "ω",
            dist.Gamma(1/n_mod_status, 1)
        )
        with pyro.plate("base", n_bases, dim=-1):
            # mean params
            μ_ν = pyro.sample(
                "μ_ν",
                dist.Normal(0, 5)
            )
            μ_σ = pyro.sample(
                "μ_σ",
                dist.InverseGamma(1, 1)
            )
            # variance params
            α = pyro.sample(
                "α",
                dist.Gamma(1, 1)
            )         
            β = pyro.sample(
                "β",
                dist.Gamma(1, 1)
            )
            # dwell params
            t_ν = pyro.sample(
                "t_ν",
                dist.Normal(0, 5)
            )
            t_σ = pyro.sample(
                "t_σ",
                dist.InverseGamma(1, 1)
            )
    
    # base params
    with pyro.plate("read_j", n_reads, dim=-2):
        ω = ω.reshape(1, -1)[0]
        z = pyro.sample(
            "z",
            dist.Categorical(ω)
        )
        with pyro.plate("base_i", n_bases, dim=-1) as i:
            pyro.sample(
                "μ",
                dist.Normal(
                    μ_ν[z, i], 
                    μ_σ[z, i]
                ).mask(is_observed[:,0:n_bases]),
                obs = valid_data[:,0:n_bases]
            )
            pyro.sample(
                "τ",
                dist.InverseGamma(
                    concentration=α[z, i], 
                    rate=β[z, i]
                ).mask(is_observed[:,n_bases:n_bases*2]),
                obs = torch.clamp(valid_data[:, n_bases:n_bases*2], min=1e-8)
            )
            pyro.sample(
                "t",
                dist.Normal(
                    t_ν[z, i], 
                    t_σ[z, i]
                ).mask(is_observed[:,n_bases*2:n_bases*3]),
                obs = valid_data[:,n_bases*2:n_bases*3]
            )
        
#pyro.render_model(gmm_simple, model_args=(mat[:50,:],), render_distributions=True, render_params=True)              


def run_svi(data, model=gmm_simple, n_mod_status=2, n_steps=2500):
    pyro.clear_param_store()
    adam = ClippedAdam({'lr': 0.01, 
                        'betas': [0.85, 0.99]})
    guide = AutoDelta(poutine.block(gmm_simple, expose=['μ_ν', 'μ_σ', 'α', 'β','t_σ', 't_ν', 'ω']))
    loss = JitTraceEnum_ELBO()
    svi = SVI(model, guide, adam, loss)
    for _ in range(n_steps):
        loss = svi.step(data=data, n_mod_status=n_mod_status)

    w = pyro.param("AutoDelta.ω").detach().numpy().reshape(1,-1)[0]
    w = w / np.sum(w)
    mu_mu = pyro.param("AutoDelta.μ_ν").detach().numpy()
    mu_sigma = pyro.param("AutoDelta.μ_σ").detach().numpy()
    alpha = pyro.param("AutoDelta.α").detach().numpy()
    beta = pyro.param("AutoDelta.β").detach().numpy()
    t_mu = pyro.param("AutoDelta.t_ν").detach().numpy()
    t_sigma = pyro.param("AutoDelta.t_σ").detach().numpy()
    return {'w': w, 'mu_mu':mu_mu, 'mu_sigma':mu_sigma, 'alpha':alpha, 
            'beta':beta, 't_mu':t_mu, 't_sigma':t_sigma}


def assign_ref_class(levels, ref_start, ref_end, mu_mu):
    '''
    set the component with less difference to reference levels as 0, the other as 1
    '''
    ref_levels = get_nbases_levels(levels, ref_start, ref_end)
    assert len(ref_levels) == mu_mu.shape[1]
    abs_diffs = [sum(np.abs(ref_levels - mu_c)) for mu_c in mu_mu]
    indices = np.zeros_like(abs_diffs, dtype=int)
    indices[np.argmax(abs_diffs)] = 1
    return indices


def compute_posteriors(data, n_bases, mu_mu, mu_sigma, alpha, beta, t_mu, t_sigma):
    means = data[:, 0:n_bases]
    variances = data[:, n_bases:n_bases*2]
    dwells = data[:, n_bases*2:n_bases*3]
    n_comps, _ = mu_mu.shape
    posteriors = []
    for z in range(n_comps):
        mu_base_sum = np.nansum(
            np.vstack([
                norm.logpdf(means[:,b], loc=mu_mu[z,b], scale=mu_sigma[z,b]) 
                for b in range(n_bases)
            ])
            , axis=0
        )
        var_base_sum = np.nansum(
            np.vstack([
                invgamma.logpdf(variances[:,b], a=alpha[z,b], scale=beta[z,b]) 
                for b in range(n_bases)
            ])
            , axis=0
        )
        t_base_sum = np.nansum(
            np.vstack([
                norm.logpdf(dwells[:,b], loc=t_mu[z,b], scale=t_sigma[z,b]) 
                for b in range(n_bases)
            ])
            , axis=0
        )
        raw_post_base = np.sum([mu_base_sum, var_base_sum, t_base_sum], axis=0)
        posteriors.append(raw_post_base)

    posteriors = np.vstack(posteriors)
    probs = np.exp(posteriors - np.max(posteriors, axis=0, keepdims=True))
    total_probs = np.sum(probs, axis=0)
    norm_posteriors = probs / total_probs
    return norm_posteriors
