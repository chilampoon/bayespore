import pyro
import torch
import numpy as np
from pyro.infer import SVI, JitTraceEnum_ELBO, TraceEnum_ELBO, config_enumerate
import pyro.distributions as dist
import pyro.poutine as poutine
from pyro.params import param_store
from pyro.infer.autoguide.guides import AutoDelta
from pyro.optim import Adam, ClippedAdam
import warnings
warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
from scipy.stats import norm, halfnorm


@config_enumerate
def gmm_simple(data, ref_means=None, w_prior=None, w_prior_concent=1e-2, n_mod_status=2):
    mean, std, dwell = data
    n_reads, n_bases = mean.shape

    mean_observed = (~torch.isnan(mean))
    valid_mean = torch.nan_to_num(mean, nan=0.0)
    std_observed = (~torch.isnan(std))
    valid_std = torch.nan_to_num(std, nan=0.0)
    dwell_observed = (~torch.isnan(dwell))
    valid_dwell = torch.nan_to_num(dwell, nan=0.0)

    if w_prior is not None:
        assert type(w_prior) == torch.Tensor
        ω_concentration = w_prior_concent
        ω = pyro.sample(
            "ω",
            dist.Dirichlet(ω_concentration * w_prior)
        )
    else:
        # noninformative prior
        ω = pyro.sample(
            "ω",
            dist.Dirichlet(1e-8 * torch.tensor([1 / n_mod_status] * n_mod_status))
        )

    # modification params:
    if ref_means is None:
        ref_means = torch.zeros(n_bases)
    assert ref_means.shape == (n_bases,)

    with pyro.plate("mod_status", n_mod_status, dim=-2):
        with pyro.plate("base", n_bases, dim=-1) as b:
            # mean params
            μ_ν = pyro.sample(
                "μ_ν",
                dist.Normal(loc=ref_means[b], scale=.1)
            )
            μ_σ2 = pyro.sample(
                "μ_σ2",
                dist.InverseGamma(1, 1)
            )
            # std params
            τ_σ2 = pyro.sample(
                "τ_σ2",
                dist.InverseGamma(1, 1)
            )
            # dwell params
            δ_ν = pyro.sample(
                "δ_ν",
                dist.Normal(0, .1)
            )
            δ_σ2 = pyro.sample(
                "δ_σ2",
                dist.InverseGamma(1, 1)
            )

    # base params
    with pyro.plate("read_j", n_reads, dim=-2):
        z = pyro.sample(
            "z",
            dist.Categorical(ω)
        )

        with pyro.plate("obs_i", n_bases, dim=-1) as i:
            pyro.sample(
                "μ",
                dist.Normal(
                    loc=μ_ν[z, i], 
                    scale=torch.sqrt(μ_σ2[z, i])
                ).mask(mean_observed),
                obs = valid_mean
            )
            pyro.sample(
                "τ",
                dist.HalfNormal(
                    scale=torch.sqrt(τ_σ2[z, i])
                ).mask(std_observed),
                obs = valid_std
            )
            pyro.sample(
                "δ",
                dist.Normal(
                    loc=δ_ν[z, i], 
                    scale=torch.sqrt(δ_σ2[z, i])
                ).mask(dwell_observed),
                obs = valid_dwell
            )

# tmp_data = (torch.from_numpy(trimmean[:50,:]), torch.from_numpy(trimsd[:50,:]), torch.from_numpy(dwell[:50,:]))
# pyro.render_model(gmm_simple, model_args=(tmp_data,), render_distributions=True, render_params=True)   


def run_svi(data, model, **kwargs):
    pyro.clear_param_store()
    adam = ClippedAdam({'lr': kwargs.get('lr'),
                        'betas': [0.85, 0.99]})
    guide = AutoDelta(poutine.block(gmm_simple, expose=['ω', 'μ_ν', 'μ_σ2', 'τ_σ2', 'δ_σ2', 'δ_ν']))
    loss = JitTraceEnum_ELBO()
    svi = SVI(model, guide, adam, loss)
    for _ in range(kwargs.get('n_steps')):
        loss = svi.step(
            data=data, 
            ref_means=kwargs.get('ref_means'), 
            w_prior=kwargs.get('w_prior'),
            w_prior_concent=kwargs.get('w_prior_concent'), 
            n_mod_status=kwargs.get('n_mod_status'))

    w = pyro.param("AutoDelta.ω").detach().numpy()
    mu_loc = pyro.param("AutoDelta.μ_ν").detach().numpy()
    mu_scale = np.sqrt(pyro.param("AutoDelta.μ_σ2").detach().numpy())
    std_scale = np.sqrt(pyro.param("AutoDelta.τ_σ2").detach().numpy())
    d_loc = pyro.param("AutoDelta.δ_ν").detach().numpy()
    d_scale = np.sqrt(pyro.param("AutoDelta.δ_σ2").detach().numpy())

    return {'w': w, 'mu_loc':mu_loc, 'mu_scale':mu_scale, 
            'd_loc':d_loc, 'd_scale':d_scale, 'tau_scale':std_scale}


def compute_posteriors(data, params):
    mean, sd, dwell = data
    if type(mean) == torch.Tensor:
        mean = mean.detach().numpy()
        sd = sd.detach().numpy()
        dwell = dwell.detach().numpy()
    w = params.get('w')
    mu_loc = params.get('mu_loc')
    mu_scale = params.get('mu_scale')
    d_loc = params.get('d_loc')
    d_scale = params.get('d_scale')
    tau_scale = params.get('tau_scale')

    n_comps, n_bases = mu_loc.shape
    posteriors = []
    for z in range(n_comps):
        mu_base_sum = np.nansum(
            np.vstack([
                norm.logpdf(
                    mean[:,b], 
                    loc=mu_loc[z,b], 
                    scale=mu_scale[z,b]) 
                for b in range(n_bases)
            ])
            , axis=0
        )
        std_base_sum = np.nansum(
            np.vstack([
                halfnorm.logpdf(sd[:,b], loc=0, scale=tau_scale[z,b])
                for b in range(n_bases)
            ])
            , axis=0
        )
        dwell_base_sum = np.nansum(
            np.vstack([
                norm.logpdf(dwell[:,b], loc=d_loc[z,b], scale=d_scale[z,b]) 
                for b in range(n_bases)
            ])
            , axis=0
        )
        raw_post_base = np.sum([mu_base_sum, std_base_sum, dwell_base_sum], axis=0)
        raw_post_base += np.log(w[z])
        posteriors.append(raw_post_base)

    posteriors = np.vstack(posteriors)
    probs = np.exp(posteriors - np.max(posteriors, axis=0, keepdims=True))
    total_probs = np.sum(probs, axis=0)
    norm_posteriors = probs / total_probs
    return np.vstack(norm_posteriors).T
