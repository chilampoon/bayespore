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
import warnings
warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)


@config_enumerate
def modification_model(data, cov, alpha, beta, n_mod_status=2):
    n_reads, n_datapoints = data.shape
    is_observed = (~torch.isnan(data))
    valid_mat = torch.nan_to_num(data, nan=0.0)

    # modification params:
    with pyro.plate("mod_k", n_mod_status, dim=-1):
        μ_kt = pyro.sample(
            "μ_kt", 
            dist.MultivariateNormal(torch.zeros(n_datapoints), cov)
        )
        # weight for modified/unmodified
        ω = pyro.sample(
            "ω",
            dist.Gamma(1/n_mod_status, 1)
        )
    
    # noise:
    τ_2 = pyro.sample(
        'τ_2', 
        dist.InverseGamma(alpha, beta) # concentration, rate
    )

    # read params:
    with pyro.plate("read_i", n_reads, dim=-1):
        m = pyro.sample("m_i", dist.Categorical(ω))
        
        y_it = pyro.sample(
            "y_it",
            dist.Normal(μ_kt[m, :], τ_2).mask(is_observed).to_event(1),
            obs=valid_mat
        )
# pyro.render_model(modification_model, model_args=(data, kmer_cov, 1, 1), 
#                   render_distributions=True, render_params=True)                    

def kmer_cov_matrix(k, n_datapoints_per_base, variance, lengthscale):
    # Coregionalized GPs
    base_cov = create_cov_matrix(n_datapoints_per_base, variance, lengthscale) 
    inter_bases_corr = torch.eye(k)
    kmer_cov = torch.kron(base_cov, inter_bases_corr)
    return kmer_cov

def create_cov_matrix(input_dim, variance, lengthscale):
    # use RBF by default
    pyro.clear_param_store()
    kernel = gp.kernels.RBF(
        input_dim=input_dim,
        variance=torch.tensor(variance), 
        lengthscale=torch.tensor(lengthscale)
    )
    cov = kernel(torch.tensor(np.arange(input_dim))) + (torch.eye(input_dim) * 1e-3)
    pyro.clear_param_store()
    return cov.detach()

def run_svi(data, kmer_cov, alpha, beta, n_mod_status, learning_rate, n_steps):
    pyro.clear_param_store()
    adam = ClippedAdam({'lr': learning_rate, 
                        'betas': [0.85, 0.99]})
    guide = AutoDelta(poutine.block(modification_model, 
                                    expose=['ρ_k', 'μ_kt', 'τ_2']))
    loss = JitTraceEnum_ELBO()
    svi = SVI(modification_model, guide, adam, loss)
    for step in range(n_steps):
        loss = svi.step(data, kmer_cov, alpha, beta, n_mod_status)

    # get and return learned params
    rho = pyro.param("AutoDelta.ρ_k").detach().numpy()
    rho = rho / np.sum(rho)
    mu = pyro.param("AutoDelta.μ_kt").detach().numpy()
    tau_2 = pyro.param("AutoDelta.τ_2").detach().numpy()
    return rho, mu, tau_2, loss

def get_read_assignments(data, mu, tau_2, rho):
    m_posterior = m_i_posterior(data, mu, tau_2, rho)
    return np.argmax(m_posterior, axis=0)

def m_i_posterior(data, mu, tau_2, rho):
    log_posterior = []
    for k in range(len(rho)):
        llh_k = -0.5*((data - mu[k])**2)/tau_2 - 0.5*np.log(2*np.pi*tau_2)
        llh_k = np.nansum(llh_k, axis=1)
        llh_k = np.log(rho[k]) + llh_k
        log_posterior.append(llh_k)
    
    log_posterior = np.array(log_posterior)
    posterior_prob_for_m = np.exp(log_posterior - logsumexp(log_posterior, axis=0))
    return posterior_prob_for_m

def mu_per_base(mu, n_mod_status, n_datapoints):
    mu_reshaped = mu.reshape(n_mod_status, -1, n_datapoints)
    mean_base = np.mean(mu_reshaped, axis=-1)
    return mean_base
