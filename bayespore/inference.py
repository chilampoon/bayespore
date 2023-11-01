import torch
from collections import defaultdict
import numpy as np
import os, gzip, pickle
from bayespore.gmm import run_svi, compute_posteriors

from sklearn.cluster import KMeans
from scipy.spatial.distance import euclidean


def iter_data(
        trimmean,
        trimsd,
        dwell_log10,
        reads,
        levels,
        seq,
        seq_region,
        win_size,
        win_dist,
        model,
        n_mod_status,
        use_ref_levels,
        learning_rate,
        n_steps,
        weight_prior,
        weight_prior_concent,
        out_dir
    ):
    assert len(levels) == len(seq)

    results = defaultdict(dict)
    for base in range(seq_region[0], seq_region[1], win_size+win_dist):
        print(f'inferring {seq[base:base+win_size]} at {base}...')
        mean_win = trimmean[:,base:base+win_size]
        sd_win = trimsd[:,base:base+win_size]
        dwell_win = dwell_log10[:,base:base+win_size]

        data, reads_win = filter_inputs(mean_win, sd_win, dwell_win, reads)
        #print(data[0].shape)
        ref_means = torch.from_numpy(levels[base:base+win_size]) if use_ref_levels else None
        params = run_svi(data=data, model=model, lr=learning_rate, 
                    ref_means=ref_means, w_prior=weight_prior, 
                    w_prior_concent=weight_prior_concent, 
                    n_mod_status=n_mod_status, n_steps=n_steps)
        results[base]['params'] = params
        #results[base]['posteriors'] = compute_posteriors(data, params)
        results[base]['reads_win'] = reads_win
        #print(f'ref means = {ref_means}')
    
    with gzip.open(f'{out_dir}/metrics/params_{"_".join(seq_region)}.pkl.gz', 'wb') as p:
        pickle.dump(results, p)
    return results


def filter_inputs(means, sds, dwells_log10, reads, 
                mean_range=(-10, 10), min_sd=0, min_dwell=0.95):
    means = np.where((means < mean_range[0]) | (means > mean_range[1]), np.nan, means)
    sds = np.where(sds < min_sd, np.nan, sds)
    dwells_log10 = np.where(dwells_log10 < min_dwell, np.nan, dwells_log10)

    non_nan_mask = (~np.isnan(means).all(axis=1)) | \
                    (~np.isnan(sds).all(axis=1)) | \
                    (~np.isnan(dwells_log10).all(axis=1))
    means = torch.from_numpy(means[non_nan_mask])
    sds = torch.from_numpy(sds[non_nan_mask])
    dwells_log10 = torch.from_numpy(dwells_log10[non_nan_mask])
    reads = reads[non_nan_mask]
    return (means, sds, dwells_log10), reads


def manhattan_dist(a, b):
    return np.abs(a - b).sum(axis=-1)

MU_WEIGHTS = np.array([1, 1, 1, 1, 1]) 

def assign_classes(mu_locs, ref_means, mu_weights=MU_WEIGHTS):
    assert mu_locs.shape[1] == len(mu_weights) == len(ref_means)
    
    # discard outlier
    dists0 = np.array([manhattan_dist(c, ref_means) for c in mu_locs])
    furthest_idx = np.argmax(dists0)
    idx_to_class = np.delete(np.arange(len(mu_locs)), furthest_idx)
    mu_locs = mu_locs[idx_to_class]
    kickout = np.array([furthest_idx])

    # classify signal clusters
    weighted_mu_locs = mu_locs * mu_weights
    kmeans = KMeans(n_clusters=2, n_init=21)
    kmeans.fit(weighted_mu_locs)
    cluster_ids = kmeans.labels_
    centroids = kmeans.cluster_centers_
    
    # get confidence based on distance to reference, range [0, 1]
    ## TODO: update confidence calculation
    dists = np.array([manhattan_dist(c, ref_means) for c in centroids/mu_weights])
    # relative to the reference for now
    confidence_scores = dists/dists.sum()

    # assign classes
    furthest_cluster = np.argmax(dists)
    mod_cluster = idx_to_class[cluster_ids == furthest_cluster]

    return mod_cluster, kickout, confidence_scores[furthest_cluster], dists
