# bayespore

A proof-of-concept Bayesian generative model aiming to de novo detect RNA modifications with ONT raw signals in single samples. Currently, it has been applied to one of the RMAP Challenge 2023 datasets.

Update: our unsupervised model demonstrates decent accuracy in predicting modification positions and exhibits lower error rates in estimating modification frequencies compared to a supervised machine learning-based model([link](https://www.researchsquare.com/article/rs-5241143/v1))! We plan to further refine the model and publish our findings — stay tuned.


### Installation

```console
git clone https://github.com/chilampoon/bayespore.git
cd bayespore
pip install -e .
```

```console
$ bayespore run --help
Usage: bayespore run [OPTIONS]

  Parse inputs and run the model

Options:
  -b, --bam PATH                  BAM file  [required]
  -p, --pod5 PATH                 pod5 file  [required]
  --kmer_table PATH               table of reference kmer levels  [required]
  --reverse_signal                Reverse signal, specify if for direct RNA-
                                  seq
  --contig TEXT                   contig name of the reference  [default:
                                  template1]
  --strand TEXT                   Reference strand  [default: +]
  --start INTEGER                 Start reference position (0-base)  [default:
                                  0]
  --end INTEGER                   End reference position (0-base)  [required]
  --iter_region TEXT              Reference region to iterate over, format:
                                  start-end  [default: 0-1000]
  --win_size INTEGER              Window size for modeling bases  [default: 5]
  --win_dist INTEGER              Distance between windows  [default: 0]
  --middel_base_dist INTEGER      Distance between window start and middle
                                  base  [default: 2]
  --n_mod_status INTEGER          Number of modification status  [default: 4]
  --use_ref_levels                Use reference levels of means as prior
  --learning_rate FLOAT           Learning rate for SVI  [default: 0.01]
  --n_steps INTEGER               Number of SVI steps  [default: 2000]
  --weight_prior TEXT             Prior for modification rates, e.g 0.1,0.9
  --weight_prior_concent FLOAT    Dirichlet concentration for weight prior
                                  [default: 0.01]
  --mod_type TEXT                 Modification type  [default: 5mC]
  --min_read_posterior_prob FLOAT
                                  Minimal posterior probability of a read to
                                  be considered as modified  [default: 0.9]
  -o, --out_dir TEXT              Output directory, default is current working
                                  directory
  --help                          Show this message and exit.
```
