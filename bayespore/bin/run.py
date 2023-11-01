import click
import logging
import os
import torch

from bayespore.input import process_inputs
from bayespore.inference import iter_data
from bayespore.bed_out import output_rmod_bed
from bayespore.gmm import gmm_simple


logging.basicConfig(level=logging.INFO,
                    format='[%(asctime)s] %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')

@click.group('bayespore')
def bayespore():
    '''
    Generative model for nanopore signals
    '''
    pass


@bayespore.command('run')
@click.option('-b', '--bam', required=True, type=click.Path(exists=True),
              help="BAM file")
@click.option('-p', '--pod5', required=True, type=click.Path(exists=True),
              help="pod5 file")
@click.option('--kmer_table', required=True, type=click.Path(exists=True),
              help="table of reference kmer levels")
@click.option('--reverse_signal', is_flag=True, show_default=True, default=False,
              help="Reverse signal, specify if for direct RNA-seq")

@click.option('--contig', show_default=True, default="template1",
              help="contig name of the reference")
@click.option('--strand', show_default=True, default="+",
              help="Reference strand")
@click.option('--start', show_default=True, default=0,
              help="Start reference position (0-base)")
@click.option('--end', type=int, required=True,
              help="End reference position (0-base)")

@click.option('--iter_region', show_default=True, default="0-1000",
              help="Reference region to iterate over, format: start-end")
@click.option('--win_size', show_default=True, default=5,
              help="Window size for modeling bases")
@click.option('--win_dist', show_default=True, default=0,
              help="Distance between windows")
@click.option('--middel_base_dist', show_default=True, default=2,
              help="Distance between window start and middle base")

@click.option('--n_mod_status', show_default=True, default=4,
              help="Number of modification status")
@click.option('--use_ref_levels', is_flag=True, show_default=True, default=False,
              help="Use reference levels of means as prior")
@click.option('--learning_rate', show_default=True, default=0.01,
              help="Learning rate for SVI")
@click.option('--n_steps', show_default=True, default=2000,
              help="Number of SVI steps")
@click.option('--weight_prior', show_default=True, default=None,
              help="Prior for modification rates, e.g 0.1,0.9")
@click.option('--weight_prior_concent', show_default=True, default=1e-2,
              help="Dirichlet concentration for weight prior")
@click.option('--mod_type', show_default=True, default="5mC",
              help="Modification type")
@click.option('--min_read_posterior_prob', show_default=True, default=0.9,
              help="Minimal posterior probability of a read to be considered as modified")

@click.option('-o', '--out_dir', default=os.getcwd(),
              help="Output directory, default is current working directory")


def run_steps(**kwargs):
    '''
    Parse inputs and run the model
    '''
    # TODO: input a bed file then get contig, st, ed from it

    inputs = process_inputs(
        kwargs.get('bam'), 
        kwargs.get('pod5'),
        kwargs.get('reverse_signal'),
        kwargs.get('kmer_table'),
        kwargs.get('contig'),
        kwargs.get('strand'),
        kwargs.get('start'),
        kwargs.get('end'),
        kwargs.get('out_dir')
    )
    read_ids, dwell, trimmean, trimsd, seq, levels = inputs

    w_prior = kwargs.get('weight_prior')
    w_prior = torch.Tensor(w_prior.split(','), type=torch.float32) if w_prior is not None else None

    results = iter_data(
        trimmean,
        trimsd,
        dwell,
        read_ids,
        levels,
        seq,
        kwargs.get('iter_region'),
        kwargs.get('win_size'),
        kwargs.get('win_dist'),
        gmm_simple,
        kwargs.get('n_mod_status'),
        kwargs.get('use_ref_levels'),
        kwargs.get('learning_rate'),
        kwargs.get('n_steps'),
        w_prior,
        kwargs.get('weight_prior_concent'),
        kwargs.get('out_dir')
    )

    output_rmod_bed(
        inputs, 
        results, 
        kwargs.get('mod_type'),
        kwargs.get('contig'),
        kwargs.get('strand'),
        kwargs.get('out_dir'),
        kwargs.get('middel_base_dist'),
        kwargs.get('post_cutoff')
    )
