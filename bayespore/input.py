import pod5
import pysam
from remora import io, refine_signal_map
import os, gzip
import pickle
import numpy as np


def process_inputs(bam_file, pod5_file, reverse_signal, kmer_table, contig, 
                   strand, start, end, out_dir):
    pod5_fh = pod5.Reader(pod5_file)
    bam_fh = pysam.AlignmentFile(bam_file)

    # level table from ont
    sig_map_refiner = refine_signal_map.SigMapRefiner(
        kmer_model_filename=kmer_table,
        scale_iters=0,
        do_fix_guage=True,
    )

    ref_reg = io.RefRegion(
        ctg=contig,
        strand=strand,
        start=start,
        end=end
    )

    bam_reads_reg = io.get_reg_bam_reads(ref_reg, bam_fh)
    io_reads = io.get_io_reads(bam_reads_reg, pod5_fh, reverse_signal=reverse_signal)

    seq, levels = io.get_ref_seq_and_levels_from_reads(
        ref_reg, bam_reads_reg, sig_map_refiner
    )

    read_ids, dwell, trimmean, trimsd = get_read_metrics(ref_reg, io_reads, sig_map_refiner)
    save_objects(dwell, trimmean, trimsd, read_ids, out_dir) # optional?
    return trimmean, trimsd, dwell, read_ids, seq, levels


def get_read_metrics(ref_reg, io_reads, sig_map_refiner):
    # NOTE: only reference anchored now

    read_ids = []
    dwell = []
    trimmean = []
    trimsd = []
    for io_read in io_reads:
        try:
            io_read.set_refine_signal_mapping(sig_map_refiner, ref_mapping=True)
            read_ids.append(io_read.read_id)
            metric = io_read.compute_per_base_metric(
                metric='dwell_trimmean_trimsd', 
                ref_anchored=True, 
                region=ref_reg, 
                signal_type="norm"
            )
            dwell.append(metric.get('dwell'))
            trimmean.append(metric.get('trimmean'))
            trimsd.append(metric.get('trimsd'))
        except Exception:
            continue

    read_ids = np.array(read_ids)
    dwell = np.log10(np.vstack(dwell))
    dwell[np.isinf(dwell)] = np.nan
    trimmean = np.vstack(trimmean)
    trimsd = np.vstack(trimsd)
    return read_ids, dwell, trimmean, trimsd


def save_objects(dwell, trimmean, trimsd, read_ids, out_dir):
    os.makedirs(f'{out_dir}/metrics', exist_ok=True)
    with gzip.open(f'{out_dir}/metrics/dwell.pkl.gz','wb') as f:
        pickle.dump(dwell, f)

    with gzip.open(f'{out_dir}/metrics/trimmean.pkl.gz','wb') as f:
        pickle.dump(trimmean, f)

    with gzip.open(f'{out_dir}/metrics/trimsd.pkl.gz','wb') as f:
        pickle.dump(trimsd, f)

    with gzip.open(f'{out_dir}/metrics/read_ids.pkl.gz','wb') as f:
        pickle.dump(read_ids, f)
