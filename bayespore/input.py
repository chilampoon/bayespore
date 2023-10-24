import pod5
import pysam
from remora import io, refine_signal_map
import torch
import numpy as np
from collections import defaultdict

## WIP

SIG_MIN = -10
SIG_MAX = 10
MIN_DPS = 5


def process_inputs(bam_file,
        pod5_file,
        level_table,
        region_bed,
        window_size,
        min_win_datapoints=10,
        reverse_signal=True):
    pod5_fh = pod5.Reader(pod5_file)
    bam_fh = pysam.AlignmentFile(bam_file)

    sig_map_refiner = refine_signal_map.SigMapRefiner(
        kmer_model_filename=level_table,
        scale_iters=0,
        do_fix_guage=True,
    )

    # fix it for 1a data for now... TODO: process bed inputs and loop over regions
    ref_reg = io.RefRegion(
        ctg='template1',
        strand='+',
        start=0,
        end=1218
    )

    bam_reads_reg = io.get_reg_bam_reads(ref_reg, bam_fh)
    io_reads = io.get_io_reads(bam_reads_reg, pod5_fh, reverse_signal=True)

    seq, levels = io.get_ref_seq_and_levels_from_reads(
        ref_reg, bam_reads_reg, sig_map_refiner
    )


    return read_sigs_in_win

def get_metrics(io_reads):
    # extract per base metrics
    read_to_metrics = {}
    base_to_reads = defaultdict(list)

    for io_read in io_reads:
        # only reference anchored now
        try:
            io_read.set_refine_signal_mapping(sig_map_refiner, ref_mapping=True)
        except Exception:
            continue
        read = io_read.extract_ref_reg(ref_reg)
        compute_per_base_metrics(read, read_to_metrics, base_to_reads, min_dps=MIN_DPS)
    return read_to_metrics, base_to_reads

def compute_per_base_metrics(
        read, 
        read_to_metrics, 
        base_to_reads, 
        min_dps=5
    ):
    '''
    return ref coordinate (0-base), mean, sd, dwell for each base
    ndarray with shape (4, n_ref_bases)
    '''
    ref_to_sig = read.ref_sig_coords
    sig = read.norm_signal
    coords_int = np.unique(np.floor(ref_to_sig)).astype(int)
    means = []; sds = []; dwells = []
    pass_bases = []
    for b in coords_int:
        mask = (ref_to_sig >= b) & (ref_to_sig <= b+1)
        base_sig = sig[mask]
        base_sig = base_sig[(base_sig <= SIG_MAX) & (base_sig >= SIG_MIN)]
        if len(base_sig) >= min_dps:
            pass_bases.append(True)
            base_to_reads[b].append(read.read_id)
            
            means.append(np.mean(base_sig))
            sds.append(np.std(base_sig))
            dwells.append(len(base_sig))
        else:
            pass_bases.append(False)
    read_info = np.vstack([coords_int[pass_bases], means, sds, dwells])
    read_to_metrics[read.read_id] = read_info

MET_IDX_MAP = {
    'ref': 0,
    'mean': 1,
    'sd': 2,
    'dwell': 3
}

def get_mat_wo_nan(reg, read_to_metrics, base_to_reads, metric='mean'):
    '''
    fetch reads within region, construct a matrix
    '''
    # get reads within region
    reads_with_all_bases = set(base_to_reads[reg[0]])
    for b in range(reg[0]+1, reg[1]+1):
        reads_with_all_bases.intersection_update(base_to_reads[b])

    read_metrics = []
    for r in reads_with_all_bases:
        r_metrics = read_to_metrics[r]
        reg_mask = (r_metrics[MET_IDX_MAP['ref']] >= reg[0]) & \
                (r_metrics[MET_IDX_MAP['ref']] <= reg[1])
        read_metrics.append(r_metrics[MET_IDX_MAP[metric]][reg_mask])
    return np.vstack(read_metrics)


def gen_model_input(base_to_reads, read_to_metrics, ref_start, n_bases=3):
    '''
    Input: 
        - base_to_reads: dict of base to read id mapping
        - read_to_metrics: dict of read to base metrics mapping
        - ref_start: start reference coord, 1-base
        - n_bases: number of bases to infer
    returns array of shape (n_reads, n_bases * 3)
    '''
    ref_start -= 1
    all_reads = set()
    for b in range(ref_start, ref_start+n_bases):
        all_reads.update(base_to_reads[b])

    mat = []
    read_ids = []
    for read in all_reads:
        if read not in read_to_metrics:
            continue
        read_ids.append(read)
        read_metrics = read_to_metrics[read]

        mus = []
        sigmas = []
        ts = []
        for b in range(ref_start, ref_start+n_bases):
            if b not in read_metrics[0]:
                mus.append(np.nan); sigmas.append(np.nan); ts.append(np.nan)
            else:
                base_mask = (read_metrics[0] == b)
                mus.append(read_metrics[1][base_mask])
                sigmas.append(read_metrics[2][base_mask])
                ts.append(read_metrics[3][base_mask])
        
        read_vec = np.hstack([np.hstack(mus), np.hstack(sigmas), np.hstack(ts)])
        mat.append(read_vec)

    mat = np.vstack(mat)
    mat[:, -3:] = np.log10(mat[:, -3:])
    mat = torch.from_numpy(mat).float()

    return mat, read_ids
