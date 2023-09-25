import pod5
import pysam
from remora import io, refine_signal_map
import torch
import numpy as np
from collections import defaultdict

## WIP

sig_map_refiner = refine_signal_map.SigMapRefiner(
    kmer_model_filename=level_table,
    scale_iters=0,
    do_fix_guage=True,
)


ref_reg = io.RefRegion(
    ctg='template1',
    strand='+',
    start=228,
    end=233
)

def extract_signals(bam, 
    pod5, 
    ref_reg, 
    sig_map_refiner, 
    reverse_signal=True):
    '''
    Extract normalized signals from bam and pod5 files.
      - bam (pysam.AlignmentFile): bam file
      - pod5 (pod5.Reader): pod5 file
      - win_coords (tuple): window coordinates (start, end)
    Returns: a list of ReadRefReg objects
    '''
    region_read_ref_regs = []
    region_bam_reads = io.get_reg_bam_reads(ref_reg, bam)
    io_reads = io.get_io_reads(region_bam_reads, pod5, reverse_signal)
    for io_read in io_reads:
        io_read.set_refine_signal_mapping(sig_map_refiner, ref_mapping=True)
        region_read_ref_regs.append(io_read.extract_ref_reg(ref_reg))
    return region_read_ref_regs


def make_signal_input(read_reg, window):
    # get lengths of signals in each bases
    sig_data_diff_len = []
    base_num_values = defaultdict(list)
    for read in read_reg:
        y = read.norm_signal
        x = read.ref_sig_coords
        read_sigs = [torch.tensor(y[(x >= b) & (x < b+1)], dtype=torch.float32) for b in range(window[0], window[1])]
        sig_data_diff_len.append(read_sigs)
        [base_num_values[i].append(len(y[(x >= b) & (x <= b+1)])) for i, b in enumerate(range(window[0], window[1]))]
    
    # simply calculate median and add paddings or subsample
    median_dps = [np.ceil(np.median(dps)) for dps in base_num_values.values()]
    len_dps = int(max(median_dps))

    sig_data_same_len = []
    for read_sigs in sig_data_diff_len:
        sigs = []
        for read_sig in read_sigs:
            if len(read_sig) <= len_dps:
                # padding
                read_sigs = torch.cat([read_sig, torch.full((len_dps - len(read_sig),), float('nan'))])
            else:
                # subsampling
                indices = torch.randperm(len(read_sig))[:len_dps]
                read_sigs = read_sig[indices]
            sigs.append(read_sigs)
        sig_data_same_len.append(torch.hstack(sigs))
    sig_data_same_len = torch.stack(sig_data_same_len)
    return sig_data_same_len, median_dps

