import pod5
import pysam
from remora import io, refine_signal_map
import torch
import numpy as np
from collections import defaultdict

## WIP

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

    read_sigs_in_win = extract_window_signals(bam_fh, pod5_fh, ref_reg, 
                    window_size, sig_map_refiner,
                    min_win_datapoints, reverse_signal)
    return read_sigs_in_win

def extract_window_signals(bam, pod5, ref_reg, window_size, sig_map_refiner,
                           min_win_datapoints, reverse_signal):
    '''
    Extract normalized signals from bam and pod5 files.
        - bam (pysam.AlignmentFile): bam file
        - pod5 (pod5.Reader): pod5 file
    Returns: a list of ReadRefReg objects
    '''
    window_list = [(i,i+window_size) for i in range(ref_reg.start, ref_reg.end, window_size)]
    read_sigs_in_win = defaultdict(lambda: defaultdict(np.array)) #{win:{read:sig}}

    bam_reads_reg = io.get_reg_bam_reads(ref_reg, bam)
    io_reads = io.get_io_reads(bam_reads_reg, pod5, reverse_signal)
    for io_read in io_reads:
        # only reference anchored now
        io_read.set_refine_signal_mapping(sig_map_refiner, ref_mapping=True)
        read = io_read.extract_ref_reg(ref_reg)

        coords = read.ref_sig_coords
        sig = read.norm_signal
        read_id = read.read_id
        for s, e in window_list:
            if s > read.ref_reg.end:
                break
            mask = (coords >= s) & (coords <= e)
            filtered_coords = coords[mask]
            if len(filtered_coords) >= min_win_datapoints:
                read_sigs_in_win[(s, e)][read_id] = sig[mask]
    return read_sigs_in_win

def make_signal_input(window, read_sigs_in_win):
    '''Generate input matrix where signals from each base 
       have the same number of datapoints
    '''
    # get lengths of signals in each bases
    sig_data_diff_len = []
    reads_in_window = []
    base_num_values = defaultdict(list)
    read_sigs = read_sigs_in_win[window]
    for read, base_sigs in read_sigs.items():
        reads_in_window.append(read)
        [base_num_values[i].append(len(b)) for i, b in enumerate(base_sigs)]
        base_sigs = [torch.tensor(s, dtype=torch.float32) for s in base_sigs]
        sig_data_diff_len.append(base_sigs)
    
    # simply calculate median and add paddings or subsample
    median_dps = [np.ceil(np.median(dps)) for dps in base_num_values.values()]
    len_dps = int(max(median_dps))

    sig_data_same_len = []
    for base_sigs in sig_data_diff_len:
        sigs = []
        for base_sig in base_sigs:
            if len(base_sig) <= len_dps:
                # padding
                base_sig = torch.cat([base_sig, torch.full((len_dps - len(base_sig),), float('nan'))])
            else:
                # subsampling
                indices = torch.randperm(len(base_sig))[:len_dps]
                base_sig = base_sig[indices]
            sigs.append(base_sig)
        sig_data_same_len.append(torch.hstack(sigs))
    sig_data_same_len = torch.stack(sig_data_same_len)
    return sig_data_same_len, median_dps, reads_in_window
