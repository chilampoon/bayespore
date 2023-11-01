"write bedRMod file"

from bayespore.inference import filter_inputs, assign_classes
from bayespore.gmm import compute_posteriors
import numpy as np

FORMAT = 'bedRModv1.6'
ORG = 'IVT'
MOD_TYPE = 'RNA' # or DNA
ASSEM = 'transcriptome'
ANNO_SOURCE = 'NA'
ANNO_VER = 'NA'
SEQ_PLATFORM = 'MinION_Mk1b_FLO-FLG001'
BASECALL_MODEL = 'rna002_70bps_hac@v3'
WORKFLOW = 'bayespore'
EXPERIMENT = 'test'
EXTERNAL_SRC = 'NA'

MOD_REF = {
    '5mC': 'C',
    'm6A': 'A',
}

MOD_RGB = {
    '5mC': '244,164,96',
    'm6A': '176,196,222',
}

CONFIDENCE_SCALE = 1000

def output_rmod_bed(data, results, mod_type, contig, strand, out_dir,
                    middel_base_dist, post_cutoff):
    read_out = f'{out_dir}/reads.bedrmod'
    site_out = f'{out_dir}/sites.bedrmod'

    with open(read_out, 'w') as read_out, open(site_out, 'w') as site_out:
        header_rows(read_out)
        header_rows(site_out)
        parse_results(data, results, mod_type, contig, strand,
                read_out, site_out, middel_base_dist, post_cutoff)


def header_rows(out):
    out.write(f'#fileformat={FORMAT}\n')
    out.write(f'#organism={ORG}\n')
    out.write(f'#modification_type={MOD_TYPE}\n')
    out.write(f'#assembly={ASSEM}\n')
    out.write(f'#annotation_source={ANNO_SOURCE}\n')
    out.write(f'#annotation_version={ANNO_VER}\n')
    out.write(f'#sequencing_platform={SEQ_PLATFORM}\n')
    out.write(f'#basecalling={BASECALL_MODEL}\n')
    out.write(f'#bioinformatics_workflow={WORKFLOW}\n')
    out.write(f'#experiment={EXPERIMENT}\n')
    out.write(f'#external_source={EXTERNAL_SRC}\n')
    out.write('\t'.join(['#chrom', 'chromStart', 'chromEnd', 'name', 'score', 'strand', 'thickStart',
        'thickEnd', 'itemRgb', 'coverage', 'frequency', 'refBase'])+'\n')


def parse_results(data, results, mod_type, contig, strand, read_out, site_out, 
                  middle_base_dist=2, post_cutoff=.95):
    # for each k bases window
    trimmean, trimsd, dwell, read_ids, seq, levels = data

    for pos, res in results.items():
        params = res['params']
        mu_loc = params.get('mu_loc')
        win_size = mu_loc.shape[1]

        # repeat, integrate into iter_data (?)
        mean_win = trimmean[:,pos:pos+win_size]
        sd_win = trimsd[:,pos:pos+win_size]
        dwell_win = dwell[:,pos:pos+win_size]
        data, reads_win = filter_inputs(mean_win, sd_win, dwell_win, read_ids)
        posteriors = compute_posteriors(data, params)

        read_labels0 = np.argmax(posteriors, axis=1)
        read_probs = np.max(posteriors, axis=1)
        ref_means = levels[pos:pos+win_size]

        try:
            mod_cluster, kickout, site_confidence, dists = assign_classes(mu_loc, ref_means)
            print(kickout, ' kickout')
        except Exception as e:
            print(e)
            continue
        
        # quick check
        close_dist = 0.5
        if abs(np.diff(dists)) < close_dist and mod_ratio > 0.5:
            mod_cluster = set(range(len(w))) - set(mod_cluster) - set(kickout)
            mod_cluster = np.array(list(mod_cluster))

        read_labels = np.where(np.isin(read_labels0, mod_cluster), 1, 0)
        read_labels[np.isin(read_labels0, kickout)] = 0
        
        pos += middle_base_dist
        mod_reads = reads_win[(read_labels == 1) & (read_probs > post_cutoff)]
        reads_win = reads_win[~np.isin(read_labels0, kickout)]

        # write read-level output
        for read_id, label, prob in zip(reads_win, read_labels, read_probs):
            if label == 1:
                r = read_bed_row(contig, read_id, pos, mod_type, prob, strand)
                read_out.write(r)
        
        # write site-level output
        mod_ratio = len(mod_reads) / len(reads_win)
        s = site_bed_row(contig, pos, mod_type, site_confidence, strand, len(reads_win), mod_ratio)
        site_out.write(s)


def read_bed_row(contig, read_id, pos, mod_type, confidence, strand):
    # pos is 0-base
    coverage = 1
    freq = 100
    chrom = f'{contig}:{read_id}'
    score = int(confidence * CONFIDENCE_SCALE)
    rgb = MOD_RGB[mod_type]
    ref_base = MOD_REF[mod_type]
    items = [chrom, pos, pos+1, mod_type, score, strand, pos, pos+1,
             rgb, coverage, freq, ref_base]
    return '\t'.join(map(str, items)) + '\n'


def site_bed_row(contig, pos, mod_type, confidence, strand, coverage, freq):
    # pos is 0-base
    chrom = contig
    score = int(confidence * CONFIDENCE_SCALE)
    rgb = MOD_RGB[mod_type]
    freq = int(freq * 100)
    ref_base = MOD_REF[mod_type]
    items = [chrom, pos, pos+1, mod_type, score, strand, pos, pos+1,
             rgb, coverage, freq, ref_base]
    return '\t'.join(map(str, items)) + '\n'
