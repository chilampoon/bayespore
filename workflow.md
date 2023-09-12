## Set up environment 

```
conda create -n ont python=3.11
conda activate ont
pip install pod5
```

## Base calling:
First convert multiple fast5s to single pod5
```
cd RMaP_challenge_ManjaMerz_DataSet/challenge_1/challenge_1a/1a
pod5 convert fast5 ./fast5/*fast5 --output 1a.pod5 -t 12
```

then use dorado with `--emit-moves`
```
dorado basecaller ~/dorado_model/rna002_70bps_hac\@v3/ ./pod5 -x cuda:all --emit-moves > 1a.basecall.bam
```

## Alignment:
Here use `map-ont` for ivt because reads are unsplicded. For real bio data, should use the splice option.

`-T*` and `-y` are to pass all tags from basecall bam to fastq then to minimap2. `--MD` is needed.
```
samtools fastq -T* 1a.basecall.bam | \
  minimap2 -y -ax map-ont -k14 --MD -t 8 reference/template1.fa - | \
  samtools view -bS --no-PG - | samtools sort -@ 8 > 1a.bam
samtools index -@ 8 1a.bam
```
## Processing with remora API
Download 5mer table from ONT:
```
wget https://github.com/nanoporetech/kmer_models/raw/master/rna_r9.4_180mv_70bps/5mer_levels_v1.txt
```

