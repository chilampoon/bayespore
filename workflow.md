## Set up environment 

```
conda create -n ont python=3.11
conda activate ont

# install pod5, dorado, remora, samtools, minimap2 ...
pip install pod5 ont-remora
conda install -c bioconda samtools==1.16 minimap2==2.26
wget https://cdn.oxfordnanoportal.com/software/analysis/dorado-0.4.3-linux-x64.tar.gz | tar -xvzf -
```

## Basecalling
First convert multiple fast5s to single pod5
```
cd RMaP_challenge_ManjaMerz_DataSet/challenge_1/challenge_1a/1a
pod5 convert fast5 ./fast5/*fast5 --output 1a.pod5 -t 12
```

then run dorado with `--emit-moves`
```
dorado basecaller ~/dorado_model/rna002_70bps_hac\@v3/ ./pod5 -x cuda:all --emit-moves > 1a.basecall.bam
```

## Alignment
Here use `map-ont` for ivt data because reads are unspliced. For real bio data, should use the splice option `splice -uf`.

`-T*` and `-y` are to pass all tags from basecall bam to fastq then to minimap2. `--MD` is needed.
```
samtools fastq -T* 1a.basecall.bam | \
  minimap2 -y -ax map-ont -k5 --secondary=no --MD -t 8 reference/template1.fa - | \
  samtools view -F 2324 -bS --no-PG - | samtools sort -@ 8 > 1a.bam
samtools index -@ 8 1a.bam
```
## Getting reference kmer levels
Download direct RNA seq 5mer table from ONT:
```
wget https://github.com/nanoporetech/kmer_models/raw/master/rna_r9.4_180mv_70bps/5mer_levels_v1.txt
```

## Running
this is another example:
```
sample=m6A_2
bayespore run -b align/${sample}.bam \
  -p pod5/${sample}.pod5 \
  --kmer_table 5mer_levels_v1.txt \
  --reverse_signal \
  --contig reference \
  --start 0 \
  --end 1908 \
  --iter_region 1-1901 \
  --win_size 1 \
  --middel_base_dist 0 \
  --n_mod_status 3 \
  --use_ref_levels \
  --mod_type m6A \
  -o ${sample}
```

