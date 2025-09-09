import argparse
import pandas as pd
import gzip

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gtf', required=True)
    ap.add_argument('--promoter_bp', type=int, default=2000)  # 总长度
    ap.add_argument('--out_bed', required=True)
    return ap.parse_args()

def parse_kv(attr, keys):
    kv = {}
    for x in attr.strip().strip(';').split(';'):
        x = x.strip()
        if not x: continue
        parts = x.split(' ')
        k = parts[0]; v = ' '.join(parts[1:]).strip('"')
        kv[k] = v
    for k in keys:
        if k in kv: return kv[k]
    return None

def main():
    args = parse_args()
    cols = ['chrom','source','feature','start','end','score','strand','frame','attribute']
    gtf = pd.read_csv(args.gtf, sep='\t', comment='#', names=cols, low_memory=False,
                      compression='gzip' if args.gtf.endswith('.gz') else None)
    sub = gtf[gtf['feature'].isin(['transcript','gene'])].copy()
    sub['gene_id']  = sub['attribute'].apply(lambda s: parse_kv(s, ['gene_id','gene']))
    sub['tx_type']  = sub['attribute'].apply(lambda s: parse_kv(s, ['transcript_biotype','transcript_type']))
    sub['gene_type']= sub['attribute'].apply(lambda s: parse_kv(s, ['gene_biotype','gene_type']))
    keep = {'protein_coding','lncRNA','lincRNA'}
    if sub['tx_type'].notna().any():
        sub = sub[(sub['tx_type'].isna()) | (sub['tx_type'].isin(keep))]
    elif sub['gene_type'].notna().any():
        sub = sub[(sub['gene_type'].isna()) | (sub['gene_type'].isin(keep))]
    sub['tss'] = sub.apply(lambda r: r['start'] if r['strand']=='+' else r['end'], axis=1)
    def pick_5prime(g):
        strand = g.iloc[0]['strand']
        idx = g['tss'].idxmin() if strand=='+' else g['tss'].idxmax()
        return g.loc[idx]
    agg = sub.groupby(['gene_id','chrom','strand'], as_index=False).apply(pick_5prime).reset_index(drop=True)
    half = args.promoter_bp // 2
    def mk_row(r):
        s = max(0, int(r['tss']) - half); e = int(r['tss']) + half
        return pd.Series([r['chrom'], s, e, r['gene_id'], 0, r['strand']])
    bed = agg.apply(mk_row, axis=1); bed.columns = ['chrom','start','end','gene_id','score','strand']
    bed = bed.sort_values(['chrom','start','end'])
    bed.to_csv(args.out_bed, sep='\t', header=False, index=False)
    print(f"[OK] promoters saved to {args.out_bed}, n={len(bed)}")

if __name__ == '__main__':
    main()