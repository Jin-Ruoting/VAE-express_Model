import pandas as pd, argparse, yaml
from pathlib import Path

def parse_attr(attr: str):
    kv = {}
    for x in attr.strip().strip(';').split(';'):
        x = x.strip()
        if not x: 
            continue
        parts = x.split(' ')
        k = parts[0]
        v = ' '.join(parts[1:]).strip('"')
        kv[k] = v
    return kv

def build_symbol_to_ensembl(gtf_path):
    cols = ['chrom','source','feature','start','end','score','strand','frame','attribute']
    gtf = pd.read_csv(gtf_path, sep='\t', comment='#', names=cols, low_memory=False,
                      compression='gzip' if str(gtf_path).endswith('.gz') else None)
    genes = gtf[gtf['feature']=='gene'].copy()
    genes['gene_id'] = genes['attribute'].apply(lambda s: parse_attr(s).get('gene_id') or parse_attr(s).get('gene'))
    genes['gene_name'] = genes['attribute'].apply(lambda s: parse_attr(s).get('gene_name') or parse_attr(s).get('Name'))
    genes['gene_type'] = genes['attribute'].apply(lambda s: parse_attr(s).get('gene_type') or parse_attr(s).get('gene_biotype'))
    # 先优先 protein_coding，其次 lncRNA/lincRNA，再其它
    priority = {'protein_coding':0, 'lncRNA':1, 'lincRNA':1}
    genes['prio'] = genes['gene_type'].map(priority).fillna(9)
    genes = genes.dropna(subset=['gene_id','gene_name'])
    genes = genes.sort_values(['gene_name','prio']).drop_duplicates(subset=['gene_name'], keep='first')
    mp = dict(zip(genes['gene_name'], genes['gene_id']))
    return mp

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--raw', required=True, help='Roadmap 57epi 解压后的 TSV')
    ap.add_argument('--gtf', required=True, help='与你的 promoters 相同版本的 GTF(.gtf 或 .gtf.gz)')
    ap.add_argument('--config', required=True, help='config/config.yaml，用于读取 eids')
    ap.add_argument('--promoters', required=True, help='promoters_2kb.hg38.bed')
    ap.add_argument('--keep_zero', action='store_true', help='不删全零基因')
    ap.add_argument('--out', default='exp/raw_exp.final.tsv')
    args = ap.parse_args()

    # 1) 构建符号->Ensembl 映射
    print('[INFO] Building symbol->Ensembl mapping from GTF...')
    sym2ens = build_symbol_to_ensembl(args.gtf)
    print(f'[INFO] Mapping size: {len(sym2ens)} symbols')

    # 2) 读取原始表达，标准化列名
    df = pd.read_csv(args.raw, sep='\t')
    first_col = df.columns[0]
    df.rename(columns={first_col: 'gene_symbol'}, inplace=True)

    # 3) 符号 -> Ensembl 映射并去未映射行
    df['gene_id'] = df['gene_symbol'].map(sym2ens)
    before = df.shape[0]
    df = df.dropna(subset=['gene_id']).drop(columns=['gene_symbol'])
    df['gene_id'] = df['gene_id'].astype(str)
    print(f'[INFO] Mapped genes: {df.shape[0]}/{before}')

    # 4) 同一 gene_id 聚合（均值）
    num_cols = [c for c in df.columns if c != 'gene_id']
    df = df.groupby('gene_id', as_index=False)[num_cols].mean()

    # 5) 仅保留 eids 列
    cfg = yaml.safe_load(open(args.config))
    eids = cfg['eids']
    keep = ['gene_id'] + [e for e in eids if e in df.columns]
    missing = [e for e in eids if e not in df.columns]
    if missing:
        print(f'[WARN] 缺失 EID 列（将被忽略）: {missing}')
    df = df[keep]

    # 6) 与 promoters 交集（保证 ID 完全对齐）
    prom = pd.read_csv(args.promoters, sep='\t', header=None,
                       names=['chrom','start','end','gene_id','score','strand'])
    gids = set(prom['gene_id'].astype(str).unique())
    df = df[df['gene_id'].isin(gids)]

    # 7) 剔除全零基因
    if not args.keep_zero:
        vals = df.iloc[:, 1:]
        nz = vals.sum(axis=1) > 0
        removed = (~nz).sum()
        df = df[nz]
        print(f'[INFO] Removed zero-only genes: {removed}')

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, sep='\t', index=False)
    print(f'[OK] Saved cleaned expression to {args.out} | genes={df.shape[0]} eids={df.shape[1]-1}')

if __name__ == '__main__':
    main()