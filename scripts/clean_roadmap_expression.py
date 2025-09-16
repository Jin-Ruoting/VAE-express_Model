import re, argparse, yaml
import pandas as pd
from pathlib import Path

def strip_version(s: pd.Series) -> pd.Series:
    return s.astype(str).str.replace(r'\.\d+$','', regex=True)

def is_ensembl_series(s: pd.Series, prefix='ENSG') -> bool:
    pat = re.compile(rf'^{prefix}\d+(\.\d+)?$')
    sample = s.dropna().astype(str).head(200)
    if sample.empty: return False
    return (sample.map(lambda x: bool(pat.match(x))).mean() > 0.8)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--raw', required=True, help='Roadmap 57epi 解压后的 TSV')
    ap.add_argument('--config', required=True, help='config/config.yaml（读取 eids）')
    ap.add_argument('--promoters', required=True, help='promoters_2kb.hg38.bed')
    ap.add_argument('--keep_zero', action='store_true', help='不删全零基因')
    ap.add_argument('--out', default='exp/raw_exp.final.tsv')
    args = ap.parse_args()

    # 1) 读取
    df = pd.read_csv(args.raw, sep='\t')
    print(f'[INFO] Raw shape: {df.shape}')
    first = df.columns[0]

    # 2) 识别 gene_id 列（Ensembl 直接用；否则当 gene_symbol 处理）
    if is_ensembl_series(df[first]):
        print(f'[INFO] Detected Ensembl gene_id in column: {first}')
        df.rename(columns={first: 'gene_id'}, inplace=True)
        df['gene_id'] = strip_version(df['gene_id'])
    else:
        # 兼容“第一列是符号”的情况（不做映射，只重命名，便于快速排错）
        print(f'[WARN] First column seems not Ensembl. Using it as gene_id (symbol).')
        df.rename(columns={first: 'gene_id'}, inplace=True)

    # 3) 去重复（均值聚合）
    num_cols = [c for c in df.columns if c != 'gene_id']
    df = df.groupby('gene_id', as_index=False)[num_cols].mean()
    print(f'[INFO] After groupby(unique gene_id): {df.shape}')

    # 4) 仅保留 config.eids
    cfg = yaml.safe_load(open(args.config))
    eids = cfg['eids']
    present = [e for e in eids if e in df.columns]
    missing = [e for e in eids if e not in df.columns]
    if not present:
        raise RuntimeError(f'表达矩阵中未找到任何配置的EID列。missing={missing[:10]}')
    if missing:
        print(f'[WARN] 缺失 EID 列（忽略）: {missing}')
    df = df[['gene_id'] + present]
    print(f'[INFO] After EID subset: {df.shape}')

    # 5) 与 promoters 交集（promoters 是 Ensembl；去版本号后比对）
    prom = pd.read_csv(args.promoters, sep='\t', header=None,
                       names=['chrom','start','end','gene_id','score','strand'])
    prom_ids = strip_version(prom['gene_id']).unique()
    before_inter = df.shape[0]
    df = df[df['gene_id'].isin(set(prom_ids))]
    print(f'[INFO] Intersect with promoters: {df.shape[0]}/{before_inter}')

    # 6) 删全零
    if not args.keep_zero:
        vals = df.iloc[:, 1:]
        zero_only = (vals.sum(axis=1) == 0)
        removed = int(zero_only.sum())
        df = df[~zero_only]
        print(f'[INFO] Removed zero-only genes: {removed}')

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, sep='\t', index=False)
    print(f'[OK] Saved cleaned expression to {args.out} | genes={df.shape[0]} eids={df.shape[1]-1}')

if __name__ == '__main__':
    main()