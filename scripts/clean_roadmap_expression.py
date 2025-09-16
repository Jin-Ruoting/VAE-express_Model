import re, argparse, yaml
import pandas as pd
from pathlib import Path

ENSEMBL_PAT = re.compile(r'^ENSG\d+(\.\d+)?$')

def strip_version(s: pd.Series) -> pd.Series:
    return s.astype(str).str.replace(r'\.\d+$', '', regex=True)

def clean_colnames(cols):
    return [str(c).replace('\ufeff','').strip() for c in cols]

def ensembl_match_rate(s: pd.Series, n=500) -> float:
    sample = s.dropna().astype(str).str.strip().head(n)
    if sample.empty: return 0.0
    return float((sample.map(lambda x: bool(ENSEMBL_PAT.match(x))).mean()))

def find_gene_id_column(df: pd.DataFrame) -> str:
    # 尝试优先使用名为 gene_id 的列
    if 'gene_id' in df.columns:
        if ensembl_match_rate(df['gene_id']) > 0.5:
            return 'gene_id'
    # 在所有列中找匹配率最高的列
    rates = {c: ensembl_match_rate(df[c]) for c in df.columns}
    best_col, best_rate = max(rates.items(), key=lambda kv: kv[1])
    print(f"[INFO] Ensembl match rates (top5): {sorted(rates.items(), key=lambda kv: kv[1], reverse=True)[:5]}")
    if best_rate > 0.5:
        return best_col
    # 回退：如果有叫 gene 或 gene_symbol 的列就用它（可能是符号）
    for cand in ['gene', 'Gene', 'gene_symbol', 'GeneSymbol', 'symbol', 'Symbol']:
        if cand in df.columns:
            return cand
    # 最后回退：第一列
    return df.columns[0]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--raw', required=True, help='Roadmap 57epi 解压后的 TSV')
    ap.add_argument('--config', required=True, help='config/config.yaml（读取 eids）')
    ap.add_argument('--promoters', required=True, help='promoters_2kb.hg38.bed')
    ap.add_argument('--keep_zero', action='store_true', help='不删全零基因（调试建议先开）')
    ap.add_argument('--out', default='exp/raw_exp.final.tsv')
    args = ap.parse_args()

    # 1) 读取并清理
    df = pd.read_csv(args.raw, sep='\t', dtype=str)  # 先以字符串读取，避免数字化破坏匹配
    df.columns = clean_colnames(df.columns)
    df = df.apply(lambda col: col.str.strip() if col.dtype == object else col)
    print(f'[INFO] Raw shape: {df.shape}')
    print(f'[INFO] Columns head: {df.columns[:10].tolist()}')

    # 2) 寻找 gene_id 列并标准化
    gene_col = find_gene_id_column(df)
    print(f'[INFO] Selected gene_id column: {gene_col}')
    df.rename(columns={gene_col: 'gene_id'}, inplace=True)
    # 将其余列转为数值（EID 列），无法转换的（非 EID/非数值）会被丢弃
    value_cols = [c for c in df.columns if c != 'gene_id']
    for c in value_cols:
        df[c] = pd.to_numeric(df[c], errors='coerce')

    # 判断 gene_id 是否像 Ensembl，若是则去版本号
    if ensembl_match_rate(df['gene_id']) > 0.5:
        df['gene_id'] = strip_version(df['gene_id'])
        print('[INFO] gene_id looks like Ensembl; version suffix removed.')
    else:
        print('[WARN] gene_id 不像 Ensembl，可能是 gene symbol；与 promoters 对齐时可能为 0。')

    # 3) 去重（均值聚合）
    df_num = df[['gene_id'] + value_cols]
    df = df_num.groupby('gene_id', as_index=False).mean(numeric_only=True)
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

    # 5) 与 promoters 交集（去空白与版本号）
    prom = pd.read_csv(args.promoters, sep='\t', header=None,
                       names=['chrom','start','end','gene_id','score','strand'], dtype={'gene_id': str})
    prom_ids = strip_version(prom['gene_id'].astype(str).str.strip()).unique()
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