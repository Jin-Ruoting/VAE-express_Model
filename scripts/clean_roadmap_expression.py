import re, argparse, yaml
import pandas as pd
from pathlib import Path

ENSEMBL_PAT = re.compile(r'ENSG\d+(?:\.\d+)?')

def strip_version(s: pd.Series) -> pd.Series:
    return s.astype(str).str.replace(r'\.\d+$', '', regex=True)

def clean_colnames(cols):
    return [str(c).replace('\ufeff','').strip() for c in cols]

def ensembl_match_rate(s: pd.Series, n=500) -> float:
    sample = s.dropna().astype(str).str.strip().head(n)
    if sample.empty: return 0.0
    return float((sample.map(lambda x: bool(re.fullmatch(r'ENSG\d+(?:\.\d+)?', x))).mean()))

def extract_ensembl(s: pd.Series) -> pd.Series:
    # 在任意字符串中提取第一个 ENSG… 子串
    return s.astype(str).str.extract(r'(ENSG\d+(?:\.\d+)?)', expand=True)[0]

def find_gene_id_column(df: pd.DataFrame) -> str:
    # 优先 gene_id 列
    if 'gene_id' in df.columns: return 'gene_id'
    for cand in ['gene', 'Gene', 'gene_symbol', 'GeneSymbol', 'symbol', 'Symbol']:
        if cand in df.columns:
            return cand
    return df.columns[0]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--raw', required=True, help='Roadmap 57epi 解压后的 TSV')
    ap.add_argument('--config', required=True, help='config/config.yaml（读取 eids）')
    ap.add_argument('--promoters', required=True, help='promoters_2kb.hg38.bed')
    ap.add_argument('--keep_zero', action='store_true', help='不删全零基因（调试建议先开）')
    ap.add_argument('--out', default='exp/raw_exp.final.tsv')
    args = ap.parse_args()

    # 1) 读取（以字符串读取，避免数值化破坏内容）
    df = pd.read_csv(args.raw, sep='\t', dtype=str, engine='python')
    df.columns = clean_colnames(df.columns)
    df = df.apply(lambda col: col.str.strip() if col.dtype == object else col)
    print(f'[INFO] Raw shape: {df.shape}')
    print(f'[INFO] Columns head: {df.columns[:10].tolist()}')

    # 2) 定位 gene_id 列并标准化
    gene_col = find_gene_id_column(df)
    df.rename(columns={gene_col: 'gene_id'}, inplace=True)

    # 2.1 从 gene_id 列中“提取”真正的 Ensembl ID
    extr = extract_ensembl(df['gene_id'])
    extr_rate = extr.notna().mean()
    print(f'[INFO] Extractable Ensembl rate in gene_id column: {extr_rate:.3f}')
    if extr_rate > 0.5:
        df['gene_id'] = strip_version(extr)
        print('[INFO] gene_id extracted and version stripped.')
    else:
        # 如果提取率很低，再尝试在所有列中寻找含 ENSG 的列
        for c in df.columns:
            if c == 'gene_id': continue
            tmp = extract_ensembl(df[c])
            rate = tmp.notna().mean()
            if rate > 0.8:
                print(f'[INFO] Found Ensembl IDs in column {c} (rate={rate:.3f}), using it as gene_id.')
                df['gene_id'] = strip_version(tmp)
                break

    # 再测一次匹配率
    print(f'[INFO] Ensembl match rate(final): {ensembl_match_rate(df["gene_id"]):.3f}')

    # 3) 其余列转为数值（EID/数值列）
    value_cols = [c for c in df.columns if c != 'gene_id']
    for c in value_cols:
        df[c] = pd.to_numeric(df[c], errors='coerce')

    # 4) 去重（均值聚合）
    df = df.groupby('gene_id', as_index=False).mean(numeric_only=True)
    print(f'[INFO] After groupby(unique gene_id): {df.shape}')

    # 5) 仅保留 config.eids
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

    # 6) 与 promoters 交集（对双方做提取+去版本号）
    prom = pd.read_csv(args.promoters, sep='\t', header=None,
                       names=['chrom','start','end','gene_id','score','strand'], dtype={'gene_id': str})
    prom_ids = strip_version(extract_ensembl(prom['gene_id']).astype(str)).dropna().unique()
    before_inter = df.shape[0]
    df = df[df['gene_id'].isin(set(prom_ids))]
    print(f'[INFO] Intersect with promoters: {df.shape[0]}/{before_inter}')

    # 7) 删全零
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