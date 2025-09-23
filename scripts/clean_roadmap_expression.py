import re, argparse, yaml, io
import pandas as pd
from pathlib import Path

ENSEMBL_PAT = re.compile(r'^ENSG\d+(?:\.\d+)?$')

def strip_version(s: pd.Series) -> pd.Series:
    return s.astype(str).str.replace(r'\.\d+$', '', regex=True)

def read_with_fixed_header(path: str):
    # 把“被换行的表头”拼回一行；遇到第一个以 ENSG 开头的行视为数据开始
    header_tokens = []
    data_lines = []
    with open(path, 'r', encoding='utf-8', errors='replace') as f:
        for line in f:
            line = line.rstrip('\n')
            if not line:
                continue
            first = line.split()[0] if line.split() else ''
            if ENSEMBL_PAT.match(first):  # 第一行数据
                data_lines.append(line)
                break
            header_tokens.extend(line.split())  # 任意空白切分累积表头
        for line in f:
            if line.strip():
                data_lines.append(line.rstrip('\n'))
    if not header_tokens:
        raise RuntimeError('无法解析表头：文件开头未找到列名。')
    cols = [c.strip().replace('\ufeff','') for c in header_tokens]
    buf = io.StringIO('\n'.join(data_lines))
    # 任意空白分隔，兼容tab/空格混排
    df = pd.read_csv(buf, sep=r'\s+', engine='python', header=None, names=cols, dtype=str)
    for c in df.columns:
        df[c] = df[c].astype(str).strip()
    return df

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--raw', required=True, help='原始表达TSV（允许表头断行）')
    ap.add_argument('--config', required=True, help='config/config.yaml（读取 eids）')
    ap.add_argument('--promoters', required=True, help='promoters_2kb.hg38.bed')
    ap.add_argument('--keep_zero', action='store_true', help='不删全零基因')
    ap.add_argument('--out', default='exp/raw_exp.final.tsv')
    args = ap.parse_args()

    # 1) 读取并修复表头
    df = read_with_fixed_header(args.raw)
    print(f'[INFO] Raw(fixed) shape: {df.shape}')

    # 2) gene_id 归一化（抽取 ENSG，去版本）
    if 'gene_id' not in df.columns:
        raise RuntimeError('修复后的表头无 gene_id 列。')

    # 抽取 ENSG 并去版本号
    df['gene_id'] = df['gene_id'].str.extract(r'(ENSG\d+(?:\.\d+)?)', expand=True)[0]
    if df['gene_id'].isna().all():
        raise RuntimeError('gene_id 列未能抽取出 ENSG*。')
    df['gene_id'] = strip_version(df['gene_id'])

    # 3) 其它列转为数值
    value_cols = [c for c in df.columns if c != 'gene_id']
    for c in value_cols:
        df[c] = pd.to_numeric(df[c], errors='coerce')

    # 4) 按 gene_id 去重聚合
    df = df.groupby('gene_id', as_index=False).mean(numeric_only=True)
    print(f'[INFO] After groupby(unique gene_id): {df.shape}')

    # 5) 仅保留 config.eids
    cfg = yaml.safe_load(open(args.config))
    eids = cfg['eids']
    present = [e for e in eids if e in df.columns]
    missing = [e for e in eids if e not in df.columns]
    if not present:
        raise RuntimeError(f'表达矩阵中未找到任何配置的EID列，missing(前10)={missing[:10]}')
    if missing:
        print(f'[WARN] 缺失 EID 列（忽略）: {missing}')
    df = df[['gene_id'] + present]
    print(f'[INFO] After EID subset: {df.shape}')

    # 6) 与 promoters 交集（去版本）
    prom = pd.read_csv(args.promoters, sep='\t', header=None,
                       names=['chrom','start','end','gene_id','score','strand'], dtype={'gene_id': str})
    prom_ids = strip_version(prom['gene_id'].astype(str).str.strip()).unique()
    before = df.shape[0]
    df = df[df['gene_id'].isin(set(prom_ids))]
    print(f'[INFO] Intersect with promoters: {df.shape[0]}/{before}')

    # 7) 删全零（可选）
    if not args.keep_zero:
        vals = df.iloc[:, 1:]
        zero_only = (vals.sum(axis=1) == 0)
        removed = int(zero_only.sum())
        df = df[~zero_only]
        print(f'[INFO] Removed zero-only genes: {removed}')

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, sep='\t', index=False)
    print(f'[OK] Saved -> {args.out} | genes={df.shape[0]} eids={df.shape[1]-1}')

if __name__ == '__main__':
    main()