import argparse, json, random
from pathlib import Path
import pyBigWig
import numpy as np
import pandas as pd

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--eids', nargs='+', required=True)
    ap.add_argument('--marks', nargs='+', required=True)
    ap.add_argument('--bw_dir', default='hist')
    ap.add_argument('--genome_sizes', required=True)
    ap.add_argument('--samples_per_chrom', type=int, default=200)
    ap.add_argument('--window', type=int, default=1000)
    ap.add_argument('--out', required=True)
    return ap.parse_args()

def load_sizes(path):
    df = pd.read_csv(path, sep='\t', header=None, names=['chrom','size'])
    return list(df.itertuples(index=False, name=None))

def main():
    args = parse_args()
    chrom_sizes = load_sizes(args.genome_sizes)
    stats = {}
    for eid in args.eids:
        stats[eid] = {}
        for mark in args.marks:
            bw_path = Path(args.bw_dir) / f"{eid}-{mark}.bw"
            if not bw_path.exists():
                continue
            bw = pyBigWig.open(str(bw_path))
            vals = []
            for chrom, size in chrom_sizes:
                if chrom not in bw.chroms():
                    continue
                step = max(args.window, size // max(1, args.samples_per_chrom))
                for start in range(0, min(size, step*args.samples_per_chrom), step):
                    end = min(size, start + args.window)
                    v = bw.values(chrom, start, end, numpy=True)
                    if v is None: continue
                    v = v[np.isfinite(v)]
                    if v.size: vals.append(v)
            bw.close()
            if len(vals)==0:
                continue
            vals = np.concatenate(vals)
            q1, q99 = np.quantile(vals, [0.01, 0.99])
            if q99 <= q1:
                q1, q99 = float(np.min(vals)), float(np.max(vals)+1e-6)
            stats[eid][mark] = {'q1': float(q1), 'q99': float(q99)}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"Saved stats to {args.out}")

if __name__ == '__main__':
    main()