import os, glob, yaml, pyBigWig, numpy as np, pandas as pd, argparse, sys

def overview(eids, marks, bw_dir):
    paths=[os.path.join(bw_dir, f'{e}-{m}.bw') for e in eids for m in marks]
    missing=[p for p in paths if not os.path.exists(p)]
    small=[p for p in paths if os.path.exists(p) and os.path.getsize(p)<1024]
    print(f'[OVERVIEW] expected={len(paths)} present={len(paths)-len(missing)} missing={len(missing)} small={len(small)}')
    if missing: print('  missing example:', missing[:5])
    if small: print('  small example:', small[:5])
    return paths, missing, small

def head_check(paths, sample_n=10):
    bad_open=[]; no_chr=[]
    want=set(['chr1','chr2','chrX','chrY','chrM'])
    for p in paths[:sample_n]:
        try:
            bw=pyBigWig.open(p)
            ch=set(bw.chroms().keys())
            bw.close()
            if not (ch & want):
                no_chr.append(p)
        except Exception as e:
            bad_open.append((p, str(e)))
    print(f'[HEADER] open_fail={len(bad_open)} no_chr_like={len(no_chr)}')
    if bad_open: print('  open_fail example:', bad_open[:2])
    if no_chr: print('  no_chr example:', no_chr[:2])

def signal_check(prom_bed, paths, sample_n=200, file_n=10):
    bed = pd.read_csv(prom_bed, sep='\t', header=None,
                      names=['chrom','start','end','gene_id','score','strand'])
    samp = bed.sample(n=min(sample_n, len(bed)), random_state=1)
    ok=0; warn=0; err=0
    for p in paths[:file_n]:
        try:
            bw=pyBigWig.open(p)
            vals=[]
            for _,r in samp.iterrows():
                try:
                    v=np.array(bw.values(str(r.chrom), int(r.start), int(r.end)))
                    v=v[np.isfinite(v)]
                    if v.size: vals.append(float(np.nanmean(v)))
                except Exception:
                    pass
            bw.close()
            if not vals or np.allclose(vals, 0.0):
                print('[WARN] no/zero signal:', os.path.basename(p))
                warn+=1
            else:
                arr=np.array(vals)
                print('[OK] ', os.path.basename(p), 'n=',len(arr),'mean=',round(arr.mean(),3),'p95=',round(np.percentile(arr,95),3))
                ok+=1
        except Exception as e:
            print('[ERR] open failed:', os.path.basename(p), e)
            err+=1
    print(f'[SIGNAL] ok={ok} warn={warn} err={err}')

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--config', default='config/config.yaml')
    ap.add_argument('--bw_dir', default='hist')
    ap.add_argument('--promoters', default='preprocessing/annotations/promoters_2kb.hg38.bed')
    args=ap.parse_args()
    cfg=yaml.safe_load(open(args.config))
    marks=list(cfg['marks']['core']) + (cfg['marks'].get('extra',[]) if cfg.get('use_extra') else [])
    eids=cfg['eids']
    paths, _, _ = overview(eids, marks, args.bw_dir)
    head_check(sorted([p for p in paths if os.path.exists(p)]))
    signal_check(args.promoters, sorted([p for p in paths if os.path.exists(p)]))
    print('[DONE] validation finished.')

if __name__ == '__main__':
    main()