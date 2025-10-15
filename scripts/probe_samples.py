import yaml, json, pandas as pd, numpy as np, pyBigWig, random, os

cfg=yaml.safe_load(open('config/config.yaml'))
prom=cfg['paths']['promoters_bed']; marks=cfg['marks']['core']; eids=cfg['eids']
stats=json.load(open(cfg['paths']['stats_json']))

bed=pd.read_csv(prom,sep='\t',header=None,names=['chrom','start','end','gene_id','score','strand'])
genes = bed['gene_id'].astype(str).str.replace(r'\.\d+$','', regex=True)
bed['gene_id']=genes
exp=pd.read_csv(cfg['paths']['expression'], sep='\t')
exp['gene_id']=exp['gene_id'].astype(str).str.replace(r'\.\d+$','', regex=True)

bed = bed[bed['gene_id'].isin(set(exp['gene_id']))].copy()
samp = bed.sample(n=min(200, len(bed)), random_state=1)

def norm_vals(v, q1, q99):
    v = np.asarray(v, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0: return v
    # 简单分位裁剪后再缩放到[0,1]
    lo = q1 if q1 is not None else 0.0
    hi = q99 if q99 is not None and q99>0 else np.nanpercentile(v, 99)
    v = np.clip(v, lo, hi)
    if hi>lo:
        v = (v - lo) / (hi - lo)
    return v

report=[]
for eid in eids:
  for m in marks:
    bwp=f"hist/{eid}-{m}.bw"
    ok=True; reason=''
    try:
      bw=pyBigWig.open(bwp); _=bw.chroms()
    except Exception as e:
      ok=False; reason=f"open_fail:{e}"
      report.append((eid,m,ok,reason,0,0)); continue
    nonzero=0; windows=0
    st = stats.get(f"{eid}:{m}", {})
    q1=st.get('q1', 0.0); q99=st.get('q99', None)
    for _,r in samp.iterrows():
      v = np.array(bw.values(str(r.chrom), int(r.start), int(r.end)))
      v = v[np.isfinite(v)]
      if v.size==0: continue
      windows += 1
      vn = norm_vals(v, q1, q99)
      if vn.size and np.any(vn>0): nonzero += 1
    bw.close()
    if windows==0:
      ok=False; reason='all_nan_windows'
    elif nonzero==0:
      ok=False; reason='all_zero_after_norm'
    report.append((eid,m,ok,reason,windows,nonzero))

bad=[r for r in report if not r[2]]
print('[PROBE] total pairs:', len(report), 'bad:', len(bad))
print('[PROBE] bad examples:', bad[:8])