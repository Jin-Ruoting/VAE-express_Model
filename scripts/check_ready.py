import os, glob, yaml, pandas as pd

cfg = yaml.safe_load(open('config/config.yaml'))
eids = cfg['eids']
marks = list(cfg['marks']['core']) + (cfg['marks'].get('extra',[]) if cfg.get('use_extra') else [])
bw_dir = 'hist'
exp_path = cfg['paths']['expression']
prom_path = cfg['paths']['promoters_bed']
stats_path = cfg['paths']['stats_json']

# 1) bigWig 覆盖度
total = 0; ok = 0; missing=[]
for e in eids:
    for m in marks:
        total += 1
        p = os.path.join(bw_dir, f'{e}-{m}.bw')
        if os.path.exists(p) and os.path.getsize(p) > 1024:
            ok += 1
        else:
            missing.append(p)
print(f'[BW] ready {ok}/{total}')
if missing: print('[BW] missing examples:', missing[:10])

# 2) 表达矩阵列
if os.path.exists(exp_path):
    df = pd.read_csv(exp_path, sep='\t')
    cols = set(df.columns)
    lack = [e for e in eids if e not in cols]
    print(f'[EXP] rows={df.shape[0]} cols={df.shape[1]} missing_eids={len(lack)}')
    if lack: print('[EXP] missing eids:', lack)
else:
    print(f'[EXP] not found: {exp_path}')

# 3) promoters 和 stats
print('[PROM] exists:', os.path.exists(prom_path))
print('[STATS] exists:', os.path.exists(stats_path), stats_path)