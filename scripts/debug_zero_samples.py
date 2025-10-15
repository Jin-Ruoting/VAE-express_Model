import os, yaml, pandas as pd, pyBigWig, json

cfg = yaml.safe_load(open('config/config.yaml'))
eids = cfg['eids']
marks = list(cfg['marks']['core']) + (cfg['marks'].get('extra',[]) if cfg.get('use_extra') else [])
paths = cfg['paths']
exp_path = paths['expression']
prom_path = paths['promoters_bed']
stats_path = paths['stats_json']

print('== CONFIG ==')
print('eids:', eids)
print('marks:', marks)
print('paths:', paths)

# 1) 表达矩阵
if not os.path.exists(exp_path):
    raise SystemExit(f'[ERR] 表达文件不存在: {exp_path}')
exp = pd.read_csv(exp_path, sep='\t')
print(f'[EXP] shape={exp.shape} head_cols={exp.columns[:8].tolist()}')
has = [e for e in eids if e in exp.columns]
miss = [e for e in eids if e not in exp.columns]
print(f'[EXP] present_eids={len(has)} missing_eids={len(miss)}')
if miss: print('[EXP] missing list:', miss[:20])

# 2) promoters 交集
if not os.path.exists(prom_path):
    raise SystemExit(f'[ERR] promoters 不存在: {prom_path}')
prom = pd.read_csv(prom_path, sep='\t', header=None,
                   names=['chrom','start','end','gene_id','score','strand'])
# 去版本 ENSG.x
exp['gene_id'] = exp['gene_id'].astype(str).str.replace(r'\.\d+$','', regex=True)
prom['gene_id'] = prom['gene_id'].astype(str).str.replace(r'\.\d+$','', regex=True)
genes = set(exp['gene_id']).intersection(set(prom['gene_id']))
print(f'[GENES] intersection with promoters: {len(genes)}')
if len(genes) == 0:
    print('[HINT] 交集为0：多半是 gene_id 不是 ENSG* 或 promoters 非 hg38。')

# 3) bigWig 完整性（存在且可打开）
bad_bw = []
for e in eids:
    for m in marks:
        p = f'hist/{e}-{m}.bw'
        if not (os.path.exists(p) and os.path.getsize(p) > 1024):
            bad_bw.append((e,m,'missing/small',p)); continue
        try:
            bw = pyBigWig.open(p); _ = bw.chroms(); bw.close()
        except Exception as ex:
            bad_bw.append((e,m,str(ex),p))
print(f'[BW] bad={len(bad_bw)} of {len(eids)*len(marks)}')
if bad_bw: print('[BW] example:', bad_bw[:5])

# 4) stats_json 键完整性
if not os.path.exists(stats_path):
    print(f'[STATS] not found: {stats_path}')
    need_stats_missing = True
else:
    stats = json.load(open(stats_path))
    miss_stats = []
    for e in eids:
        for m in marks:
            key = f'{e}:{m}'
            if key not in stats:
                miss_stats.append(key)
    print(f'[STATS] missing={len(miss_stats)} (expect {len(eids)*len(marks)})')
    if miss_stats: print('[STATS] example missing:', miss_stats[:10])

# 5) 推断可用 eids（即：表达有列 + 7个bigWig都可读 + stats存在）
usable = []
for e in eids:
    ok_exp = e in exp.columns
    ok_bw = all(os.path.exists(f'hist/{e}-{m}.bw') and os.path.getsize(f'hist/{e}-{m}.bw')>1024 for m in marks)
    ok_stats = os.path.exists(stats_path) and all(f'{e}:{m}' in stats for m in marks)
    if ok_exp and ok_bw and ok_stats:
        usable.append(e)
print('[SUGGEST] usable_eids:', usable)

if len(genes) == 0:
    print('[FIX] 重新生成 promoters 或表达表：')
    print('  - promoters: 用 build_promoters_bed.py 从 hg38 GTF 生成')
    print('  - 表达表: 用 clean_roadmap_expression.py 清洗，并确认 gene_id 是 ENSG*')

elif bad_bw:
    print('[FIX] 修复上述 bad bigWig（重下或手动替换），再重算 stats_json。')

elif os.path.exists(stats_path) and len(usable) == 0:
    print('[FIX] 将 config.yaml 的 eids 临时替换为 usable_eids 再试训练。')

print('== DONE ==')