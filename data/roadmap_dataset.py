import os, json
import numpy as np
import pyBigWig

from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import pandas as pd
import torch

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RoadmapDataset(Dataset):
    def __init__(self, 
                 data_dir, promoters_bed, expression_tsv, genome_sizes,
                 eids, marks_core, marks_extra=None, use_extra=True,
                 promoter_bp=2000, use_enhancers=False, enhancer_map_tsv=None,
                 enhancer_bp=1000, top_k=5, stats_json=None,
                 min_expr_threshold=0.0, zscore_per_eid=False):
        """
        在线从 bigWig 切片，返回 X: [C, L], y: 标量(log1p或zscore后)
        """
        self.data_dir = Path(data_dir)
        self.promoter_bp = promoter_bp
        self.use_enhancers = use_enhancers
        self.enhancer_bp = enhancer_bp
        self.top_k = top_k
        self.eids = list(eids)
        self.marks_core = list(marks_core)
        self.marks_extra = list(marks_extra or [])
        self.use_extra = use_extra
        self.marks = self.marks_core + (self.marks_extra if use_extra else [])
        self.genome_sizes = genome_sizes
        self.zscore_per_eid = zscore_per_eid

        # 载入表达矩阵
        self.exp = pd.read_csv(expression_tsv, sep='\t')
        # 统一 gene_id 列名
        if 'gene' in self.exp.columns[0].lower() or 'gene_id' in self.exp.columns[0].lower():
            self.gene_col = self.exp.columns[0]
        else:
            self.exp.rename(columns={self.exp.columns[0]:'gene_id'}, inplace=True)
            self.gene_col = 'gene_id'
        # 数值化表达列，NaN 用0填充，避免后续 drop 掉全部
        for eid in eids:
            if eid in self.exp.columns:
                self.exp[eid] = pd.to_numeric(self.exp[eid], errors='coerce').fillna(0.0).clip(lower=0)
        # 去版本号，保证与 promoters 对齐
        self.exp[self.gene_col] = self.exp[self.gene_col].astype(str).str.replace(r'\.\d+$','', regex=True)
        keep_cols = [self.gene_col] + [eid for eid in self.eids if eid in self.exp.columns]
        self.exp = self.exp[keep_cols]
        self.exp.set_index(self.gene_col, inplace=True)

        # 可选zscore统计
        self.exp_mean = {}
        self.exp_std = {}
        if self.zscore_per_eid:
            for eid in self.eids:
                v = np.log1p(self.exp[eid].values)
                self.exp_mean[eid] = float(np.mean(v))
                self.exp_std[eid] = float(np.std(v) + 1e-6)

        # 载入promoter窗口并去版本号
        prom_cols = ['chrom','start','end','gene_id','score','strand']
        promoters = pd.read_csv(promoters_bed, sep='\t', header=None, names=prom_cols)
        promoters['gene_id'] = promoters['gene_id'].astype(str).str.replace(r'\.\d+$','', regex=True)
        promoters = promoters[promoters['gene_id'].isin(self.exp.index)]

        # 载入增强子映射（可选，Promoter-only时不需要）
        self.enhancers = None
        if self.use_enhancers and enhancer_map_tsv:
            enh = pd.read_csv(enhancer_map_tsv, sep='\t')
            # 需要列: gene_id, eid, enh_chr, enh_start, enh_end, score
            req = {'gene_id','eid','enh_chr','enh_start','enh_end','score'}
            missing = req - set(enh.columns)
            if missing:
                raise ValueError(f"enhancer_map缺少列: {missing}")
            self.enhancers = enh

        # 载入缩放统计（支持扁平与嵌套，最终存为 EID:MARK）
        self.stats = {}
        if stats_json and Path(stats_json).exists():
            try:
                raw = json.load(open(stats_json))
                keys = list(raw.keys())
                flat = {}
                if any(isinstance(k, str) and ':' in k for k in keys):
                    # 已是扁平
                    flat = raw
                elif all(isinstance(raw[k], dict) for k in keys):
                    # 形如 EID -> {MARK: {q1,q99}}
                    for e in eids:
                        sub = raw.get(e, {})
                        if isinstance(sub, dict):
                            for m in (list(marks_core) + (list(marks_extra) if use_extra and marks_extra else [])):
                                if m in sub:
                                    flat[f"{e}:{m}"] = sub[m]
                else:
                    # 其他形态（如只有 MARK 级别），复制到每个 EID
                    mark_keys = set(list(marks_core) + (list(marks_extra) if use_extra and marks_extra else []))
                    if mark_keys.issubset(set(keys)):
                        for e in eids:
                            for m in mark_keys:
                                flat[f"{e}:{m}"] = raw[m]
                self.stats = flat
            except Exception as ex:
                logger.warning(f"Failed to load/parse stats_json {stats_json}: {ex}; using empty stats.")
                self.stats = {}
        # 调试信息
        if os.getenv('DATASET_DEBUG','0') == '1':
            colon = sum(1 for k in self.stats if isinstance(k,str) and ':' in k)
            print(f"[DATASET] stats_path={stats_json} keys={len(self.stats)} colon_keys={colon}")

        # 载入染色体长度（用于裁剪窗口）
        self.chrom_sizes = {}
        with open(genome_sizes) as f:
            for line in f:
                if not line.strip(): continue
                chrom, size = line.split()[:2]
                self.chrom_sizes[chrom] = int(size)

        # 构建 bigWig 路径映射，不在 __init__ 打开文件（避免跨进程）
        self.bw_paths = {}
        self.marks = list(marks_core) + (list(marks_extra) if use_extra and marks_extra else [])
        for eid in eids:
            for mark in self.marks:
                path = Path(data_dir) / "hist" / f"{eid}-{mark}.bw"
                self.bw_paths[(eid, mark)] = str(path)

        # 每进程懒加载句柄缓存
        self._bw_cache = {}
        self._pid = os.getpid()

        # 构建样本索引（(gene_id, eid, chrom, start, end, strand, enh_list)）
        self.samples = []
        half = promoter_bp // 2

        # 调试计数
        debug = os.getenv('DATASET_DEBUG','0') == '1'
        cnt = {
            'genes_total': 0,
            'after_promoters': 0,
            'after_expr': 0,
        }

        for _, r in promoters.iterrows():
            cnt['genes_total'] += 1
            gene = r['gene_id']
            chrom = r['chrom']
            center = (r['start'] + r['end'])//2
            p_start = max(0, int(center - half))
            p_end = int(center + half)
            strand = r['strand']
            cnt['after_promoters'] += 1
            for eid in self.eids:
                if eid not in self.exp.columns:
                    continue
                y_raw = float(self.exp.at[gene, eid]) if gene in self.exp.index else float('nan')
                if pd.isna(y_raw):
                    continue
                # 按需过滤低表达（默认不过滤）
                if (min_expr_threshold is not None) and (y_raw <= float(min_expr_threshold)):
                    # 如果希望过滤，改为 continue
                    # continue
                    pass
                enh_list = []
                if self.use_enhancers and self.enhancers is not None:
                    sub = self.enhancers[(self.enhancers['gene_id']==gene) & (self.enhancers['eid']==eid)]
                    sub = sub.sort_values('score', ascending=False).head(self.top_k)
                    for _, er in sub.iterrows():
                        enh_list.append((er['enh_chr'], int(er['enh_start']), int(er['enh_end'])))
                self.samples.append((gene, eid, chrom, p_start, p_end, strand, enh_list))
                cnt['after_expr'] += 1

        self.seq_len = promoter_bp + (len(enh_list)*enhancer_bp if self.use_enhancers and len(self.samples)>0 else 0)
        self.input_channels = len(self.marks)
        logger.info(f"Dataset built: samples={len(self.samples)}, channels={self.input_channels}, L={self.seq_len}")
        if debug:
            print('[DATASET DEBUG]', json.dumps(cnt))

    def __len__(self):
        return len(self.samples)

    def _reset_cache_if_forked(self):
        pid = os.getpid()
        if pid != self._pid:
            # 子进程：清空并重置缓存（避免复用主进程句柄）
            self._bw_cache = {}
            self._pid = pid

    def _get_bw(self, eid, mark):
        self._reset_cache_if_forked()
        key = (eid, mark)
        h = self._bw_cache.get(key)
        if h is None:
            path = self.bw_paths.get(key)
            if path and os.path.exists(path):
                try:
                    h = pyBigWig.open(path)
                except Exception:
                    h = None
            self._bw_cache[key] = h
        return h

    def _safe_window(self, chrom, start, end):
        # 将窗口裁剪到合法范围，无法裁剪时返回 None
        if chrom not in self.chrom_sizes:
            return None, None, None
        size = self.chrom_sizes[chrom]
        s = max(0, int(start))
        e = min(int(end), size)
        if e <= s:
            return None, None, None
        return chrom, s, e

    def _get_vals(self, eid, mark, chrom, start, end):
        # 返回长度为 (end-start) 的 np.float32 数组；异常时返回全零
        L = int(end - start)
        if L <= 0:
            return np.zeros((0,), dtype=np.float32)
        # 窗口裁剪
        adj = self._safe_window(chrom, start, end)
        if adj[0] is None:
            return np.zeros((L,), dtype=np.float32)
        chrom2, s2, e2 = adj
        h = self._get_bw(eid, mark)
        if h is None:
            return np.zeros((L,), dtype=np.float32)
        try:
            vals = np.array(h.values(chrom2, s2, e2, numpy=True), dtype=np.float32)
            # 对齐长度：前后用0填充到原始 L
            pre = s2 - int(start)
            post = int(end) - e2
            if pre > 0:
                vals = np.pad(vals, (pre, 0), mode='constant', constant_values=0.0)
            if post > 0:
                vals = np.pad(vals, (0, post), mode='constant', constant_values=0.0)
            if vals.size != L:
                # 兜底截断/填充
                if vals.size > L:
                    vals = vals[:L]
                else:
                    vals = np.pad(vals, (0, L - vals.size), mode='constant', constant_values=0.0)
            return vals
        except Exception:
            # pyBigWig 取值失败：返回全零，避免崩溃
            return np.zeros((L,), dtype=np.float32)

    def _normalize(self, eid, mark, arr):
        # 使用扁平键 EID:MARK
        s = self.stats.get(f"{eid}:{mark}", None)
        if s is None:
            q1, q99 = np.quantile(arr, [0.01, 0.99])
        else:
            q1 = float(s.get('q1', 0.0))
            q99 = float(s.get('q99', 0.0))
        if not np.isfinite(q99) or q99 <= q1:
            q1, q99 = float(np.min(arr)), float(np.max(arr) + 1e-6)
        arr = np.clip(arr, q1, q99)
        arr = (arr - q1) / (q99 - q1 + 1e-6)
        return arr.astype(np.float32)

    def __getitem__(self, idx):
        gene, eid, chrom, p_start, p_end, strand, enh_list = self.samples[idx]
        segments = []
        # Promoter段
        for mark in self.marks:
            v = self._get_vals(eid, mark, chrom, p_start, p_end)
            v = self._normalize(eid, mark, v)
            # 统一正链方向（负链则反向）
            if strand == '-':
                v = v[::-1].copy()
            segments.append(v)
        # Enhancer段（可选）
        if self.use_enhancers and len(enh_list)>0:
            half = self.enhancer_bp // 2
            for (echr, estart, eend) in enh_list:
                center = (estart + eend)//2
                s = max(0, center - half)
                e = center + half
                for mark in self.marks:
                    v = self._get_vals(eid, mark, echr, s, e)
                    v = self._normalize(eid, mark, v)
                    segments.append(v)

        # 组装 X: [C, L]（C=marks + K*marks(若用enhancers)）
        # 若不足K，前面构建sample时已限制K；若某bw缺失则是0通道
        X = np.stack(segments, axis=0)  # [C, L]
        # 标签 y
        y_val = float(self.exp.at[gene, eid])
        y = np.log1p(y_val)
        if self.zscore_per_eid:
            mu, sd = self.exp_mean.get(eid, 0.0), self.exp_std.get(eid, 1.0)
            y = (y - mu) / sd

        # 假设最终拼好的通道数组为 X，形状 [C, L]；表达为 y（标量或向量）
        # 在返回前做数值清洗
        X = np.asarray(X, dtype=np.float32)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        if isinstance(y, np.ndarray):
            y = np.nan_to_num(y.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        else:
            try:
                y = float(y)
            except Exception:
                y = 0.0
            if not np.isfinite(y):
                y = 0.0

        return torch.from_numpy(X), torch.as_tensor(y, dtype=torch.float32)

def create_dataloaders(config):
    """
    使用 config/config.yaml 中的设定创建 DataLoader
    """
    import yaml
    with open(config, 'r') as f:
        cfg = yaml.safe_load(f)

    marks = cfg["marks"]["core"] + (cfg["marks"]["extra"] if cfg.get("use_extra", False) else [])
    ds = RoadmapDataset(
        data_dir=".",
        promoters_bed=cfg["paths"]["promoters_bed"],
        expression_tsv=cfg["paths"]["expression"],
        genome_sizes=cfg["paths"]["genome_sizes"],
        eids=cfg["eids"],
        marks_core=cfg["marks"]["core"],
        marks_extra=cfg["marks"].get("extra", []),
        use_extra=cfg.get("use_extra", False),
        promoter_bp=cfg["sequence"]["promoter_bp"],
        use_enhancers=False,
        enhancer_map_tsv=cfg["paths"].get("enhancer_map", None),
        enhancer_bp=cfg["sequence"]["enhancer_bp"],
        top_k=cfg["sequence"]["top_k"],
        stats_json=cfg["paths"]["stats_json"],
        min_expr_threshold=0.0,
        zscore_per_eid=True
    )
    N = len(ds)
    if N == 0:
        raise RuntimeError("RoadmapDataset 构建后样本数为0，请检查 gene_id 对齐、stats 键格式、以及是否误保留了重复的 _build 定义。")
    train_ratio, val_ratio = 0.7, 0.2
    n_train = int(N * train_ratio)
    n_val = int(N * val_ratio)
    n_test = N - n_train - n_val
    g = torch.Generator().manual_seed(42)
    train_set, val_set, test_set = torch.utils.data.random_split(ds, [n_train, n_val, n_test], generator=g)

    num_workers = int(os.getenv('NUM_WORKERS', '0'))
    pin_mem = torch.cuda.is_available()

    def make_loader(d, shuffle, drop_last):
        return DataLoader(d, batch_size=16, shuffle=shuffle, drop_last=drop_last,
                          num_workers=num_workers, pin_memory=pin_mem)

    return make_loader(train_set, True, True), make_loader(val_set, False, False), make_loader(test_set, False, False)