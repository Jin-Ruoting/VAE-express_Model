import os, json, math
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import pyBigWig

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RoadmapDataset(Dataset):
    def __init__(self,
                 data_dir,
                 promoters_bed,
                 expression_tsv,
                 genome_sizes,
                 eids,
                 marks_core,
                 marks_extra=None,
                 use_extra=True,
                 promoter_bp=2000,
                 use_enhancers=False,
                 enhancer_map_tsv=None,
                 enhancer_bp=1000,
                 top_k=5,
                 stats_json=None,
                 min_expr_threshold=0.0,
                 zscore_per_eid=False):
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

        # bigWig 句柄
        self.bw = {}
        for eid in self.eids:
            self.bw[eid] = {}
            for mark in self.marks:
                p = self.data_dir / f"hist/{eid}-{mark}.bw"
                if not p.exists():
                    logger.warning(f"缺少 bigWig: {p}")
                    continue
                self.bw[eid][mark] = pyBigWig.open(str(p))

        # 缩放统计（扁平 EID:MARK）
        self.stats = {}
        if stats_json and Path(stats_json).exists():
            with open(stats_json, 'r') as f:
                self.stats = json.load(f)

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

    def _get_vals(self, eid, mark, chrom, start, end):
        handler = self.bw.get(eid, {}).get(mark, None)
        if handler is None:
            return np.zeros(end-start, dtype=np.float32)
        vals = handler.values(chrom, start, end, numpy=True)
        if vals is None:
            return np.zeros(end-start, dtype=np.float32)
        vals = np.nan_to_num(vals, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        return vals

    def _normalize(self, eid, mark, arr):
        # 先用全局 stats（EID:MARK 扁平键），找不到时回退窗口内分位
        s = self.stats.get(f"{eid}:{mark}", None)
        if s is None:
            q1, q99 = np.quantile(arr, [0.01, 0.99])
        else:
            q1, q99 = float(s.get('q1', 0.0)), float(s['q99'])
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
        return torch.from_numpy(X), torch.tensor(y, dtype=torch.float32)

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
        zscore_per_eid=False
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

    def make_loader(d, shuffle, drop_last):
        return DataLoader(d, batch_size=16, shuffle=shuffle, drop_last=drop_last,
                          num_workers=4, pin_memory=True)

    return make_loader(train_set, True, True), make_loader(val_set, False, False), make_loader(test_set, False, False)