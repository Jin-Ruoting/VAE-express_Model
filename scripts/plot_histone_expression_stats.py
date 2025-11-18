#!/usr/bin/env python3
"""
可视化 11 个细胞系的表达与组蛋白修饰统计：
1. log2(RPKM+1) 小提琴图（按 EID）
2. 每个组蛋白标记在 11 个细胞系中的箱型图
3. 7×11 Pearson 相关热图（表达 vs. 对应 mark 的平均信号）

依赖：seaborn、matplotlib、scipy
"""

import argparse
import os
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from scipy.stats import pearsonr

# 将项目根目录加入 sys.path 以便复用 RoadmapDataset
import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from data.roadmap_dataset import RoadmapDataset  
import yaml  


def parse_args():
    ap = argparse.ArgumentParser(description="Plot expression and histone mark statistics.")  # 绘制统计图
    ap.add_argument("-c", "--config", default="config/config.yaml", help="Config file.")  # 配置文件
    ap.add_argument("-o", "--output-dir", default="results/plots", help="Directory for figures.")  # 输出目录
    ap.add_argument("--expr-plot", default="expr_violin.png", help="Filename for expression violin plot.")  # 小提琴图文件名
    ap.add_argument("--mark-plot", default="marks_box.png", help="Filename for histone box plots.")  # 箱型图文件名
    ap.add_argument("--heatmap", default="expr_mark_heatmap.png", help="Filename for correlation heatmap.")  # 热图文件名
    ap.add_argument("--max-samples", type=int, default=20000,
                    help="Maximum number of gene×EID samples (default 20k).")  # 抽样上限
    ap.add_argument("--seed", type=int, default=42, help="Random seed for subsampling.")  # 随机种子
    ap.add_argument("--batch-size", type=int, default=64, help="Batch size when iterating dataset.")  # 遍历批大小
    return ap.parse_args()


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def build_dataset(cfg):
    marks = cfg["marks"]["core"] + (cfg["marks"].get("extra", []) if cfg.get("use_extra") else [])
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
        stats_json=cfg["paths"]["stats_json"],
        min_expr_threshold=cfg.get("expression", {}).get("min_threshold", 0.0),
        enhancer_bp=cfg["sequence"]["enhancer_bp"],
        top_k=cfg["sequence"]["top_k"],
        zscore_per_eid=False,
        cfg=cfg,
    )
    return ds, marks


def subsample_indices(n, max_samples, seed):
    idx = list(range(n))
    if max_samples is None or max_samples <= 0 or max_samples >= n:
        return idx
    random.seed(seed)
    return random.sample(idx, max_samples)


def collect_statistics(ds, marks, indices):
    expr_records = []
    mark_records = []
    mark_values = {(mark, eid): [] for eid in ds.eids for mark in marks}
    expr_values = {eid: [] for eid in ds.eids}

    for idx in indices:
        (gene, eid, *_rest) = ds.samples[idx]
        X, y = ds[idx]
        expr = float(y.reshape(-1)[0])
        expr_records.append({"eid": eid, "expr": expr})
        expr_values[eid].append(expr)

        arr = X.numpy()
        mark_means = arr.mean(axis=1)  # [num_marks]
        for m_idx, mark in enumerate(marks):
            val = float(mark_means[m_idx])
            mark_records.append({"eid": eid, "mark": mark, "value": val})
            mark_values[(mark, eid)].append(val)

    expr_df = pd.DataFrame(expr_records)
    mark_df = pd.DataFrame(mark_records)
    return expr_df, mark_df, expr_values, mark_values


def plot_expression_violin(expr_df, eids, out_path):
    plt.figure(figsize=(max(8, 0.6 * len(eids)), 5))
    sns.violinplot(data=expr_df, x="eid", y="expr", order=eids, cut=0, scale="width", inner="quartile")
    plt.ylabel("log2(RPKM + 1)")
    plt.xlabel("EID")
    plt.title("Gene expression distribution (log2(RPKM+1))")  # 小提琴图标题
    plt.xticks(rotation=40, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_marks_box(mark_df, eids, marks, out_path):
    g = sns.catplot(
        data=mark_df,
        kind="box",
        x="eid",
        y="value",
        col="mark",
        col_wrap=2,
        order=eids,
        sharey=False,
        height=4,
        aspect=1.2,
    )
    g.set_xticklabels(rotation=40, ha="right")
    g.set_axis_labels("EID", "Normalized signal")
    g.fig.subplots_adjust(top=0.92)
    g.fig.suptitle("Histone mark distributions (per-EID min-max normalized)")  # 组蛋白箱型图标题
    g.savefig(out_path, dpi=300)
    plt.close(g.fig)


def plot_heatmap(expr_values, mark_values, eids, marks, out_path):
    heat = np.full((len(marks), len(eids)), np.nan, dtype=float)
    for i, mark in enumerate(marks):
        for j, eid in enumerate(eids):
            expr = np.array(expr_values[eid], dtype=float)
            mark_arr = np.array(mark_values[(mark, eid)], dtype=float)
            if expr.size >= 2 and mark_arr.size == expr.size:
                r, _ = pearsonr(mark_arr, expr)
                heat[i, j] = r
    plt.figure(figsize=(1.2 * len(eids), 0.8 * len(marks) + 2))
    sns.heatmap(
        heat,
        annot=True,
        fmt=".2f",
        xticklabels=eids,
        yticklabels=marks,
        vmin=-1,
        vmax=1,
        cmap="coolwarm",
        linewidths=0.5,
        cbar_kws={"label": "Pearson R"},
    )
    plt.title("Pearson correlation between histone marks and expression")  # 相关性热图标题
    plt.xlabel("EID")
    plt.ylabel("Histone mark")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    ds, marks = build_dataset(cfg)
    indices = subsample_indices(len(ds), args.max_samples, args.seed)
    print(f"[INFO] Dataset samples={len(ds)}, using {len(indices)} for stats")  # 数据量提示

    expr_df, mark_df, expr_vals, mark_vals = collect_statistics(ds, marks, indices)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    expr_path = out_dir / args.expr_plot
    mark_path = out_dir / args.mark_plot
    heat_path = out_dir / args.heatmap

    plot_expression_violin(expr_df, ds.eids, expr_path)
    plot_marks_box(mark_df, ds.eids, marks, mark_path)
    plot_heatmap(expr_vals, mark_vals, ds.eids, marks, heat_path)

    print(f"[OK] Expression violin plot -> {expr_path}")  # 小提琴图
    print(f"[OK] Histone box plots -> {mark_path}")  # 箱型图
    print(f"[OK] Correlation heatmap -> {heat_path}")  # 相关性热图


if __name__ == "__main__":
    main()

