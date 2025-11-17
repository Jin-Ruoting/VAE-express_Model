#!/usr/bin/env python3
"""
分析多个细胞系中 7 个组蛋白修饰通道与基因表达 (log2(RPKM+1)) 的分布是否一致。

功能：
1. 读取配置文件，获得 EID、marks、表达矩阵与信号统计路径。
2. 计算各 EID 表达值的 log2(RPKM+1) 分布统计量。
3. 读取 signal_stats.json（若存在），检查每个 EID:MARK 的 q1/q99 范围差异。
4. 给出是否需要归一化的判断与建议。
5. 可选输出 CSV，便于进一步可视化。
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import yaml


def parse_args():
    ap = argparse.ArgumentParser(
        description="检查多细胞系的组蛋白通道与 log2(RPKM+1) 分布差异并给出归一化建议"
    )
    ap.add_argument("-c", "--config", default="config/config.yaml", help="项目配置文件路径")
    ap.add_argument("-e", "--expression", help="表达矩阵路径（覆盖 config.paths.expression）")
    ap.add_argument("-s", "--stats-json", help="histone 信号统计 json 路径（覆盖 config.paths.stats_json）")
    ap.add_argument("--expr-out", help="保存表达统计结果的 CSV")
    ap.add_argument("--hist-out", help="保存 histone 统计结果的 CSV")
    ap.add_argument("--cv-threshold", type=float, default=0.15,
                    help="判定分布不一致的变异系数阈值（默认 0.15）")
    return ap.parse_args()


def _load_config(cfg_path: str) -> Dict:
    cfg_path = Path(cfg_path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg


def _infer_gene_col(df: pd.DataFrame) -> str:
    first_col = df.columns[0]
    if "gene" in first_col.lower():
        return first_col
    return first_col


def _norm_gene_id(series: pd.Series) -> pd.Series:
    return series.astype(str).str.replace(r"\.\d+$", "", regex=True)


def compute_expression_stats(expr_path: str, eids: List[str]) -> pd.DataFrame:
    if not Path(expr_path).exists():
        raise FileNotFoundError(f"Expression file not found: {expr_path}")
    expr = pd.read_csv(expr_path, sep="\t")
    gene_col = _infer_gene_col(expr)
    expr[gene_col] = _norm_gene_id(expr[gene_col])
    expr = expr.set_index(gene_col)

    rows = []
    for eid in eids:
        if eid not in expr.columns:
            continue
        vals = pd.to_numeric(expr[eid], errors="coerce").fillna(0.0).clip(lower=0.0).values
        log_vals = np.log2(vals + 1.0)
        rows.append({
            "eid": eid,
            "n_records": len(log_vals),
            "mean": float(np.mean(log_vals)),
            "std": float(np.std(log_vals, ddof=0)),
            "median": float(np.median(log_vals)),
            "p05": float(np.percentile(log_vals, 5)),
            "p95": float(np.percentile(log_vals, 95)),
        })
    return pd.DataFrame(rows).sort_values("eid").reset_index(drop=True)


def _flatten_stats(raw_stats: Dict, eids: List[str], marks: List[str]) -> Dict[str, Dict]:
    flat = {}
    if not raw_stats:
        return flat
    # already EID:MARK?
    if all(isinstance(k, str) and ":" in k for k in raw_stats.keys()):
        return raw_stats
    # nested structure {eid: {mark: {...}}}
    if all(isinstance(raw_stats[k], dict) for k in raw_stats.keys()):
        for eid in eids:
            for mark in marks:
                if eid in raw_stats and isinstance(raw_stats[eid], dict) and mark in raw_stats[eid]:
                    flat[f"{eid}:{mark}"] = raw_stats[eid][mark]
        return flat
    # only mark-level stats
    for eid in eids:
        for mark in marks:
            if mark in raw_stats and isinstance(raw_stats[mark], dict):
                flat[f"{eid}:{mark}"] = raw_stats[mark]
    return flat


def load_histone_stats(stats_path: str, eids: List[str], marks: List[str]) -> pd.DataFrame:
    if not stats_path or not Path(stats_path).exists():
        return pd.DataFrame()
    with open(stats_path, "r") as f:
        raw = json.load(f)
    flat = _flatten_stats(raw, eids, marks)
    rows = []
    for eid in eids:
        for mark in marks:
            key = f"{eid}:{mark}"
            stats = flat.get(key)
            if not stats:
                continue
            rows.append({
                "eid": eid,
                "mark": mark,
                "q01": float(stats.get("q1", stats.get("q01", 0.0))),
                "q99": float(stats.get("q99", stats.get("q99", 1.0))),
                "median": float(stats.get("median", np.nan)),
            })
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["mark", "eid"]).reset_index(drop=True)
    return df


def evaluate_variation(values: np.ndarray, threshold: float) -> Tuple[bool, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return False, float("nan")
    mean = np.mean(values)
    if abs(mean) < 1e-6:
        return False, float("nan")
    cv = float(np.std(values) / (abs(mean) + 1e-6))
    return cv > threshold, cv


def main():
    args = parse_args()
    cfg = _load_config(args.config)
    eids = cfg.get("eids", [])
    marks = cfg.get("marks", {}).get("core", [])
    if cfg.get("use_extra", False):
        marks = marks + cfg.get("marks", {}).get("extra", [])
    expr_path = args.expression or cfg.get("paths", {}).get("expression")
    stats_path = args.stats_json or cfg.get("paths", {}).get("stats_json")

    print("=" * 80)
    print("表达值 log2(RPKM+1) 统计")
    print("=" * 80)
    expr_df = compute_expression_stats(expr_path, eids)
    print(expr_df.to_string(index=False, float_format=lambda x: f"{x:8.4f}"))

    expr_flag, expr_cv = evaluate_variation(expr_df["median"].values, args.cv_threshold)
    if expr_flag:
        print(f"\n[提示] 不同 EID 的表达分布差异较大 (median CV={expr_cv:.3f} > {args.cv_threshold}).")
        print("      建议：对每个 EID 的表达执行 z-score 或使用批次特定均值/方差进行归一化。")
    else:
        print(f"\n表达分布在阈值 {args.cv_threshold} 内 (median CV={expr_cv:.3f}); 可直接使用 log2(RPKM+1)。")

    if not expr_df.empty and args.expr_out:
        Path(args.expr_out).parent.mkdir(parents=True, exist_ok=True)
        expr_df.to_csv(args.expr_out, index=False)
        print(f"[保存] 表达统计 -> {args.expr_out}")

    print("\n" + "=" * 80)
    print("组蛋白通道 q01/q99 统计 (来自 signal_stats.json)")
    print("=" * 80)
    hist_df = load_histone_stats(stats_path, eids, marks)
    if hist_df.empty:
        print("未找到 histone 统计，无法评估。请先运行 preprocessing/scripts/compute_signal_stats.py。")
    else:
        print(hist_df.to_string(index=False, float_format=lambda x: f"{x:8.4f}"))
        for mark in sorted(hist_df["mark"].unique()):
            sub = hist_df[hist_df["mark"] == mark]
            flag, cv = evaluate_variation(sub["q99"] - sub["q01"], args.cv_threshold)
            if flag:
                print(f"[提示] {mark} 在不同 EID 中动态范围差异较大 (Δq CV={cv:.3f}).")
                print("      建议：继续使用 per-EID 量化剪裁 + min-max 归一化，或引入 learnable scaling。")

    if not hist_df.empty and args.hist_out:
        Path(args.hist_out).parent.mkdir(parents=True, exist_ok=True)
        hist_df.to_csv(args.hist_out, index=False)
        print(f"[保存] histone 统计 -> {args.hist_out}")


if __name__ == "__main__":
    main()

