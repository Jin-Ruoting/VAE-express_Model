#!/usr/bin/env python3
"""
全面评估最佳模型
"""
import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import torch
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from models.vae import VAE
from data.roadmap_dataset import create_dataloaders
import yaml

def evaluate_model():
    """评估模型在测试集上的性能"""
    
    print("="*60)
    print("  VAE 模型全面评估")
    print("="*60)
    
    # 1. 加载配置
    cfg = yaml.safe_load(open('config/config.yaml'))
    
    # 2. 创建数据加载器
    print("\n[1/5] 加载数据...")
    _, _, test_loader = create_dataloaders('config/config.yaml')
    print(f"✓ 测试集批次数: {len(test_loader)}")
    
    # 3. 加载模型
    print("\n[2/5] 加载模型...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    marks = cfg['marks']['core'] + (cfg['marks']['extra'] if cfg.get('use_extra') else [])
    seq_len = cfg['sequence']['promoter_bp']
    
    model = VAE(
        input_channels=len(marks),
        latent_dim=64,
        sequence_length=seq_len
    )
    
    # 加载最佳模型权重
    model_path = 'results/models/vae_promoter_only_best.pt'
    if not Path(model_path).exists():
        print(f"✗ 模型文件不存在: {model_path}")
        return
    
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    print(f"✓ 模型已加载: {model_path}")
    
    # 4. 运行预测
    print("\n[3/5] 运行预测...")
    
    all_true = []
    all_pred = []
    all_mu = []
    all_logvar = []
    
    with torch.no_grad():
        for x, y in tqdm(test_loader, desc="预测进度"):
            x = x.to(device)
            y = y.to(device)
            
            x_hat, mu, logvar, expr_pred = model(x)
            
            all_true.append(y.cpu().numpy())
            all_pred.append(expr_pred.cpu().numpy())
            all_mu.append(mu.cpu().numpy())
            all_logvar.append(logvar.cpu().numpy())
    
    # 合并结果
    y_true = np.concatenate(all_true, axis=0).reshape(-1)
    y_pred = np.concatenate(all_pred, axis=0).reshape(-1)
    mu = np.concatenate(all_mu, axis=0)
    logvar = np.concatenate(all_logvar, axis=0)
    
    print(f"✓ 完成预测: {len(y_true)} 个样本")
    
    # 5. 计算指标
    print("\n[4/5] 计算评估指标...")
    
    # 过滤无效值
    valid = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true_valid = y_true[valid]
    y_pred_valid = y_pred[valid]
    
    metrics = {
        'pearson_r': float(pearsonr(y_true_valid, y_pred_valid)[0]),
        'spearman_r': float(spearmanr(y_true_valid, y_pred_valid)[0]),
        'r2_score': float(r2_score(y_true_valid, y_pred_valid)),
        'mse': float(mean_squared_error(y_true_valid, y_pred_valid)),
        'rmse': float(np.sqrt(mean_squared_error(y_true_valid, y_pred_valid))),
        'mae': float(mean_absolute_error(y_true_valid, y_pred_valid)),
        'n_samples': int(len(y_true_valid)),
        'n_invalid': int(len(y_true) - len(y_true_valid))
    }
    
    # 打印结果
    print("\n" + "="*60)
    print("  测试集性能")
    print("="*60)
    print(f"Pearson R:    {metrics['pearson_r']:.4f}")
    print(f"Spearman R:   {metrics['spearman_r']:.4f}")
    print(f"R² Score:     {metrics['r2_score']:.4f}")
    print(f"MSE:          {metrics['mse']:.4f}")
    print(f"RMSE:         {metrics['rmse']:.4f}")
    print(f"MAE:          {metrics['mae']:.4f}")
    print(f"样本数:       {metrics['n_samples']}")
    if metrics['n_invalid'] > 0:
        print(f"无效样本:     {metrics['n_invalid']}")
    print("="*60)
    
    # 6. 保存结果
    print("\n[5/5] 保存结果...")
    
    # 确保目录存在
    Path('results/plots').mkdir(parents=True, exist_ok=True)
    
    # 保存指标
    import json
    with open('results/test_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    print("✓ 指标已保存: results/test_metrics.json")
    
    # 保存预测结果
    results_df = pd.DataFrame({
        'true_expr': y_true_valid,
        'pred_expr': y_pred_valid
    })
    results_df.to_csv('results/test_predictions.csv', index=False)
    print("✓ 预测已保存: results/test_predictions.csv")
    
    # 保存潜在表示
    np.savez('results/test_latent.npz', 
             mu=mu, logvar=logvar)
    print("✓ 潜在表示已保存: results/test_latent.npz")
    
    # 7. 生成可视化
    print("\n生成可视化...")
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 散点图
    axes[0, 0].scatter(y_true_valid, y_pred_valid, alpha=0.3, s=1)
    axes[0, 0].plot([y_true_valid.min(), y_true_valid.max()], 
                    [y_true_valid.min(), y_true_valid.max()], 
                    'r--', lw=2)
    axes[0, 0].set_xlabel('真实表达 (log2(RPKM+1))')
    axes[0, 0].set_ylabel('预测表达 (log2(RPKM+1))')
    axes[0, 0].set_title(f'预测 vs 真实 (R={metrics["pearson_r"]:.3f})')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 残差图
    residuals = y_pred_valid - y_true_valid
    axes[0, 1].scatter(y_true_valid, residuals, alpha=0.3, s=1)
    axes[0, 1].axhline(y=0, color='r', linestyle='--', lw=2)
    axes[0, 1].set_xlabel('真实表达 (log2(RPKM+1))')
    axes[0, 1].set_ylabel('残差 (预测 - 真实)')
    axes[0, 1].set_title('残差分析')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 误差分布
    axes[1, 0].hist(residuals, bins=50, edgecolor='black', alpha=0.7)
    axes[1, 0].axvline(x=0, color='r', linestyle='--', lw=2)
    axes[1, 0].set_xlabel('残差')
    axes[1, 0].set_ylabel('频数')
    axes[1, 0].set_title(f'残差分布 (μ={residuals.mean():.3f}, σ={residuals.std():.3f})')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 按表达水平的性能
    bins = [0, 3, 7, 12, np.inf]
    labels = ['低\n(0-3)', '中\n(3-7)', '高\n(7-12)', '极高\n(>12)']
    results_df['level'] = pd.cut(results_df['true_expr'], bins=bins, labels=labels)
    
    level_metrics = []
    for level in labels:
        subset = results_df[results_df['level'] == level]
        if len(subset) > 10:
            r = pearsonr(subset['true_expr'], subset['pred_expr'])[0]
            level_metrics.append(r)
        else:
            level_metrics.append(0)
    
    axes[1, 1].bar(labels, level_metrics, edgecolor='black', alpha=0.7)
    axes[1, 1].set_xlabel('表达水平')
    axes[1, 1].set_ylabel('Pearson R')
    axes[1, 1].set_title('不同表达水平的性能')
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    axes[1, 1].set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig('results/plots/evaluation_report.png', dpi=300, bbox_inches='tight')
    print("✓ 可视化已保存: results/plots/evaluation_report.png")
    
    print("\n" + "="*60)
    print("  评估完成！")
    print("="*60)
    print("\n生成的文件:")
    print("  - results/test_metrics.json")
    print("  - results/test_predictions.csv")
    print("  - results/test_latent.npz")
    print("  - results/plots/evaluation_report.png")
    print()

if __name__ == '__main__':
    evaluate_model()

