#!/usr/bin/env python3
"""
全面评估最佳模型
"""
import os
import sys
from pathlib import Path
import argparse
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

EXPR_MEAN = 2.3276
EXPR_STD = 2.1382

def evaluate_model(run_dir=None, model_name='vae_best.pt'):
    """Evaluate model performance on test set
    
    Args:
        run_dir: 运行目录路径，如 'results/20241112-1430_raw'
        model_name: 模型文件名，默认 'vae_best.pt'
    """
    
    # 确定运行目录和输出目录
    if run_dir is None:
        # 默认使用兼容路径
        run_dir = Path('results')
        model_path = run_dir / 'models' / 'vae_promoter_only_best.pt'
        output_dir = run_dir
    else:
        run_dir = Path(run_dir)
        model_path = run_dir / 'models' / model_name
        output_dir = run_dir
    
    print("="*70)
    print("  VAE Model Evaluation")
    print("="*70)
    print(f"Run directory: {run_dir}")
    print(f"Model path: {model_path}")
    print(f"Output directory: {output_dir}")
    print("="*70)
    print()
    
    # 1. Load configuration
    cfg = yaml.safe_load(open('config/config.yaml'))
    
    # 2. Create data loaders
    print("\n[1/5] Loading data...")
    _, _, test_loader = create_dataloaders('config/config.yaml')
    print(f"Test batches: {len(test_loader)}")
    
    # 3. Load model
    print("\n[2/5] Loading model...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    marks = cfg['marks']['core'] + (cfg['marks']['extra'] if cfg.get('use_extra') else [])
    seq_len = cfg['sequence']['promoter_bp']
    
    model = VAE(
        input_channels=len(marks),
        latent_dim=64,
        sequence_length=seq_len
    )
    
    # Load best model weights
    if not model_path.exists():
        print(f"Error: model file not found: {model_path}")
        return
    
    state_dict = torch.load(model_path, map_location=device, weights_only=False)
    
    # Fix key mismatch: regressor.regressor.* -> regressor.core.*
    new_state_dict = {}
    for key, value in state_dict.items():
        if 'regressor.regressor.' in key:
            new_key = key.replace('regressor.regressor.', 'regressor.core.')
            new_state_dict[new_key] = value
        else:
            new_state_dict[key] = value
    
    model.load_state_dict(new_state_dict)
    model = model.to(device)
    model.eval()
    print(f"Model loaded: {model_path}")
    
    # 4. Run prediction
    print("\n[3/5] Running predictions...")
    
    all_true = []
    all_pred = []
    all_mu = []
    all_logvar = []
    
    # 用于调试：收集原始预测值（反归一化前）
    raw_preds_for_debug = []
    
    with torch.no_grad():
        for x, y in tqdm(test_loader, desc="Prediction"):
            x = x.to(device)
            y = y.to(device)
            
            x_hat, mu, logvar, expr_pred = model(x)
            
            # 保存原始预测值用于调试
            if len(raw_preds_for_debug) < 100:  # 只保存前100个batch
                raw_preds_for_debug.append(expr_pred.cpu().numpy())
            
            # Denormalize predictions from z-score to log2(RPKM+1)
            expr_pred = expr_pred * EXPR_STD + EXPR_MEAN
            
            all_true.append(y.cpu().numpy())
            all_pred.append(expr_pred.cpu().numpy())
            all_mu.append(mu.cpu().numpy())
            all_logvar.append(logvar.cpu().numpy())
    
    # Concatenate results
    y_true = np.concatenate(all_true, axis=0).reshape(-1)
    y_pred = np.concatenate(all_pred, axis=0).reshape(-1)
    mu = np.concatenate(all_mu, axis=0)
    logvar = np.concatenate(all_logvar, axis=0)
    
    # 调试输出：检查原始预测值的统计
    if raw_preds_for_debug:
        raw_pred_sample = np.concatenate(raw_preds_for_debug, axis=0).reshape(-1)
        print(f"\n[DEBUG] Raw model output (before denormalization):")
        print(f"  Mean: {raw_pred_sample.mean():.4f}")
        print(f"  Std:  {raw_pred_sample.std():.4f}")
        print(f"  Range: [{raw_pred_sample.min():.4f}, {raw_pred_sample.max():.4f}]")
        print(f"  Expected if z-score: mean≈0, std≈1")
        print(f"  Expected if log2(RPKM+1): mean≈2.3, std≈2.1\n")
    
    print(f"Predictions completed: {len(y_true)} samples")
    
    # 5. Calculate metrics
    print("\n[4/5] Computing evaluation metrics...")
    
    # Filter invalid values
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
    
    # Print results
    print("\n" + "="*60)
    print("  Test Set Performance")
    print("="*60)
    print(f"Pearson R:       {metrics['pearson_r']:.4f}")
    print(f"Spearman R:      {metrics['spearman_r']:.4f}")
    print(f"R² Score:        {metrics['r2_score']:.4f}")
    print(f"MSE:             {metrics['mse']:.4f}")
    print(f"RMSE:            {metrics['rmse']:.4f}")
    print(f"MAE:             {metrics['mae']:.4f}")
    print(f"Sample count:    {metrics['n_samples']}")
    if metrics['n_invalid'] > 0:
        print(f"Invalid samples: {metrics['n_invalid']}")
    print("="*60)
    
    # 6. Save results
    print("\n[5/5] Saving results...")
    
    # Ensure directories exist
    (output_dir / 'plots').mkdir(parents=True, exist_ok=True)
    
    # Save metrics
    import json
    metrics_path = output_dir / 'test_metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print("Metrics saved: results/test_metrics.json")
    
    # Save predictions
    results_df = pd.DataFrame({
        'true_expr': y_true_valid,
        'pred_expr': y_pred_valid
    })
    pred_path = output_dir / 'test_predictions.csv'
    results_df.to_csv(pred_path, index=False)
    print(f"Predictions saved: {pred_path}")
    
    # Save latent representations
    latent_path = output_dir / 'test_latent.npz'
    np.savez(latent_path, mu=mu, logvar=logvar)
    print(f"Latent representations saved: {latent_path}")
    
    # 7. Generate visualizations
    print("\nGenerating visualizations...")
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Scatter plot
    axes[0, 0].scatter(y_true_valid, y_pred_valid, alpha=0.3, s=1)
    axes[0, 0].plot([y_true_valid.min(), y_true_valid.max()], 
                    [y_true_valid.min(), y_true_valid.max()], 
                    'r--', lw=2)
    axes[0, 0].set_xlabel('True Expression (log2(RPKM+1))')
    axes[0, 0].set_ylabel('Predicted Expression (log2(RPKM+1))')
    axes[0, 0].set_title(f'Prediction vs. Ground Truth (R={metrics["pearson_r"]:.3f})')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Residual plot
    residuals = y_pred_valid - y_true_valid
    axes[0, 1].scatter(y_true_valid, residuals, alpha=0.3, s=1)
    axes[0, 1].axhline(y=0, color='r', linestyle='--', lw=2)
    axes[0, 1].set_xlabel('True Expression (log2(RPKM+1))')
    axes[0, 1].set_ylabel('Residuals (Predicted - True)')
    axes[0, 1].set_title('Residual Analysis')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Residual distribution
    axes[1, 0].hist(residuals, bins=50, edgecolor='black', alpha=0.7)
    axes[1, 0].axvline(x=0, color='r', linestyle='--', lw=2)
    axes[1, 0].set_xlabel('Residuals')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title(f'Residual Distribution (μ={residuals.mean():.3f}, σ={residuals.std():.3f})')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Performance by expression level
    bins = [0, 3, 7, 12, np.inf]
    labels = ['Low\n(0-3)', 'Medium\n(3-7)', 'High\n(7-12)', 'Very High\n(>12)']
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
    axes[1, 1].set_xlabel('Expression Level')
    axes[1, 1].set_ylabel('Pearson R')
    axes[1, 1].set_title('Performance by Expression Level')
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    axes[1, 1].set_ylim(0, 1)
    
    plt.tight_layout()
    plot_path = output_dir / 'plots' / 'evaluation_report.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Visualization saved: {plot_path}")
    
    print("\n" + "="*70)
    print("  Evaluation completed")
    print("="*70)
    print("\nGenerated files:")
    print(f"  - {metrics_path}")
    print(f"  - {pred_path}")
    print(f"  - {latent_path}")
    print(f"  - {plot_path}")
    print("="*70)
    print()
    
    return output_dir

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='评估 VAE 模型')
    parser.add_argument('--run_dir', type=str, default=None,
                       help='运行目录路径，如 results/20241112-1430_raw。默认使用 results/')
    parser.add_argument('--model_name', type=str, default='vae_best.pt',
                       help='模型文件名，默认 vae_best.pt')
    args = parser.parse_args()
    
    evaluate_model(run_dir=args.run_dir, model_name=args.model_name)

