# train/losses.py

import os
import warnings
import torch
import torch.nn.functional as F

def mse_reconstruction_loss(x_hat, x):
    """
    输入和重建的组蛋白信号之间的MSE损失
    """
    x_hat = torch.nan_to_num(x_hat, nan=0.0, posinf=0.0, neginf=0.0)
    x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    return F.mse_loss(x_hat, x, reduction='mean')

def kl_divergence_loss(mu, logvar):
    """
    计算变分自编码器中的KL散度项
    KL(N(μ,σ²) || N(0,1)) = -0.5 * Σ(1 + logσ² - μ² - σ²)
    """
    mu = torch.nan_to_num(mu, nan=0.0, posinf=0.0, neginf=0.0)
    logvar = torch.nan_to_num(logvar, nan=0.0, posinf=0.0, neginf=0.0)
    mu = mu.view(mu.shape[0], -1)
    logvar = logvar.view(logvar.shape[0], -1)
    if mu.shape[1] != logvar.shape[1]:
        d = min(mu.shape[1], logvar.shape[1])
        warnings.warn(f"KL: mu/logvar dim mismatch {mu.shape[1]} vs {logvar.shape[1]} -> truncate to {d}")
        mu = mu[:, :d]
        logvar = logvar[:, :d]
    kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    kl = kl.sum(dim=-1).mean()
    return kl

def _align_flat(a: torch.Tensor):
    if a.dim() == 1:
        return a.view(a.shape[0], 1)
    if a.dim() > 2:
        return a.view(a.shape[0], -1)
    return a

def _batch_zscore(a: torch.Tensor, eps=1e-6):
    # 对 batch 维做标准化，消除尺度差异
    mean = a.mean(dim=0, keepdim=True)
    std = a.std(dim=0, unbiased=False, keepdim=True).clamp_min(eps)
    return (a - mean) / std

def _corr_loss(pred, target, eps=1e-6):
    # Pearson 相关损失：1 - mean_corr（按列取平均）
    px = pred - pred.mean(dim=0, keepdim=True)
    py = target - target.mean(dim=0, keepdim=True)
    px = px / px.norm(dim=0, keepdim=True).clamp_min(eps)
    py = py / py.norm(dim=0, keepdim=True).clamp_min(eps)
    r = (px * py).sum(dim=0).mean()  # 平均到标量
    return 1.0 - r

def expression_prediction_loss(pred_expr, true_expr, alpha=None):
    """
    表达项 = alpha * MSE + (1-alpha) * 相关损失
    
    可通过环境变量控制：
    - EXPR_ALPHA: MSE 权重 (默认 0.3)
    - EXPR_LOSS_MODE: 'zscore' 或 'raw' (默认 'zscore')
      - 'zscore': MSE 在批内标准化后的值上计算（关注相关性）
      - 'raw': MSE 在原始值上计算（关注绝对值准确性）
    """
    if alpha is None:
        alpha = float(os.getenv('EXPR_ALPHA', '0.3'))  # 更强调相关性
    
    loss_mode = os.getenv('EXPR_LOSS_MODE', 'zscore').lower()
    
    pred_expr = torch.nan_to_num(pred_expr, nan=0.0, posinf=0.0, neginf=0.0)
    true_expr = torch.nan_to_num(true_expr, nan=0.0, posinf=0.0, neginf=0.0)
    pred_expr = _align_flat(pred_expr)
    true_expr = _align_flat(true_expr)
    
    # 对齐列数
    if pred_expr.shape[1] != true_expr.shape[1]:
        d = min(pred_expr.shape[1], true_expr.shape[1])
        pred_expr = pred_expr[:, :d]
        true_expr = true_expr[:, :d]
    
    # MSE 计算方式
    if loss_mode == 'raw':
        # 直接在原始值上计算 MSE（优化 R²）
        mse = F.mse_loss(pred_expr, true_expr, reduction='mean')
    else:
        # 批内标准化后计算 MSE（优化相关性）
        pred_z = _batch_zscore(pred_expr)
        true_z = _batch_zscore(true_expr)
        mse = F.mse_loss(pred_z, true_z, reduction='mean')
    
    # 相关损失
    corr = _corr_loss(pred_expr, true_expr)
    
    return alpha * mse + (1.0 - alpha) * corr

def total_vae_loss(x_hat, x, mu, logvar, pred_expr, y,
                   recon_weight=None, kl_weight=None, expr_weight=None):
    """
    权重可用环境变量覆盖：RECON_W / KL_W / EXPR_W
    建议先弱化 KL 与重构
    """
    rw = float(os.getenv('RECON_W', '0.01')) if recon_weight is None else float(recon_weight)
    kw = float(os.getenv('KL_W',    '0.05')) if kl_weight    is None else float(kl_weight)
    ew = float(os.getenv('EXPR_W',  '2.0'))  if expr_weight  is None else float(expr_weight)

    recon = mse_reconstruction_loss(x_hat, x)
    kl    = kl_divergence_loss(mu, logvar)
    expr  = expression_prediction_loss(pred_expr, y, alpha=0.5)

    total = rw * recon + kw * kl + ew * expr
    return total, {'recon_loss': float(recon.detach().cpu()),
                   'kl_loss': float(kl.detach().cpu()),
                   'expr_loss': float(expr.detach().cpu())}