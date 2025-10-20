# train/losses.py

import torch
import torch.nn.functional as F

def mse_reconstruction_loss(x_hat, x):
    """
    输入和重建的组蛋白信号之间的MSE损失
    """
    return F.mse_loss(x_hat, x)

def kl_divergence_loss(mu, logvar):
    """
    计算变分自编码器中的KL散度项
    KL(N(μ,σ²) || N(0,1)) = -0.5 * Σ(1 + logσ² - μ² - σ²)
    """
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
    return kl.mean()

def expression_prediction_loss(pred_expr, true_expr):
    """
    基因表达预测误差 (MSE)
    """
    return F.mse_loss(pred_expr.squeeze(), true_expr.squeeze())

def total_vae_loss(x_hat, x, mu, logvar, pred_expr, y,
                   recon_weight=None, kl_weight=None, expr_weight=None):
    """
    支持在训练时传入动态的权重（不传则使用模块内默认值）
    """
    recon_loss = mse_reconstruction_loss(x_hat, x)
    kl = kl_divergence_loss(mu, logvar)
    expr_loss = expression_prediction_loss(pred_expr, y)

    # 读取默认权重
    default_recon_w = 1.0
    default_kl_w    = 1e-4  # Increased from 1e-5
    default_expr_w  = 5.0   # Decreased from 15.0

    rw = default_recon_w if recon_weight is None else recon_weight
    kw = default_kl_w    if kl_weight    is None else kl_weight
    ew = default_expr_w  if expr_weight  is None else expr_weight

    total = rw * recon_loss + kw * kl + ew * expr_loss
    return total, {
        'recon_loss': float(rw * recon_loss),
        'kl_loss':    float(kw * kl),
        'expr_loss':  float(ew * expr_loss)
    }