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

def total_vae_loss(x_hat, x, mu, logvar, pred_expr, true_expr,
                   recon_weight=1.0, kl_weight=0.01, expr_weight=10.0):
    """
    三项联合损失函数
    """
    recon = mse_reconstruction_loss(x_hat, x)
    kl = kl_divergence_loss(mu, logvar)
    expr = expression_prediction_loss(pred_expr, true_expr)

    total = recon_weight * recon + kl_weight * kl + expr_weight * expr
    return total, {'recon_loss': recon.item(),
                   'kl_loss': kl.item(),
                   'expr_loss': expr.item()}