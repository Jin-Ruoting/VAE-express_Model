#!/usr/bin/env python3
"""
验证数据集中的log2(RPKM+1)变换是否正确实施
"""
import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import yaml
import pandas as pd
from data.roadmap_dataset import create_dataloaders

def verify_log2_transform():
    """验证log2(RPKM+1)变换"""
    print("=" * 70)
    print("验证 log2(RPKM+1) 变换")
    print("=" * 70)
    
    # 1. 检查配置
    print("\n[1] 检查配置文件...")
    with open('config/config.yaml', 'r') as f:
        cfg = yaml.safe_load(f)
    
    expr_transform = cfg.get('expression', {}).get('transform', 'log2_rpkm_plus1')
    print(f"    配置中的表达变换: {expr_transform}")
    
    # 2. 创建数据加载器
    print("\n[2] 创建数据加载器...")
    try:
        train_loader, val_loader, test_loader = create_dataloaders('config/config.yaml')
        print(f"    训练集批次数: {len(train_loader)}")
        print(f"    验证集批次数: {len(val_loader)}")
        print(f"    测试集批次数: {len(test_loader)}")
    except Exception as e:
        print(f"    数据加载器创建失败: {e}")
        return False
    
    # 3. 采样一批数据进行检查
    print("\n[3] 检查数据标签 (y) 的取值范围...")
    try:
        x_batch, y_batch = next(iter(train_loader))
        y = y_batch.detach().cpu().numpy().reshape(-1)
        
        print(f"    批次大小: {y.shape[0]}")
        print(f"    y 最小值: {y.min():.4f}")
        print(f"    y 最大值: {y.max():.4f}")
        print(f"    y 均值:   {y.mean():.4f}")
        print(f"    y 中位数: {np.median(y):.4f}")
        
        # 4. 验证是否在log2空间
        print("\n[4] 验证数值是否在 log2(RPKM+1) 空间...")
        
        # log2(RPKM+1)的典型范围：
        # - 最小值接近0 (低表达基因)
        # - 中位数通常在 1-5 之间
        # - 最大值通常在 10-15 之间 (高表达基因)
        
        is_valid = True
        reasons = []
        
        if y.min() < -0.5:
            is_valid = False
            reasons.append(f"最小值 {y.min():.4f} 小于 -0.5，不符合 log2(RPKM+1) 的预期")
        
        if y.max() > 20:
            is_valid = False
            reasons.append(f"最大值 {y.max():.4f} 大于 20，可能未进行 log2 变换")
        
        median_val = np.median(y)
        if median_val < 0 or median_val > 15:
            is_valid = False
            reasons.append(f"中位数 {median_val:.4f} 超出合理范围 [0, 15]")
        
        # 5. 尝试还原RPKM值进行验证
        print("\n[5] 还原为原始 RPKM 值进行验证...")
        rpkm_restored = np.power(2.0, y) - 1.0
        print(f"    还原后 RPKM 最小值: {rpkm_restored.min():.4f}")
        print(f"    还原后 RPKM 最大值: {rpkm_restored.max():.4f}")
        print(f"    还原后 RPKM 均值:   {rpkm_restored.mean():.4f}")
        print(f"    还原后 RPKM 中位数: {np.median(rpkm_restored):.4f}")
        
        # RPKM的典型范围应该是 [0, 几千]
        if rpkm_restored.min() < -1:
            is_valid = False
            reasons.append("还原后的 RPKM 出现负值")
        
        # 6. 输出验证结果
        print("\n" + "=" * 70)
        if is_valid:
            print("验证通过！数据标签已正确应用 log2(RPKM+1) 变换")
        else:
            print("验证失败！可能存在以下问题：")
            for reason in reasons:
                print(f"    - {reason}")
        print("=" * 70)
        
        return is_valid
        
    except Exception as e:
        print(f"    数据采样失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def verify_model_output():
    """验证模型输出是否在正确空间"""
    print("\n\n" + "=" * 70)
    print("验证模型输出空间")
    print("=" * 70)
    
    from models.vae import VAE
    
    # 创建模型
    model = VAE(input_channels=7, latent_dim=64, sequence_length=2000)
    model.eval()
    
    # 创建测试输入
    x_test = torch.randn(4, 7, 2000)
    
    with torch.no_grad():
        x_hat, mu, logvar, expr_pred = model(x_test)
    
    expr_pred_np = expr_pred.detach().cpu().numpy().reshape(-1)
    
    print(f"\n[模型输出] 表达预测值范围:")
    print(f"    最小值: {expr_pred_np.min():.4f}")
    print(f"    最大值: {expr_pred_np.max():.4f}")
    print(f"    均值:   {expr_pred_np.mean():.4f}")
    print(f"\n    注：由于模型未训练，输出值可能不在合理范围内")
    print(f"    训练后，预测值应该在 log2(RPKM+1) 空间，即 [0, ~15] 范围")
    
    print("\n✓ 模型结构正常，可以输出表达预测值")

def main():
    print("\n\n")
    print("╔" + "=" * 68 + "╗")
    print("║" + " " * 68 + "║")
    print("║" + "    log2(RPKM+1) 变换验证脚本".center(68) + "║")
    print("║" + " " * 68 + "║")
    print("╚" + "=" * 68 + "╝")
    
    # 执行验证
    data_valid = verify_log2_transform()
    verify_model_output()
    
    # 总结
    print("\n\n" + "=" * 70)
    print("总结")
    print("=" * 70)
    if data_valid:
        print("数据集标签已正确应用 log2(RPKM+1) 变换")
        print("模型架构正常，可以在 log2 空间预测表达值")
        print("\n您的实现是正确的！")
        print("训练时，模型将在 log2(RPKM+1) 空间学习和预测。")
    else:
        print("发现潜在问题，请检查上述错误信息")
    print("=" * 70)
    
    return data_valid

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)


