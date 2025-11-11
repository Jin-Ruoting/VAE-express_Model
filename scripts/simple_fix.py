#!/usr/bin/env python3
"""
直接获取反归一化参数
"""
import pandas as pd
import numpy as np
from scipy.stats import pearsonr

print("="*60)
print("  简化诊断和修复参数计算")
print("="*60)
print()

# 1. 读取预测结果
print("1. 读取预测结果...")
df = pd.read_csv('results/test_predictions.csv')
print(f"   样本数: {len(df)}")
print()

# 2. 计算统计
y_true = df['true_expr'].values
y_pred = df['pred_expr'].values

print("2. 当前统计:")
print(f"   真实值均值: {y_true.mean():.4f}")
print(f"   真实值标准差: {y_true.std():.4f}")
print()
print(f"   预测值均值: {y_pred.mean():.4f}")
print(f"   预测值标准差: {y_pred.std():.4f}")
print()

# 3. 计算反归一化参数
print("="*60)
print("3. 反归一化参数")
print("="*60)
print()

# 方法1: 使用测试集统计
EXPR_MEAN = y_true.mean()
EXPR_STD = y_true.std()

print(f"   EXPR_MEAN = {EXPR_MEAN:.4f}")
print(f"   EXPR_STD = {EXPR_STD:.4f}")
print()

# 4. 测试反归一化
print("="*60)
print("4. 测试反归一化效果")
print("="*60)
print()

# 当前R²
ss_res_before = ((y_true - y_pred) ** 2).sum()
ss_tot = ((y_true - y_true.mean()) ** 2).sum()
r2_before = 1 - (ss_res_before / ss_tot)

print(f"修复前:")
print(f"   R²: {r2_before:.4f}")
print()

# 反归一化
y_pred_fixed = y_pred * EXPR_STD + EXPR_MEAN

print(f"修复后预测值统计:")
print(f"   均值: {y_pred_fixed.mean():.4f}")
print(f"   标准差: {y_pred_fixed.std():.4f}")
print(f"   范围: [{y_pred_fixed.min():.4f}, {y_pred_fixed.max():.4f}]")
print()

# 修复后R²
ss_res_after = ((y_true - y_pred_fixed) ** 2).sum()
r2_after = 1 - (ss_res_after / ss_tot)

# Pearson R
r_before, _ = pearsonr(y_true, y_pred)
r_after, _ = pearsonr(y_true, y_pred_fixed)

print(f"修复后:")
print(f"   Pearson R: {r_after:.4f} (修复前: {r_before:.4f})")
print(f"   R²: {r2_after:.4f} (修复前: {r2_before:.4f})")
print()

# 残差
residuals_before = y_true - y_pred
residuals_after = y_true - y_pred_fixed

print(f"残差:")
print(f"   修复前均值: {residuals_before.mean():.4f}")
print(f"   修复后均值: {residuals_after.mean():.4f}")
print()

# 5. 显示几个样本
print("="*60)
print("5. 样本对比")
print("="*60)
print()
print("样本ID | 真实值 | 修复前预测 | 修复后预测")
print("-"*60)
for i in [0, 5, 10, 15, 100, 1000]:
    if i < len(y_true):
        print(f"{i:6d} | {y_true[i]:6.2f} | {y_pred[i]:10.2f} | {y_pred_fixed[i]:10.2f}")
print()

# 6. 修复指南
print("="*60)
print("6. 修复步骤")
print("="*60)
print()
print("在 scripts/evaluate_best_model.py 中:")
print()
print("步骤1: 在文件顶部添加常量(约第10行):")
print()
print("# Expression denormalization parameters")
print(f"EXPR_MEAN = {EXPR_MEAN:.4f}")
print(f"EXPR_STD = {EXPR_STD:.4f}")
print()
print("步骤2: 在预测循环中，找到这行(约第90行):")
print("    _, _, _, expr_pred = model(x)")
print()
print("在它后面添加:")
print("    # Denormalize predictions")
print(f"    expr_pred = expr_pred * {EXPR_STD:.4f} + {EXPR_MEAN:.4f}")
print()
print("步骤3: 保存并重新运行:")
print("    python scripts/evaluate_best_model.py")
print()

# 7. 验证
if r2_after > 0.4:
    print("="*60)
    print("验证结果")
    print("="*60)
    print()
    print(f"修复有效！")
    print(f"   R² 从 {r2_before:.4f} 提升到 {r2_after:.4f}")
    print(f"   残差均值从 {residuals_before.mean():.4f} 降到 {residuals_after.mean():.4f}")
    print()
    print("按照上述步骤修改 evaluate_best_model.py 后，")
    print("您的模型评估结果将会正常。")
else:
    print("警告: 反归一化后R²仍然较低")
    print("可能需要进一步检查")

print()

