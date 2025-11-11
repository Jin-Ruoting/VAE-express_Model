#!/usr/bin/env python3
"""
诊断预测值异常问题
"""
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt

print("="*60)
print("  预测结果诊断")
print("="*60)
print()

# 1. 读取预测结果
print("读取预测结果...")
df = pd.read_csv('results/test_predictions.csv')
print(f"样本数: {len(df)}")
print()

# 2. 基本统计
print("="*60)
print("基本统计")
print("="*60)
print()

print("真实值统计:")
print(f"  最小值: {df['true_expr'].min():.4f}")
print(f"  最大值: {df['true_expr'].max():.4f}")
print(f"  均值:   {df['true_expr'].mean():.4f}")
print(f"  标准差: {df['true_expr'].std():.4f}")
print(f"  中位数: {df['true_expr'].median():.4f}")
print()

print("预测值统计:")
print(f"  最小值: {df['pred_expr'].min():.4f}")
print(f"  最大值: {df['pred_expr'].max():.4f}")
print(f"  均值:   {df['pred_expr'].mean():.4f}")
print(f"  标准差: {df['pred_expr'].std():.4f}")
print(f"  中位数: {df['pred_expr'].median():.4f}")
print()

# 3. 关键问题检查
print("="*60)
print("关键问题检查")
print("="*60)
print()

# 问题1: 预测值范围
true_range = df['true_expr'].max() - df['true_expr'].min()
pred_range = df['pred_expr'].max() - df['pred_expr'].min()
print(f"1. 预测值范围问题:")
print(f"   真实值范围: {true_range:.4f}")
print(f"   预测值范围: {pred_range:.4f}")
print(f"   范围比值:   {pred_range/true_range:.4f}")
if pred_range < true_range * 0.5:
    print(f"   预测值范围过窄！仅覆盖真实值范围的 {pred_range/true_range*100:.1f}%")
else:
    print(f"   预测值范围正常")
print()

# 问题2: 预测值是否都接近0
near_zero = (df['pred_expr'].abs() < 0.5).sum()
print(f"2. 预测值接近0的问题:")
print(f"   接近0的样本数 (|pred| < 0.5): {near_zero} / {len(df)} ({near_zero/len(df)*100:.1f}%)")
if near_zero / len(df) > 0.8:
    print(f"   大部分预测值接近0！")
else:
    print(f"   预测值分布正常")
print()

# 问题3: 系统性偏差
residuals = df['true_expr'] - df['pred_expr']
print(f"3. 系统性偏差:")
print(f"   残差均值: {residuals.mean():.4f}")
print(f"   残差标准差: {residuals.std():.4f}")
if abs(residuals.mean()) > 1.0:
    print(f"   存在严重系统性偏差！")
    if residuals.mean() > 0:
        print(f"   → 预测值系统性偏低 {residuals.mean():.2f}")
    else:
        print(f"   → 预测值系统性偏高 {abs(residuals.mean()):.2f}")
else:
    print(f"   无明显系统性偏差")
print()

# 问题4: R²异常
from scipy.stats import pearsonr
r, _ = pearsonr(df['true_expr'], df['pred_expr'])
ss_res = ((df['true_expr'] - df['pred_expr']) ** 2).sum()
ss_tot = ((df['true_expr'] - df['true_expr'].mean()) ** 2).sum()
r2 = 1 - (ss_res / ss_tot)
print(f"4. R²异常:")
print(f"   Pearson R: {r:.4f}")
print(f"   R²:        {r2:.4f}")
print(f"   SS_res:    {ss_res:.2f}")
print(f"   SS_tot:    {ss_tot:.2f}")
if r2 < 0:
    print(f"   R²为负！模型预测比直接预测均值还差")
    print(f"   → 可能原因：预测值范围/尺度有严重问题")
else:
    print(f"   R²正常")
print()

# 5. 预测值分布
print("="*60)
print("预测值分布统计")
print("="*60)
print()

# 按真实值分组
df['true_quartile'] = pd.qcut(df['true_expr'], q=4, labels=['Q1 (低)', 'Q2 (中低)', 'Q3 (中高)', 'Q4 (高)'])
print("不同真实表达水平的预测情况:")
for q in ['Q1 (低)', 'Q2 (中低)', 'Q3 (中高)', 'Q4 (高)']:
    subset = df[df['true_quartile'] == q]
    if len(subset) > 0:
        print(f"\n{q}:")
        print(f"  样本数:       {len(subset)}")
        print(f"  真实值范围:   [{subset['true_expr'].min():.2f}, {subset['true_expr'].max():.2f}]")
        print(f"  预测值范围:   [{subset['pred_expr'].min():.2f}, {subset['pred_expr'].max():.2f}]")
        print(f"  预测均值:     {subset['pred_expr'].mean():.4f}")
        r_q, _ = pearsonr(subset['true_expr'], subset['pred_expr'])
        print(f"  组内R值:      {r_q:.4f}")

print()
print("="*60)
print("可能的原因")
print("="*60)
print()

# 综合诊断
issues = []
if pred_range < true_range * 0.5:
    issues.append("预测值范围过窄")
if near_zero / len(df) > 0.8:
    issues.append("预测值集中在0附近")
if abs(residuals.mean()) > 1.0:
    issues.append(f"系统性偏差 ({residuals.mean():.2f})")
if r2 < 0:
    issues.append("R²为负")

if issues:
    print("检测到以下问题:")
    for i, issue in enumerate(issues, 1):
        print(f"  {i}. {issue}")
    print()
    print("可能的原因:")
    print("  1. 数据归一化/标准化问题")
    print("     - 训练时归一化了，但预测时忘记反归一化")
    print("     - 或者归一化参数错误")
    print()
    print("  2. 模型输出层问题")
    print("     - 激活函数限制了输出范围")
    print("     - 输出层权重初始化不当")
    print()
    print("  3. 损失函数/标签问题")
    print("     - 训练时标签的变换与预测时不一致")
    print("     - log2变换应用错误")
    print()
    print("  4. 数据加载问题")
    print("     - 测试集数据预处理与训练集不同")
    print()
else:
    print("未检测到明显问题")

print("="*60)
print("建议的修复方向")
print("="*60)
print()

if pred_range < true_range * 0.5 or near_zero / len(df) > 0.8:
    print("1. 检查 data/roadmap_dataset.py 中的:")
    print("   - _transform_expr 函数")
    print("   - __getitem__ 中的标签处理")
    print()
    print("2. 检查 scripts/evaluate_best_model.py 中的:")
    print("   - 是否正确处理模型输出")
    print("   - 是否需要反变换")
    print()
    print("3. 打印几个样本查看:")
    print("   python -c \"")
    print("   import pandas as pd")
    print("   df = pd.read_csv('results/test_predictions.csv')")
    print("   print(df.head(20))")
    print("   \"")

print()

