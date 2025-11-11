#!/usr/bin/env python3
"""
检查训练和评估配置差异
找出预测值异常的根本原因
"""
import torch
import yaml
import numpy as np
from data.roadmap_dataset import create_dataloaders
from models.vae import VAE

print("="*60)
print("  配置差异检查")
print("="*60)
print()

# 1. 加载数据
print("1. 加载数据集...")
cfg = yaml.safe_load(open('config/config.yaml'))
train_loader, val_loader, test_loader = create_dataloaders('config/config.yaml')

# 获取test_loader的dataset
test_dataset = test_loader.dataset.dataset  # RandomSplit的dataset属性

print(f"   测试集样本数: {len(test_loader.dataset)}")
print(f"   zscore_per_eid: {test_dataset.zscore_per_eid}")
print(f"   expr_scale: {test_dataset.expr_scale}")
print()

# 2. 检查几个样本的原始数据
print("2. 检查原始数据...")
sample_indices = [0, 100, 1000]
for idx in sample_indices:
    sample = test_loader.dataset[idx]
    y_val = sample['y'].item()
    print(f"   样本 {idx}: y = {y_val:.4f}")

print()

# 3. 检查stats.json（如果存在）
import json
stats_path = cfg['paths']['stats_json']
try:
    with open(stats_path) as f:
        stats = json.load(f)
    
    print("3. stats.json内容:")
    if 'expression' in stats:
        print("   表达值统计:")
        expr_stats = stats['expression']
        for key, val in expr_stats.items():
            if isinstance(val, (int, float)):
                print(f"     {key}: {val:.4f}")
            else:
                print(f"     {key}: {val}")
    
    if 'per_eid_stats' in stats:
        print("   每个EID的统计:")
        for eid, eid_stats in list(stats['per_eid_stats'].items())[:3]:
            print(f"     {eid}:")
            if 'expression_mean' in eid_stats:
                print(f"       mean: {eid_stats['expression_mean']:.4f}")
            if 'expression_std' in eid_stats:
                print(f"       std: {eid_stats['expression_std']:.4f}")
except Exception as e:
    print(f"3. 无法读取stats.json: {e}")

print()

# 4. 加载模型并预测
print("4. 加载模型并检查输出...")
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"   使用设备: {device}")

model = VAE(input_channels=7, latent_dim=64, sequence_length=2000)
model = model.to(device)

# 加载权重
try:
    state_dict = torch.load('results/models/vae_promoter_only_best.pt', map_location=device)
    # 处理key mismatch
    new_state_dict = {}
    for key, value in state_dict.items():
        if 'regressor.regressor.' in key:
            new_key = key.replace('regressor.regressor.', 'regressor.core.')
            new_state_dict[new_key] = value
        else:
            new_state_dict[key] = value
    model.load_state_dict(new_state_dict)
    model.eval()
    print("   模型加载成功")
except Exception as e:
    print(f"   模型加载失败: {e}")
    exit(1)

print()

# 5. 对比batch的输入和输出
print("5. 对比模型输入输出...")
batch = next(iter(test_loader))
x = batch['x'].to(device)
y_true = batch['y'].cpu().numpy()

with torch.no_grad():
    _, _, _, expr_pred = model(x)
    y_pred = expr_pred.cpu().numpy()

print(f"   Batch size: {len(y_true)}")
print()
print(f"   真实值 (y_true):")
print(f"     范围: [{y_true.min():.4f}, {y_true.max():.4f}]")
print(f"     均值: {y_true.mean():.4f}")
print(f"     标准差: {y_true.std():.4f}")
print()
print(f"   预测值 (y_pred):")
print(f"     范围: [{y_pred.min():.4f}, {y_pred.max():.4f}]")
print(f"     均值: {y_pred.mean():.4f}")
print(f"     标准差: {y_pred.std():.4f}")
print()

# 6. 对比几个样本
print("6. 对比具体样本:")
for i in range(min(5, len(y_true))):
    print(f"   样本 {i}: 真实={y_true[i]:.4f}, 预测={y_pred[i]:.4f}, 差值={y_true[i]-y_pred[i]:.4f}")

print()

# 7. 关键诊断
print("="*60)
print("  关键诊断")
print("="*60)
print()

# 计算相关性
from scipy.stats import pearsonr
r, _ = pearsonr(y_true.flatten(), y_pred.flatten())
print(f"1. Batch内Pearson R: {r:.4f}")
print()

# 检查是否需要反归一化
std_ratio = y_pred.std() / y_true.std()
mean_diff = abs(y_pred.mean() - y_true.mean())

print(f"2. 尺度检查:")
print(f"   标准差比值 (pred/true): {std_ratio:.4f}")
print(f"   均值差异: {mean_diff:.4f}")
print()

if std_ratio < 0.7:
    print("   问题: 预测值标准差明显小于真实值")
    print("   可能原因: 训练时对标签做了标准化，但模型输出没有反归一化")
    print()
    
    # 尝试反归一化
    print("3. 尝试反归一化:")
    
    # 假设训练时做了 y = (y - mean) / std
    # 那么需要 y_original = y_normalized * std + mean
    
    # 方案1: 使用当前batch的统计
    y_pred_denorm1 = y_pred * y_true.std() + y_true.mean()
    r1, _ = pearsonr(y_true.flatten(), y_pred_denorm1.flatten())
    ss_res1 = ((y_true - y_pred_denorm1) ** 2).sum()
    ss_tot1 = ((y_true - y_true.mean()) ** 2).sum()
    r2_1 = 1 - (ss_res1 / ss_tot1)
    
    print(f"   方案1 (用测试集统计反归一化):")
    print(f"     预测值范围: [{y_pred_denorm1.min():.4f}, {y_pred_denorm1.max():.4f}]")
    print(f"     Pearson R: {r1:.4f}")
    print(f"     R²: {r2_1:.4f}")
    print()
    
    # 方案2: 尝试不同的均值和标准差
    # 如果训练时用的是全局统计
    print(f"   方案2 (尝试调整参数):")
    print(f"     当前预测值均值≈0，标准差≈1.29")
    print(f"     真实值均值≈2.33，标准差≈2.14")
    print(f"     需要: y_original = y_pred * 2.14 + 2.33")
    
    y_pred_denorm2 = y_pred * 2.14 + 2.33
    r2, _ = pearsonr(y_true.flatten(), y_pred_denorm2.flatten())
    ss_res2 = ((y_true - y_pred_denorm2) ** 2).sum()
    ss_tot2 = ((y_true - y_true.mean()) ** 2).sum()
    r2_2 = 1 - (ss_res2 / ss_tot2)
    
    print(f"     预测值范围: [{y_pred_denorm2.min():.4f}, {y_pred_denorm2.max():.4f}]")
    print(f"     Pearson R: {r2:.4f}")
    print(f"     R²: {r2_2:.4f}")
    print()
else:
    print("   标准差比值正常")

print()

if mean_diff > 1.0:
    print("   问题: 预测值均值与真实值均值相差很大")
    print("   可能原因: 训练时对标签做了中心化")
    print()

print("="*60)
print("  结论")
print("="*60)
print()

if std_ratio < 0.7 or mean_diff > 1.0:
    print("检测到归一化/标准化问题！")
    print()
    print("训练时很可能:")
    print("  1. 对标签做了 z-score标准化: y = (y - mean) / std")
    print("  2. 模型学会了预测标准化后的值")
    print("  3. 但评估时没有反归一化")
    print()
    print("修复方法:")
    print("  选项A: 找到训练时的mean和std，在评估脚本中反归一化")
    print("  选项B: 重新训练，确保标签不被标准化")
    print()
else:
    print("未检测到明显的归一化问题")
    print("需要进一步检查其他可能原因")

print()

