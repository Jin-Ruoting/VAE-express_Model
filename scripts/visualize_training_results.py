#!/usr/bin/env python3
"""
可视化训练曲线和结果
"""
import re
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def parse_training_log(log_file='logs/train.log'):
    """解析训练日志"""
    
    # 尝试多个可能的日志位置
    possible_paths = [
        log_file,
        'logs/train.log',
        'results/logs/train.log'
    ]
    
    log_path = None
    for path in possible_paths:
        if Path(path).exists():
            log_path = path
            break
    
    if not log_path:
        print(f"✗ 找不到训练日志文件")
        print(f"  尝试的路径: {possible_paths}")
        return None
    
    print(f"✓ 解析日志: {log_path}")
    
    epochs = []
    train_loss, val_loss = [], []
    train_r, val_r = [], []
    train_recon, val_recon = [], []
    train_kl, val_kl = [], []
    train_expr, val_expr = [], []
    kl_beta = []
    
    current_epoch = None
    current_kl_beta = None
    
    with open(log_path, 'r') as f:
        for line in f:
            # 提取 epoch 和 KL beta
            epoch_match = re.search(r'\(epoch (\d+)\) KL beta: ([\d.e+-]+)', line)
            if epoch_match:
                current_epoch = int(epoch_match.group(1))
                current_kl_beta = float(epoch_match.group(2))
            
            # 提取训练指标
            if 'Train -' in line and current_epoch is not None:
                loss_match = re.search(r'Loss: ([\d.]+)', line)
                r_match = re.search(r'R: ([\d.]+)', line)
                recon_match = re.search(r'Recon: ([\d.]+)', line)
                kl_match = re.search(r'KL: ([\d.]+)', line)
                expr_match = re.search(r'Expr: ([\d.]+)', line)
                
                if all([loss_match, r_match]):
                    train_loss.append(float(loss_match.group(1)))
                    train_r.append(float(r_match.group(1)))
                    train_recon.append(float(recon_match.group(1)) if recon_match else 0)
                    train_kl.append(float(kl_match.group(1)) if kl_match else 0)
                    train_expr.append(float(expr_match.group(1)) if expr_match else 0)
            
            # 提取验证指标
            elif 'Val   -' in line and current_epoch is not None:
                loss_match = re.search(r'Loss: ([\d.]+)', line)
                r_match = re.search(r'R: ([\d.]+)', line)
                recon_match = re.search(r'Recon: ([\d.]+)', line)
                kl_match = re.search(r'KL: ([\d.]+)', line)
                expr_match = re.search(r'Expr: ([\d.]+)', line)
                
                if all([loss_match, r_match]):
                    epochs.append(current_epoch)
                    val_loss.append(float(loss_match.group(1)))
                    val_r.append(float(r_match.group(1)))
                    val_recon.append(float(recon_match.group(1)) if recon_match else 0)
                    val_kl.append(float(kl_match.group(1)) if kl_match else 0)
                    val_expr.append(float(expr_match.group(1)) if expr_match else 0)
                    if current_kl_beta is not None:
                        kl_beta.append(current_kl_beta)
    
    if not epochs:
        print("✗ 未能从日志中提取训练数据")
        return None
    
    return {
        'epochs': epochs,
        'train_loss': train_loss,
        'val_loss': val_loss,
        'train_r': train_r,
        'val_r': val_r,
        'train_recon': train_recon,
        'val_recon': val_recon,
        'train_kl': train_kl,
        'val_kl': val_kl,
        'train_expr': train_expr,
        'val_expr': val_expr,
        'kl_beta': kl_beta
    }

def plot_training_curves(data):
    """绘制训练曲线"""
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    
    epochs = data['epochs']
    
    # 总损失
    axes[0, 0].plot(epochs, data['train_loss'], label='训练', linewidth=2)
    axes[0, 0].plot(epochs, data['val_loss'], label='验证', linewidth=2)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('总损失')
    axes[0, 0].set_title('总损失')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Pearson R
    axes[0, 1].plot(epochs, data['train_r'], label='训练', linewidth=2)
    axes[0, 1].plot(epochs, data['val_r'], label='验证', linewidth=2)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Pearson R')
    axes[0, 1].set_title('表达预测相关性')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_ylim(0, 1)
    
    # 找出最佳epoch
    best_idx = np.argmax(data['val_r'])
    best_epoch = epochs[best_idx]
    axes[0, 1].axvline(x=best_epoch, color='red', linestyle='--', 
                       linewidth=2, label=f'最佳 (epoch {best_epoch})')
    axes[0, 1].legend()
    
    # 表达损失
    axes[0, 2].plot(epochs, data['train_expr'], label='训练', linewidth=2)
    axes[0, 2].plot(epochs, data['val_expr'], label='验证', linewidth=2)
    axes[0, 2].set_xlabel('Epoch')
    axes[0, 2].set_ylabel('表达损失')
    axes[0, 2].set_title('表达预测损失')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # 重构损失
    axes[1, 0].plot(epochs, data['train_recon'], label='训练', linewidth=2)
    axes[1, 0].plot(epochs, data['val_recon'], label='验证', linewidth=2)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('重构损失')
    axes[1, 0].set_title('重构损失')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # KL 损失和 beta
    ax_kl = axes[1, 1]
    ax_kl.plot(epochs, data['train_kl'], label='训练KL', linewidth=2)
    ax_kl.plot(epochs, data['val_kl'], label='验证KL', linewidth=2)
    ax_kl.set_xlabel('Epoch')
    ax_kl.set_ylabel('KL 散度', color='tab:blue')
    ax_kl.set_title('KL 散度与 Beta')
    ax_kl.tick_params(axis='y', labelcolor='tab:blue')
    ax_kl.legend(loc='upper left')
    ax_kl.grid(True, alpha=0.3)
    
    # 添加 KL beta 曲线
    if len(data['kl_beta']) == len(epochs):
        ax_beta = ax_kl.twinx()
        ax_beta.plot(epochs, data['kl_beta'], 'r--', 
                     label='KL Beta', linewidth=2, alpha=0.7)
        ax_beta.set_ylabel('KL Beta', color='tab:red')
        ax_beta.tick_params(axis='y', labelcolor='tab:red')
        ax_beta.legend(loc='upper right')
    
    # 最佳性能总结
    axes[1, 2].axis('off')
    best_r = data['val_r'][best_idx]
    summary_text = f"""
训练总结

最佳Epoch: {best_epoch}
最佳验证R: {best_r:.4f}

最终性能 (Epoch {epochs[-1]}):
  训练R: {data['train_r'][-1]:.4f}
  验证R: {data['val_r'][-1]:.4f}
  
  训练损失: {data['train_loss'][-1]:.4f}
  验证损失: {data['val_loss'][-1]:.4f}
  
  表达损失: {data['val_expr'][-1]:.4f}
  重构损失: {data['val_recon'][-1]:.4f}
  KL散度: {data['val_kl'][-1]:.4f}
  
总Epoch数: {len(epochs)}
    """
    axes[1, 2].text(0.1, 0.5, summary_text, 
                    fontsize=11, family='monospace',
                    verticalalignment='center',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.tight_layout()
    
    # 确保目录存在
    Path('results/plots').mkdir(parents=True, exist_ok=True)
    plt.savefig('results/plots/training_curves.png', dpi=300, bbox_inches='tight')
    print("✓ 训练曲线已保存: results/plots/training_curves.png")

def main():
    print("="*60)
    print("  可视化训练结果")
    print("="*60)
    
    print("\n解析训练日志...")
    data = parse_training_log()
    
    if data:
        print(f"✓ 解析完成: {len(data['epochs'])} 个epoch")
        print(f"  Epoch 范围: {data['epochs'][0]} - {data['epochs'][-1]}")
        print(f"  最佳验证R: {max(data['val_r']):.4f}")
        print(f"  最终验证R: {data['val_r'][-1]:.4f}")
        
        print("\n绘制训练曲线...")
        plot_training_curves(data)
        
        print("\n" + "="*60)
        print("  完成！")
        print("="*60)
    else:
        print("\n✗ 日志解析失败")
        print("\n请确保:")
        print("  1. 训练日志文件存在")
        print("  2. 日志格式正确")

if __name__ == '__main__':
    main()

