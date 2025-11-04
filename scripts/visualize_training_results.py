#!/usr/bin/env python3
"""
Visualize training curves and results
"""
import re
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def parse_training_log(log_file='logs/train.log'):
    """Parse training log file"""
    
    # Try multiple possible log locations
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
        print(f"Error: Training log file not found")
        print(f"  Tried paths: {possible_paths}")
        return None
    
    print(f"Parsing log: {log_path}")
    
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
        print("Error: Failed to extract training data from log")
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
    """Plot training curves"""
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    
    epochs = data['epochs']
    
    # Total loss
    axes[0, 0].plot(epochs, data['train_loss'], label='Train', linewidth=2)
    axes[0, 0].plot(epochs, data['val_loss'], label='Validation', linewidth=2)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Total Loss')
    axes[0, 0].set_title('Total Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Pearson R
    axes[0, 1].plot(epochs, data['train_r'], label='Train', linewidth=2)
    axes[0, 1].plot(epochs, data['val_r'], label='Validation', linewidth=2)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Pearson R')
    axes[0, 1].set_title('Expression Prediction Correlation')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_ylim(0, 1)
    
    # Mark best epoch
    best_idx = np.argmax(data['val_r'])
    best_epoch = epochs[best_idx]
    axes[0, 1].axvline(x=best_epoch, color='red', linestyle='--', 
                       linewidth=2, label=f'Best (epoch {best_epoch})')
    axes[0, 1].legend()
    
    # Expression loss
    axes[0, 2].plot(epochs, data['train_expr'], label='Train', linewidth=2)
    axes[0, 2].plot(epochs, data['val_expr'], label='Validation', linewidth=2)
    axes[0, 2].set_xlabel('Epoch')
    axes[0, 2].set_ylabel('Expression Loss')
    axes[0, 2].set_title('Expression Prediction Loss')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # Reconstruction loss
    axes[1, 0].plot(epochs, data['train_recon'], label='Train', linewidth=2)
    axes[1, 0].plot(epochs, data['val_recon'], label='Validation', linewidth=2)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Reconstruction Loss')
    axes[1, 0].set_title('Reconstruction Loss')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # KL divergence and beta
    ax_kl = axes[1, 1]
    ax_kl.plot(epochs, data['train_kl'], label='Train KL', linewidth=2)
    ax_kl.plot(epochs, data['val_kl'], label='Val KL', linewidth=2)
    ax_kl.set_xlabel('Epoch')
    ax_kl.set_ylabel('KL Divergence', color='tab:blue')
    ax_kl.set_title('KL Divergence and Beta')
    ax_kl.tick_params(axis='y', labelcolor='tab:blue')
    ax_kl.legend(loc='upper left')
    ax_kl.grid(True, alpha=0.3)
    
    # Add KL beta curve
    if len(data['kl_beta']) == len(epochs):
        ax_beta = ax_kl.twinx()
        ax_beta.plot(epochs, data['kl_beta'], 'r--', 
                     label='KL Beta', linewidth=2, alpha=0.7)
        ax_beta.set_ylabel('KL Beta', color='tab:red')
        ax_beta.tick_params(axis='y', labelcolor='tab:red')
        ax_beta.legend(loc='upper right')
    
    # Training summary
    axes[1, 2].axis('off')
    best_r = data['val_r'][best_idx]
    summary_text = f"""
Training Summary

Best Epoch: {best_epoch}
Best Val R: {best_r:.4f}

Final Performance (Epoch {epochs[-1]}):
  Train R: {data['train_r'][-1]:.4f}
  Val R:   {data['val_r'][-1]:.4f}
  
  Train Loss: {data['train_loss'][-1]:.4f}
  Val Loss:   {data['val_loss'][-1]:.4f}
  
  Expr Loss:   {data['val_expr'][-1]:.4f}
  Recon Loss:  {data['val_recon'][-1]:.4f}
  KL Div:      {data['val_kl'][-1]:.4f}
  
Total Epochs: {len(epochs)}
    """
    axes[1, 2].text(0.1, 0.5, summary_text, 
                    fontsize=11, family='monospace',
                    verticalalignment='center',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.tight_layout()
    
    # Ensure directory exists
    Path('results/plots').mkdir(parents=True, exist_ok=True)
    plt.savefig('results/plots/training_curves.png', dpi=300, bbox_inches='tight')
    print("Training curves saved: results/plots/training_curves.png")

def main():
    print("="*60)
    print("  Visualize Training Results")
    print("="*60)
    
    print("\nParsing training log...")
    data = parse_training_log()
    
    if data:
        print(f"Parsing complete: {len(data['epochs'])} epochs")
        print(f"  Epoch range: {data['epochs'][0]} - {data['epochs'][-1]}")
        print(f"  Best val R: {max(data['val_r']):.4f}")
        print(f"  Final val R: {data['val_r'][-1]:.4f}")
        
        print("\nPlotting training curves...")
        plot_training_curves(data)
        
        print("\n" + "="*60)
        print("  Complete")
        print("="*60)
    else:
        print("\nError: Log parsing failed")
        print("\nPlease ensure:")
        print("  1. Training log file exists")
        print("  2. Log format is correct")

if __name__ == '__main__':
    main()

