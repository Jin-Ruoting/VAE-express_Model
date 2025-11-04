#!/usr/bin/env python3
"""
Create training visualization from terminal output or tensorboard logs
如果没有训练日志文件，可以从其他来源创建可视化
"""
import re
import sys
from pathlib import Path
import argparse

def parse_from_text_file(text_file):
    """Parse training metrics from a text file (e.g., redirected stdout)"""
    print(f"Parsing metrics from: {text_file}")
    
    epochs = []
    train_loss, val_loss = [], []
    train_r, val_r = [], []
    train_recon, val_recon = [], []
    train_kl, val_kl = [], []
    train_expr, val_expr = [], []
    kl_beta = []
    
    current_epoch = None
    current_kl_beta = None
    
    with open(text_file, 'r') as f:
        for line in f:
            # Extract epoch and KL beta
            epoch_match = re.search(r'\(epoch (\d+)\) KL beta: ([\d.e+-]+)', line)
            if epoch_match:
                current_epoch = int(epoch_match.group(1))
                current_kl_beta = float(epoch_match.group(2))
            
            # Extract training metrics
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
            
            # Extract validation metrics
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
        print("Error: No training data found in file")
        return None
    
    print(f"Found {len(epochs)} epochs")
    
    # Save as standard log format
    output_path = 'logs/train.log'
    Path('logs').mkdir(exist_ok=True)
    
    with open(output_path, 'w') as f:
        for i, epoch in enumerate(epochs):
            f.write(f"(epoch {epoch}) KL beta: {kl_beta[i] if i < len(kl_beta) else 0:.1e}\n")
            f.write(f"Train - Loss: {train_loss[i]:.4f}, R: {train_r[i]:.3f}, "
                   f"Recon: {train_recon[i]:.4f}, KL: {train_kl[i]:.6f}, Expr: {train_expr[i]:.4f}\n")
            f.write(f"Val   - Loss: {val_loss[i]:.4f}, R: {val_r[i]:.3f}, "
                   f"Recon: {val_recon[i]:.4f}, KL: {val_kl[i]:.6f}, Expr: {val_expr[i]:.4f}\n")
    
    print(f"Created log file: {output_path}")
    return output_path

def create_log_from_manual_input():
    """Interactively create a log file from manual input"""
    print("\n" + "="*60)
    print("  Manual Training Log Creation")
    print("="*60)
    print("\nBased on your training output (epoch 30-40):")
    print("Best model: Epoch 36, Val R = 0.685")
    print("\nCreating log file with this information...")
    
    # Your actual training data from the output you provided
    data = [
        (30, 6.0e-06, 1.9209, 0.679, 0.1743, 5.487084, 0.4716, 1.9304, 0.678, 0.1743, 5.003422, 0.4747),
        (31, 6.2e-06, 1.9087, 0.679, 0.1746, 5.142456, 0.4688, 1.9067, 0.682, 0.1744, 4.973609, 0.4685),
        (32, 6.4e-06, 1.9113, 0.680, 0.1744, 4.931769, 0.4695, 1.9035, 0.683, 0.1745, 5.217336, 0.4671),
        (33, 6.6e-06, 1.9066, 0.681, 0.1748, 4.910558, 0.4681, 1.9097, 0.681, 0.1748, 4.943605, 0.4688),
        (34, 6.8e-06, 1.8996, 0.682, 0.1749, 4.680286, 0.4665, 1.9272, 0.678, 0.1749, 4.312480, 0.4740),
        (35, 7.0e-06, 1.8895, 0.683, 0.1754, 4.547753, 0.4640, 1.9008, 0.683, 0.1754, 4.811518, 0.4663),
        (36, 7.2e-06, 1.8832, 0.684, 0.1755, 4.497458, 0.4623, 1.8940, 0.685, 0.1757, 4.907114, 0.4642),
        (37, 7.4e-06, 1.8856, 0.684, 0.1755, 4.396513, 0.4628, 1.9038, 0.682, 0.1757, 4.323472, 0.4675),
        (38, 7.6e-06, 1.8777, 0.685, 0.1762, 4.236691, 0.4609, 1.9157, 0.680, 0.1760, 4.213983, 0.4705),
        (39, 7.8e-06, 1.8683, 0.687, 0.1763, 4.166730, 0.4585, 1.9169, 0.681, 0.1762, 4.498753, 0.4700),
        (40, 8.0e-06, 1.8737, 0.686, 0.1766, 4.125176, 0.4597, 1.9320, 0.677, 0.1764, 3.983086, 0.4746),
    ]
    
    output_path = 'logs/train.log'
    Path('logs').mkdir(exist_ok=True)
    
    with open(output_path, 'w') as f:
        for row in data:
            epoch, kl_beta, train_loss, train_r, train_recon, train_kl, train_expr, \
                val_loss, val_r, val_recon, val_kl, val_expr = row
            
            f.write(f"(epoch {epoch}) KL beta: {kl_beta:.1e}\n")
            f.write(f"Train - Loss: {train_loss:.4f}, R: {train_r:.3f}, "
                   f"Recon: {train_recon:.4f}, KL: {train_kl:.6f}, Expr: {train_expr:.4f}\n")
            f.write(f"Val   - Loss: {val_loss:.4f}, R: {val_r:.3f}, "
                   f"Recon: {val_recon:.4f}, KL: {val_kl:.6f}, Expr: {val_expr:.4f}\n")
    
    print(f"\nCreated log file: {output_path}")
    print(f"Epochs: 30-40 (11 epochs)")
    print(f"Best validation R: 0.685 (Epoch 36)")
    
    return output_path

def main():
    parser = argparse.ArgumentParser(description='Create training log for visualization')
    parser.add_argument('--input', type=str, help='Input text file with training output')
    parser.add_argument('--auto', action='store_true', 
                       help='Automatically create log from your provided training data')
    
    args = parser.parse_args()
    
    if args.input:
        # Parse from provided file
        if not Path(args.input).exists():
            print(f"Error: File not found: {args.input}")
            sys.exit(1)
        log_path = parse_from_text_file(args.input)
    elif args.auto:
        # Use your actual training data
        log_path = create_log_from_manual_input()
    else:
        print("\nNo input provided. Using your training data (epochs 30-40)...")
        log_path = create_log_from_manual_input()
    
    if log_path:
        print("\n" + "="*60)
        print("  Success!")
        print("="*60)
        print(f"\nLog file created: {log_path}")
        print("\nNow you can run:")
        print("  python scripts/visualize_training_results.py")
        print()

if __name__ == '__main__':
    main()

