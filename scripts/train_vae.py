#!/usr/bin/env python3
"""
VAE模型训练脚本
"""
import os
import sys
import argparse
from pathlib import Path
from datetime import datetime
sys.path.append(str(Path(__file__).resolve().parents[1]))

import torch
import yaml
import pandas as pd
from data.roadmap_dataset import create_dataloaders
from models.vae import VAE
from train.trainer import VAETrainer
from train.losses import total_vae_loss

# 合理使用CPU：限制BLAS线程，避免和DataLoader互相抢
torch.set_num_threads(int(os.getenv("OMP_NUM_THREADS", "8")))
torch.set_num_interop_threads(1)
torch.backends.cudnn.benchmark = True

def get_run_dir():
    """
    获取本次运行的结果目录
    格式: results/YYYYMMDD-HHMM_mode/
    mode 从环境变量 RUN_NAME 读取，默认为 'run'
    """
    timestamp = datetime.now().strftime("%Y%m%d-%H%M")
    run_name = os.getenv('RUN_NAME', 'run')
    run_dir = Path(f'results/{timestamp}_{run_name}')
    return run_dir

def ensure_dirs(run_dir):
    """创建运行目录结构"""
    (run_dir / 'models').mkdir(parents=True, exist_ok=True)
    (run_dir / 'logs').mkdir(parents=True, exist_ok=True)
    (run_dir / 'plots').mkdir(parents=True, exist_ok=True)
    
    # 同时保持旧的 results/ 结构作为软链接（兼容性）
    Path('results/models').mkdir(parents=True, exist_ok=True)
    Path('results/logs').mkdir(parents=True, exist_ok=True)
    Path('results/plots').mkdir(parents=True, exist_ok=True)
    
    return run_dir

def infer_seq_len_from_promoters(prom_path: str, default_len: int = 2000) -> int:
    """从promoters BED推断窗口长度，失败时返回默认值"""
    try:
        df = pd.read_csv(prom_path, sep='\t', header=None, nrows=1,
                         names=['chrom','start','end','gene_id','score','strand'])
        L = int(df.iloc[0]['end']) - int(df.iloc[0]['start'])
        return L if L > 0 else default_len
    except Exception:
        return default_len

def parse_args():
    ap = argparse.ArgumentParser(description="Train VAE with optional k-fold cross-validation.")
    ap.add_argument("--config", default="config/config.yaml", help="Path to config YAML.")
    ap.add_argument("--num-folds", type=int, default=int(os.getenv("NUM_FOLDS", "1")),
                    help="Number of folds for k-fold CV (<=1 disables).")
    ap.add_argument("--fold-idx", type=int, default=int(os.getenv("FOLD_IDX", "0")),
                    help="Fold index (0-based).")
    ap.add_argument("--fold-val-ratio", type=float, default=float(os.getenv("FOLD_VAL_RATIO", "0.5")),
                    help="Within held-out fold, fraction used for validation (rest for test).")
    ap.add_argument("--fold-seed", type=int, default=int(os.getenv("FOLD_SEED", "42")),
                    help="Random seed for k-fold shuffling.")
    ap.add_argument("--fold-split-mode", default=os.getenv("FOLD_SPLIT_MODE", "random"),
                    choices=["random", "chrom"],
                    help="How to split held-out fold into val/test: random or chrom.")
    return ap.parse_args()


def main():
    args = parse_args()
    # 获取本次运行的目录
    run_dir = get_run_dir()
    ensure_dirs(run_dir)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print("="*70)
    print("  VAE 模型训练")
    print("="*70)
    print(f"运行目录: {run_dir}")
    print(f"设备: {device}")
    print(f"运行名称: {os.getenv('RUN_NAME', 'run')}")
    print("="*70)
    print()

    cfg = yaml.safe_load(open(args.config))
    marks = list(cfg['marks']['core']) + (cfg['marks'].get('extra', []) if cfg.get('use_extra') else [])
    in_channels = len(marks)
    seq_cfg = cfg.get('sequence', {})
    if seq_cfg.get('num_bins'):
        seq_len = int(seq_cfg['num_bins'])
    else:
        seq_len = seq_cfg.get('promoter_bp',
                  infer_seq_len_from_promoters(cfg['paths']['promoters_bed'], default_len=2000))

    if args.num_folds and args.num_folds > 1:
        print(f"[CV] Using fold {args.fold_idx}/{args.num_folds-1}, "
              f"val_ratio={args.fold_val_ratio}, split={args.fold_split_mode}")
    # 构建 DataLoader
    train_loader, val_loader, test_loader = create_dataloaders(
        args.config,
        num_folds=args.num_folds,
        fold_idx=args.fold_idx,
        fold_val_ratio=args.fold_val_ratio,
        fold_seed=args.fold_seed,
        fold_split_mode=args.fold_split_mode,
    )

    # 不从 DataLoader 提前取 batch
    n_train = len(getattr(train_loader, 'dataset', []))
    n_val = len(getattr(val_loader, 'dataset', []))
    n_test = len(getattr(test_loader, 'dataset', []))
    print(f"Dataset sizes -> train={n_train}, val={n_val}, test={n_test}, channels={in_channels}, seq_len≈{seq_len}")
    if n_train == 0:
        raise RuntimeError("训练集样本为0。")

    # 构建模型
    model = VAE(input_channels=in_channels, latent_dim=64, sequence_length=seq_len)

    # 多卡与设备放置
    if device == 'cuda':
        if torch.cuda.device_count() > 1:
            print(f"Using DataParallel over {torch.cuda.device_count()} GPUs")
            model = torch.nn.DataParallel(model)
        model = model.to('cuda')
    else:
        model = model.to('cpu')

    # 优化器放在 DataParallel 之后创建
    optim = torch.optim.Adam(model.parameters(), lr=5e-5, weight_decay=1e-4)

    trainer = VAETrainer(model=model, optimizer=optim, loss_fn=total_vae_loss, device=device)

    # 保存到本次运行的目录
    best_path = run_dir / 'models' / 'vae_best.pt'
    last_path = run_dir / 'models' / 'vae_last.pt'
    
    # 同时保存到旧路径（兼容性）
    legacy_best_path = 'results/models/vae_promoter_only_best.pt'
    
    trainer.fit(train_loader=train_loader,
                val_loader=val_loader,
                num_epochs=100,
                save_path=str(best_path),
                patience=20,
                kl_beta_max=1e-5,
                kl_warmup_epochs=50)

    # 保存最终模型
    if device == 'cuda' and isinstance(model, torch.nn.DataParallel):
        torch.save(model.module.state_dict(), last_path)
    else:
        torch.save(model.state_dict(), last_path)
    
    # 复制最佳模型到兼容路径
    import shutil
    shutil.copy(best_path, legacy_best_path)
    
    print()
    print("="*70)
    print("  训练完成！")
    print("="*70)
    print(f"最佳模型: {best_path}")
    print(f"最终模型: {last_path}")
    print(f"兼容路径: {legacy_best_path}")
    print("="*70)

if __name__ == '__main__':
    main()