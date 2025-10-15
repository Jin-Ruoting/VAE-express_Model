#!/usr/bin/env python3
"""
VAE模型训练脚本
"""
import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import torch
import yaml
from data.roadmap_dataset import create_dataloaders
from models.vae import VAE
from train.trainer import VAETrainer
from train.losses import total_vae_loss

# 合理使用CPU：限制BLAS线程，避免和DataLoader互相抢
torch.set_num_threads(int(os.getenv("OMP_NUM_THREADS", "8")))
torch.set_num_interop_threads(1)
torch.backends.cudnn.benchmark = True

def ensure_dirs():
    Path('results/models').mkdir(parents=True, exist_ok=True)
    Path('results/logs').mkdir(parents=True, exist_ok=True)
    Path('results/plots').mkdir(parents=True, exist_ok=True)

def main():
    ensure_dirs()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    cfg = yaml.safe_load(open('config/config.yaml'))
    marks = list(cfg['marks']['core']) + (cfg['marks'].get('extra', []) if cfg.get('use_extra') else [])
    in_channels = len(marks)
    seq_len = infer_seq_len_from_promoters(cfg['paths']['promoters_bed'], default_len=2000)

    train_loader, val_loader, test_loader = create_dataloaders('config/config.yaml')

    # 不从 DataLoader 提前取 batch
    n_train = len(getattr(train_loader, 'dataset', []))
    n_val = len(getattr(val_loader, 'dataset', []))
    n_test = len(getattr(test_loader, 'dataset', []))
    print(f"Dataset sizes -> train={n_train}, val={n_val}, test={n_test}, channels={in_channels}, seq_len≈{seq_len}")
    if n_train == 0:
        raise RuntimeError("训练集样本为0。")

    model = VAE(input_channels=in_channels, latent_dim=64, sequence_length=seq_len)
    optim = torch.optim.Adam(model.parameters(), lr=5e-5, weight_decay=1e-4)

    # 多卡与设备放置
    if device == 'cuda':
        if torch.cuda.device_count() > 1:
            print(f"Using DataParallel over {torch.cuda.device_count()} GPUs")
            model = torch.nn.DataParallel(model)
        model = model.to('cuda')
    else:
        model = model.to('cpu')

    trainer = VAETrainer(model=model, optimizer=optim, loss_fn=total_vae_loss, device=device)

    best_path = 'results/models/vae_promoter_only_best.pt'
    trainer.fit(train_loader=train_loader,
                val_loader=val_loader,
                num_epochs=100,
                save_path=best_path,
                patience=20,
                kl_beta_max=1e-5,
                kl_warmup_epochs=50)

    print("Training completed!")

if __name__ == '__main__':
    main()