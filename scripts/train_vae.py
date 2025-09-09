#!/usr/bin/env python3
"""
VAE模型训练脚本
"""
import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import torch
from data.roadmap_dataset import create_dataloaders
from models.vae import VAE
from train.trainer import VAETrainer
from train.losses import total_vae_loss

def ensure_dirs():
    Path('results/models').mkdir(parents=True, exist_ok=True)
    Path('results/logs').mkdir(parents=True, exist_ok=True)
    Path('results/plots').mkdir(parents=True, exist_ok=True)

def main():
    ensure_dirs()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # 使用我们新Dataset封装
    train_loader, val_loader, test_loader = create_dataloaders('config/config.yaml')

    # 根据Dataset通道与长度更新模型输入
    sample_X, _ = next(iter(train_loader))
    C, L = sample_X.shape[1], sample_X.shape[2]
    print(f"Input channels={C}, seq_len={L}")

    model = VAE(input_channels=C, latent_dim=64, sequence_length=L).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=5e-5, weight_decay=1e-4)

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