#!/usr/bin/env python3
"""
VAE模型训练脚本
"""
import sys
sys.path.append('/data/zqjinruoting/VAE-express_Model')

import torch
import torch.optim as optim
from models.vae import VAE
from train.trainer import VAETrainer
from train.losses import total_vae_loss
from config.roadmap_config import RoadmapVAEConfig
from data.roadmap_dataset import create_dataloaders

def main():
    print("=== Starting VAE Training ===")
    
    # 加载配置
    config = RoadmapVAEConfig()
    config.print_config()
    config.create_directories()
    
    # 设置随机种子
    torch.manual_seed(config.RANDOM_SEED)
    
    # 创建数据加载器
    print("Loading data...")
    train_loader, val_loader, test_loader = create_dataloaders(config)
    
    # 初始化模型
    print("Initializing model...")
    model = VAE(
        input_channels=config.INPUT_CHANNELS,
        latent_dim=config.LATENT_DIM,
        sequence_length=config.SEQUENCE_LENGTH
    )
    
    # 优化器
    optimizer = optim.Adam(
        model.parameters(), 
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY
    )
    
    # 训练器
    trainer = VAETrainer(
        model=model,
        optimizer=optimizer,
        loss_fn=total_vae_loss,
        device=config.DEVICE
    )
    
    # 开始训练
    save_path = config.get_model_save_path(best=True)
    trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=config.NUM_EPOCHS,
        save_path=save_path,
        patience=config.PATIENCE,
        kl_beta_max=config.KL_WEIGHT,
        kl_warmup_epochs=config.KL_ANNEAL_EPOCHS
    )
    
    print("Training completed!")

if __name__ == "__main__":
    main()