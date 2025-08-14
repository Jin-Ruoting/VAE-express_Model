#!/usr/bin/env python3
"""
测试VAE模型和数据加载器
"""
import sys
sys.path.append('/data/zqjinruoting/VAE-express_Model')

import torch
from models.vae import VAE
from config.roadmap_config import RoadmapVAEConfig
from data.roadmap_dataset import create_dataloaders

def test_model_architecture():
    """测试模型架构"""
    print("=== Testing Model Architecture ===")
    
    config = RoadmapVAEConfig()
    model = VAE(
        input_channels=config.INPUT_CHANNELS,
        latent_dim=config.LATENT_DIM,
        sequence_length=config.SEQUENCE_LENGTH
    )
    
    # 创建假数据测试
    batch_size = 4
    x = torch.randn(batch_size, config.INPUT_CHANNELS, config.SEQUENCE_LENGTH)
    
    print(f"Input shape: {x.shape}")
    
    # 测试前向传播
    try:
        x_hat, expr_pred, mu, logvar = model(x)
        
        print(f"Reconstructed shape: {x_hat.shape}")
        print(f"Expression prediction shape: {expr_pred.shape}")
        print(f"Latent mu shape: {mu.shape}")
        print(f"Latent logvar shape: {logvar.shape}")
        print("[SUCCESS] Model architecture test passed!")
        return True
        
    except Exception as e:
        print(f"[ERROR] Model test failed: {e}")
        return False

def test_data_loading():
    """测试数据加载"""
    print("\n=== Testing Data Loading ===")
    
    config = RoadmapVAEConfig()
    
    try:
        train_loader, val_loader, test_loader = create_dataloaders(config)
        
        # 测试一个batch
        for x, y in train_loader:
            print(f"Data batch shape: {x.shape}")
            print(f"Expression shape: {y.shape}")
            print(f"Data range: [{x.min():.3f}, {x.max():.3f}]")
            print(f"Expression range: [{y.min():.3f}, {y.max():.3f}]")
            break
        
        print(f"[SUCCESS] Data loading test passed!")
        print(f"Train batches: {len(train_loader)}")
        print(f"Val batches: {len(val_loader)}")
        print(f"Test batches: {len(test_loader)}")
        return True
        
    except Exception as e:
        print(f"[ERROR] Data loading test failed: {e}")
        return False

def test_training_step():
    """测试训练步骤"""
    print("\n=== Testing Training Step ===")
    
    config = RoadmapVAEConfig()
    
    try:
        # 初始化模型
        model = VAE(
            input_channels=config.INPUT_CHANNELS,
            latent_dim=config.LATENT_DIM,
            sequence_length=config.SEQUENCE_LENGTH
        )
        
        # 初始化优化器
        optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
        
        # 创建假数据
        x = torch.randn(2, config.INPUT_CHANNELS, config.SEQUENCE_LENGTH)
        y = torch.randn(2, 1)
        
        # 前向传播
        x_hat, expr_pred, mu, logvar = model(x)
        
        # 计算损失
        from train.losses import total_vae_loss
        loss, loss_dict = total_vae_loss(x_hat, x, mu, logvar, expr_pred, y)
        
        print(f"Total loss: {loss.item():.4f}")
        print(f"Recon loss: {loss_dict['recon_loss']:.4f}")
        print(f"KL loss: {loss_dict['kl_loss']:.6f}")
        print(f"Expr loss: {loss_dict['expr_loss']:.4f}")
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        print("[SUCCESS] Training step test passed!")
        return True
        
    except Exception as e:
        print(f"[ERROR] Training step test failed: {e}")
        return False

if __name__ == "__main__":
    print("Starting VAE model tests...\n")
    
    # 运行所有测试
    tests = [
        test_model_architecture,
        test_data_loading, 
        test_training_step
    ]
    
    passed = 0
    for test_func in tests:
        if test_func():
            passed += 1
    
    print(f"\n{'='*50}")
    print(f"Test Results: {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        print("All tests passed! Ready to start training.")
    else:
        print("Some tests failed. Please check the errors above.")
    print("="*50)