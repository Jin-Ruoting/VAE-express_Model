# train/trainer.py

import torch
from scipy.stats import pearsonr
from train.losses import total_vae_loss
from tqdm import tqdm
import numpy as np

class VAETrainer:
    def __init__(self, model, optimizer, loss_fn, device):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device

    def train_epoch(self, dataloader):
        self.model.train()
        total_loss = 0
        total_recon_loss = 0
        total_kl_loss = 0
        total_expr_loss = 0
        expr_true_all = []
        expr_pred_all = []

        pbar = tqdm(dataloader, desc="Training")
        for x, y in pbar:
            x = x.to(self.device)  # [B, 7, seq_len]
            y = y.to(self.device)

            self.optimizer.zero_grad()
            x_hat, pred_expr, mu, logvar = self.model(x)

            loss, loss_dict = self.loss_fn(x_hat, x, mu, logvar, pred_expr, y)
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()

            total_loss += loss.item()
            total_recon_loss += loss_dict['recon_loss']
            total_kl_loss += loss_dict['kl_loss']
            total_expr_loss += loss_dict['expr_loss']
            
            expr_true_all.extend(y.detach().cpu().numpy().flatten())
            expr_pred_all.extend(pred_expr.detach().cpu().numpy().flatten())
            
            pbar.set_postfix({
                'Loss': f"{loss.item():.4f}",
                'Recon': f"{loss_dict['recon_loss']:.4f}",
                'KL': f"{loss_dict['kl_loss']:.6f}",
                'Expr': f"{loss_dict['expr_loss']:.4f}"
            })

        # 计算相关系数
        if len(expr_true_all) > 1:
            pearson_r = pearsonr(expr_true_all, expr_pred_all)[0]
            if np.isnan(pearson_r):
                pearson_r = 0.0
        else:
            pearson_r = 0.0
            
        avg_loss = total_loss / len(dataloader)
        avg_recon = total_recon_loss / len(dataloader)
        avg_kl = total_kl_loss / len(dataloader)
        avg_expr = total_expr_loss / len(dataloader)
        
        return avg_loss, pearson_r, avg_recon, avg_kl, avg_expr

    def validate(self, dataloader):
        self.model.eval()
        total_loss = 0
        total_recon_loss = 0
        total_kl_loss = 0
        total_expr_loss = 0
        expr_true_all = []
        expr_pred_all = []

        with torch.no_grad():
            for x, y in tqdm(dataloader, desc="Validating"):
                x = x.to(self.device)
                y = y.to(self.device)

                x_hat, pred_expr, mu, logvar = self.model(x)
                loss, loss_dict = self.loss_fn(x_hat, x, mu, logvar, pred_expr, y)

                total_loss += loss.item()
                total_recon_loss += loss_dict['recon_loss']
                total_kl_loss += loss_dict['kl_loss']
                total_expr_loss += loss_dict['expr_loss']
                
                expr_true_all.extend(y.detach().cpu().numpy().flatten())
                expr_pred_all.extend(pred_expr.detach().cpu().numpy().flatten())

        # 计算相关系数
        if len(expr_true_all) > 1:
            pearson_r = pearsonr(expr_true_all, expr_pred_all)[0]
            if np.isnan(pearson_r):
                pearson_r = 0.0
        else:
            pearson_r = 0.0
            
        avg_loss = total_loss / len(dataloader)
        avg_recon = total_recon_loss / len(dataloader)
        avg_kl = total_kl_loss / len(dataloader)
        avg_expr = total_expr_loss / len(dataloader)
        
        return avg_loss, pearson_r, avg_recon, avg_kl, avg_expr

    def fit(self, train_loader, val_loader, num_epochs=50, save_path="best_model.pt", patience=5):
        best_val_loss = float('inf')
        patience_counter = 0
        
        # 学习率调度器
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=3, verbose=True
        )

        print("Starting training...")
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            print("-" * 50)
            
            # 训练
            train_loss, train_r, train_recon, train_kl, train_expr = self.train_epoch(train_loader)
            
            # 验证
            val_loss, val_r, val_recon, val_kl, val_expr = self.validate(val_loader)
            
            # 学习率调度
            scheduler.step(val_loss)
            
            # 打印结果
            print(f"Train - Loss: {train_loss:.4f}, R: {train_r:.3f}, "
                  f"Recon: {train_recon:.4f}, KL: {train_kl:.6f}, Expr: {train_expr:.4f}")
            print(f"Val   - Loss: {val_loss:.4f}, R: {val_r:.3f}, "
                  f"Recon: {val_recon:.4f}, KL: {val_kl:.6f}, Expr: {val_expr:.4f}")

            # 保存最佳模型
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'val_r': val_r
                }, save_path)
                print(f"[SAVED] New best model at epoch {epoch+1}")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"[EARLY STOP] No improvement for {patience} epochs")
                    break
        
        print("Training completed!")