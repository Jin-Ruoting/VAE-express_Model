# train/trainer.py

import torch
from scipy.stats import pearsonr
from train.losses import total_vae_loss
from tqdm import tqdm
import numpy as np
from torch.nn.utils import clip_grad_norm_

class VAETrainer:
    def __init__(self, model, optimizer, loss_fn, device):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device

    def _safe_nan_to_num(self, t: torch.Tensor) -> torch.Tensor:
        return torch.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0)

    def _safe_pearson(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        # 过滤非有限
        mask = np.isfinite(y_true) & np.isfinite(y_pred)
        y_true = y_true[mask]
        y_pred = y_pred[mask]
        if y_true.size < 2 or np.std(y_true) == 0 or np.std(y_pred) == 0:
            return 0.0
        if pearsonr is not None:
            try:
                return float(pearsonr(y_true, y_pred)[0])
            except Exception:
                pass
        # 退化到 numpy 相关系数
        r = np.corrcoef(y_true, y_pred)[0, 1]
        if not np.isfinite(r):
            return 0.0
        return float(r)

    def train_epoch(self, train_loader, kl_beta=1e-5, max_grad_norm=1.0):
        self.model.train()
        loss_sum = recon_sum = kl_sum = expr_sum = 0.0
        n_batches = 0
        expr_true_all, expr_pred_all = [], []

        for x, y in train_loader:
            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)
            x = self._safe_nan_to_num(x)
            y = self._safe_nan_to_num(y)

            self.optimizer.zero_grad(set_to_none=True)
            out = self.model(x)
            # out 内部若含 NaN/Inf，先清洗
            if isinstance(out, (list, tuple)):
                out = tuple(self._safe_nan_to_num(t) if torch.is_tensor(t) else t for t in out)
            elif torch.is_tensor(out):
                out = self._safe_nan_to_num(out)

            loss, recon, kl, expr, y_pred = total_vae_loss(out, y, kl_beta=kl_beta)

            # 任何一个非有限就跳过该 batch
            if not torch.isfinite(loss):
                continue

            loss.backward()
            if max_grad_norm is not None and max_grad_norm > 0:
                try:
                    clip_grad_norm_(self.model.parameters(), max_grad_norm)
                except Exception:
                    pass
            self.optimizer.step()

            # 累计
            loss_sum += float(loss.detach().cpu())
            recon_sum += float(recon.detach().cpu())
            kl_sum    += float(kl.detach().cpu())
            expr_sum  += float(expr.detach().cpu())
            n_batches += 1

            # 收集表达用于 Pearson
            if y_pred is not None:
                yp = self._safe_nan_to_num(y_pred).detach().cpu().numpy().reshape(-1)
                yt = self._safe_nan_to_num(y).detach().cpu().numpy().reshape(-1)
                expr_pred_all.append(yp)
                expr_true_all.append(yt)

        if n_batches == 0:
            return float('nan'), float('nan'), float('nan'), float('nan'), 0.0

        expr_r = 0.0
        if expr_true_all and expr_pred_all:
            yt = np.concatenate(expr_true_all, axis=0)
            yp = np.concatenate(expr_pred_all, axis=0)
            expr_r = self._safe_pearson(yt, yp)

        return (loss_sum / n_batches,
                expr_r,
                recon_sum / n_batches,
                kl_sum / n_batches,
                expr_sum / n_batches)

    def validate(self, val_loader, kl_beta=1e-5):
        self.model.eval()
        loss_sum = recon_sum = kl_sum = expr_sum = 0.0
        n_batches = 0
        expr_true_all, expr_pred_all = [], []

        with torch.no_grad():
            for x, y in val_loader:
                x = self._safe_nan_to_num(x.to(self.device, non_blocking=True))
                y = self._safe_nan_to_num(y.to(self.device, non_blocking=True))
                out = self.model(x)
                if isinstance(out, (list, tuple)):
                    out = tuple(self._safe_nan_to_num(t) if torch.is_tensor(t) else t for t in out)
                elif torch.is_tensor(out):
                    out = self._safe_nan_to_num(out)

                loss, recon, kl, expr, y_pred = total_vae_loss(out, y, kl_beta=kl_beta)
                if not torch.isfinite(loss):
                    continue

                loss_sum += float(loss.detach().cpu())
                recon_sum += float(recon.detach().cpu())
                kl_sum    += float(kl.detach().cpu())
                expr_sum  += float(expr.detach().cpu())
                n_batches += 1

                if y_pred is not None:
                    yp = self._safe_nan_to_num(y_pred).detach().cpu().numpy().reshape(-1)
                    yt = self._safe_nan_to_num(y).detach().cpu().numpy().reshape(-1)
                    expr_pred_all.append(yp)
                    expr_true_all.append(yt)

        if n_batches == 0:
            return float('nan'), 0.0, float('nan'), float('nan'), float('nan')

        expr_r = 0.0
        if expr_true_all and expr_pred_all:
            yt = np.concatenate(expr_true_all, axis=0)
            yp = np.concatenate(expr_pred_all, axis=0)
            expr_r = self._safe_pearson(yt, yp)

        return (loss_sum / n_batches,
                expr_r,
                recon_sum / n_batches,
                kl_sum / n_batches,
                expr_sum / n_batches)

    def fit(self, train_loader, val_loader, num_epochs=50, save_path="best_model.pt",
            patience=5, kl_beta_max=1e-5, kl_warmup_epochs=50):
        best_val_loss = float('inf')
        patience_counter = 0
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=3, verbose=True
        )

        for epoch in range(num_epochs):
            # 线性退火：从0逐步升到 kl_beta_max
            warmup_ratio = min(1.0, (epoch + 1) / max(1, kl_warmup_epochs))
            kl_beta = kl_beta_max * warmup_ratio

            # 训练/验证时传入当期 kl_beta
            train_loss, train_r, train_recon, train_kl, train_expr = self.train_epoch(train_loader, kl_beta=kl_beta)
            val_loss, val_r, val_recon, val_kl, val_expr = self.validate(val_loader, kl_beta=kl_beta)

            scheduler.step(val_loss)
            print(f"(epoch {epoch+1}) KL beta: {kl_beta:.6g}")
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