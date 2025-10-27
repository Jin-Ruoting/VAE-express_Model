# train/trainer.py

import os
import torch
from scipy.stats import pearsonr
from torch.utils.data import DataLoader
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

    def _align_expr_shapes(self, pred_expr: torch.Tensor, y: torch.Tensor):
        pe, ty = pred_expr, y
        if pe.dim() == 1: pe = pe.view(pe.shape[0], 1)
        elif pe.dim() > 2: pe = pe.view(pe.shape[0], -1)
        if ty.dim() == 1: ty = ty.view(ty.shape[0], 1)
        elif ty.dim() > 2: ty = ty.view(ty.shape[0], -1)
        if pe.shape[1] != ty.shape[1]:
            if pe.shape[1] > 1 and ty.shape[1] == 1:
                pe = pe.mean(dim=1, keepdim=True)
            else:
                d = min(pe.shape[1], ty.shape[1])
                pe = pe[:, :d]; ty = ty[:, :d]
        return pe, ty

    def _parse_model_out(self, out, x: torch.Tensor, y: torch.Tensor):
        # 支持 tuple/list/dict/单张量
        x_hat = mu = logvar = pred_expr = None
        if isinstance(out, (list, tuple)):
            if len(out) >= 4:
                x_hat, mu, logvar, pred_expr = out[0], out[1], out[2], out[3]
            elif len(out) == 3:
                x_hat, mu, logvar = out
                pred_expr = torch.zeros((y.shape[0], 1), device=y.device, dtype=y.dtype)
            elif len(out) == 2:
                x_hat, mu = out
                logvar = torch.zeros_like(mu)
                pred_expr = torch.zeros((y.shape[0], 1), device=y.device, dtype=y.dtype)
            elif len(out) == 1:
                x_hat = out[0]
                mu = torch.zeros((x.shape[0], 1), device=x.device, dtype=x.dtype)
                logvar = torch.zeros_like(mu)
                pred_expr = torch.zeros((y.shape[0], 1), device=y.device, dtype=y.dtype)
        elif isinstance(out, dict):
            x_hat = out.get('x_hat') or out.get('recon') or out.get('recon_x') or out.get('x_recon')
            mu = out.get('mu')
            logvar = out.get('logvar') or out.get('log_var') or out.get('log_sigma2')
            pred_expr = out.get('pred_expr') or out.get('expr') or out.get('y_pred') \
                        or torch.zeros((y.shape[0], 1), device=y.device, dtype=y.dtype)
            if x_hat is None:
                raise RuntimeError("模型输出字典中缺少 x_hat/recon。")
            if mu is None:
                mu = torch.zeros((x.shape[0], 1), device=x.device, dtype=x.dtype)
            if logvar is None:
                logvar = torch.zeros_like(mu)
        elif torch.is_tensor(out):
            x_hat = out
            mu = torch.zeros((x.shape[0], 1), device=x.device, dtype=x.dtype)
            logvar = torch.zeros_like(mu)
            pred_expr = torch.zeros((y.shape[0], 1), device=y.device, dtype=y.dtype)
        else:
            raise TypeError(f"不支持的模型输出类型: {type(out)}")

        # 数值清洗与形状对齐
        x_hat = self._safe_nan_to_num(x_hat)
        mu = self._safe_nan_to_num(mu)
        logvar = self._safe_nan_to_num(logvar)
        pred_expr = self._safe_nan_to_num(pred_expr)
        pred_expr, y = self._align_expr_shapes(pred_expr, y)
        return x_hat, mu, logvar, pred_expr, y

    def _compute_loss(self, out, x, y, kl_beta):
        """
        兼容 total_vae_loss 的两类签名：
        1) 旧版: total_vae_loss(out, y, kl_beta=...) 或位置参数
        2) 新版: total_vae_loss(x_hat, x, mu, logvar, pred_expr, y) -> (total, dict)
        返回: (loss_tensor, recon_tensor, kl_tensor, expr_tensor, y_pred_tensor或None)
        兼容旧/新签名。新签名时，注入权重并对 KL 做 warmup:
        kw = base_kw * (kl_beta / kl_beta_max)
        kl_beta_max 通过环境变量 KL_BETA_MAX（默认 1e-5）控制，需与 fit() 打印一致
        """
        # 优先尝试旧签名
        try:
            return total_vae_loss(out, y, kl_beta=kl_beta)
        except TypeError:
            try:
                return total_vae_loss(out, y, kl_beta)
            except TypeError:
                pass

        # 解析为新签名
        x_hat, mu, logvar, pred_expr, y_aligned = self._parse_model_out(out, x, y)

        # 读取权重 + KL warmup
        rw = float(os.getenv('RECON_W', '0.01'))
        base_kw = float(os.getenv('KL_W', '0.02'))
        kl_beta_max = float(os.getenv('KL_BETA_MAX', '1e-5'))
        sched = os.getenv('KL_SCHEDULE', 'linear')  # linear / none
        if sched == 'linear' and kl_beta_max > 0:
            kw = base_kw * max(0.0, min(1.0, kl_beta / kl_beta_max))
        else:
            kw = base_kw
        ew = float(os.getenv('EXPR_W', '3.0'))

        total, parts = total_vae_loss(
            x_hat, x, mu, logvar, pred_expr, y_aligned,
            recon_weight=rw, kl_weight=kw, expr_weight=ew
        )
        device = total.device
        recon = torch.tensor(parts.get('recon_loss', 0.0), device=device, dtype=total.dtype)
        kl    = torch.tensor(parts.get('kl_loss', 0.0),   device=device, dtype=total.dtype)
        expr  = torch.tensor(parts.get('expr_loss', 0.0), device=device, dtype=total.dtype)
        return total, recon, kl, expr, pred_expr

    def train_epoch(self, train_loader, kl_beta=1e-5, max_grad_norm=1.0):
        self.model.train()
        loss_sum = recon_sum = kl_sum = expr_sum = 0.0
        n_batches = 0
        expr_true_all, expr_pred_all = [], []

        for x, y in train_loader:
            x = self._safe_nan_to_num(x.to(self.device, non_blocking=True))
            y = self._safe_nan_to_num(y.to(self.device, non_blocking=True))

            self.optimizer.zero_grad(set_to_none=True)
            out = self.model(x)
            loss, recon, kl, expr, y_pred = self._compute_loss(out, x, y, kl_beta)

            # 任何一个非有限就跳过该 batch
            if not torch.isfinite(loss):
                continue

            loss.backward()
            if max_grad_norm and max_grad_norm > 0:
                clip_grad_norm_(self.model.parameters(), max_grad_norm)
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
            return float('nan'), 0.0, float('nan'), float('nan'), float('nan')

        expr_r = 0.0
        if expr_true_all and expr_pred_all:
            yt = np.concatenate(expr_true_all, axis=0)
            yp = np.concatenate(expr_pred_all, axis=0)
            m = np.isfinite(yt) & np.isfinite(yp)
            yt, yp = yt[m], yp[m]
            if yt.size >= 2 and np.std(yt) > 0 and np.std(yp) > 0:
                expr_r = float(np.corrcoef(yt, yp)[0, 1])

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
                loss, recon, kl, expr, y_pred = self._compute_loss(out, x, y, kl_beta)

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
            m = np.isfinite(yt) & np.isfinite(yp)
            yt, yp = yt[m], yp[m]
            if yt.size >= 2 and np.std(yt) > 0 and np.std(yp) > 0:
                expr_r = float(np.corrcoef(yt, yp)[0, 1])

        return (loss_sum / n_batches,
                expr_r,
                recon_sum / n_batches,
                kl_sum / n_batches,
                expr_sum / n_batches)

    def _set_requires_grad(self, module, flag: bool):
        for p in module.parameters(): p.requires_grad = flag

    def fit(self, train_loader, val_loader, num_epochs, save_path, patience=10,
            kl_beta_max=1e-4, kl_warmup_epochs=10, verbose=True, monitor='r'):
        """
        monitor: 'r' 用验证集 Pearson R 作为改进标准；'loss' 用验证集损失
        """
        best_metric = -float('inf') if monitor == 'r' else float('inf')
        wait = 0
        freeze_epochs = int(os.getenv('FREEZE_EPOCHS', '0'))

        for epoch in range(1, num_epochs + 1):
            # 两阶段：前 freeze_epochs 只训练表达头
            if freeze_epochs > 0:
                net = self.model.module if isinstance(self.model, torch.nn.DataParallel) else self.model
                if epoch == 1 and freeze_epochs > 0:
                    print(f"[Freeze] Freeze encoder+decoder for {freeze_epochs} epochs, train regressor only")
                if epoch <= freeze_epochs:
                    self._set_requires_grad(net.encoder, False)
                    self._set_requires_grad(net.decoder, False)
                    self._set_requires_grad(net.regressor, True)
                elif epoch == freeze_epochs + 1:
                    print("[Unfreeze] Unfreeze all modules")
                    self._set_requires_grad(net.encoder, True)
                    self._set_requires_grad(net.decoder, True)
                    self._set_requires_grad(net.regressor, True)

            kl_beta = min(kl_beta_max, kl_beta_max * epoch / max(1, kl_warmup_epochs))
            train_loss, train_r, train_recon, train_kl, train_expr = self.train_epoch(train_loader, kl_beta=kl_beta)
            val_loss, val_r, val_recon, val_kl, val_expr = self.validate(val_loader, kl_beta=kl_beta)

            if verbose:
                print(f"(epoch {epoch}) KL beta: {kl_beta:.1e}")
                print(f"Train - Loss: {train_loss:.4f}, R: {train_r:.3f}, Recon: {train_recon:.4f}, KL: {train_kl:.6f}, Expr: {train_expr:.4f}")
                print(f"Val   - Loss: {val_loss:.4f}, R: {val_r:.3f}, Recon: {val_recon:.4f}, KL: {val_kl:.6f}, Expr: {val_expr:.4f}")

            metric = val_r if monitor == 'r' else val_loss
            is_better = (metric > best_metric) if monitor == 'r' else (metric < best_metric)

            if is_better:
                best_metric = metric
                wait = 0
                torch.save(self.model.state_dict() if not isinstance(self.model, torch.nn.DataParallel) else self.model.module.state_dict(), save_path)
                print("[SAVED] New best model" + (f" (val R={val_r:.3f})" if monitor=='r' else f" (val loss={val_loss:.4f})"))
            else:
                wait += 1
                if wait >= patience:
                    print("[EARLY STOP] No improvement for %d epochs" % patience)
                    break
        
        print("Training completed!")