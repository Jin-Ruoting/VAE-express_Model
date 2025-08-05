# train/trainer.py

import torch
from scipy.stats import pearsonr
from train.losses import total_vae_loss

class VAETrainer:
    def __init__(self, model, optimizer, loss_fn, device):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device

    def train_epoch(self, dataloader):
        self.model.train()
        total_loss = 0
        expr_true_all = []
        expr_pred_all = []

        for x, y in dataloader:
            x = x.to(self.device)  # [B, C, 60]
            y = y.to(self.device)

            self.optimizer.zero_grad()
            x_hat, pred_expr, mu, logvar = self.model(x)

            loss, loss_dict = self.loss_fn(x_hat, x, mu, logvar, pred_expr, y)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            expr_true_all.extend(y.detach().cpu().numpy())
            expr_pred_all.extend(pred_expr.detach().cpu().numpy())

        pearson = pearsonr(expr_true_all, expr_pred_all)[0]
        avg_loss = total_loss / len(dataloader)
        return avg_loss, pearson

    def validate(self, dataloader):
        self.model.eval()
        total_loss = 0
        expr_true_all = []
        expr_pred_all = []

        with torch.no_grad():
            for x, y in dataloader:
                x = x.to(self.device)
                y = y.to(self.device)

                x_hat, pred_expr, mu, logvar = self.model(x)
                loss, loss_dict = self.loss_fn(x_hat, x, mu, logvar, pred_expr, y)

                total_loss += loss.item()
                expr_true_all.extend(y.detach().cpu().numpy())
                expr_pred_all.extend(pred_expr.detach().cpu().numpy())

        pearson = pearsonr(expr_true_all, expr_pred_all)[0]
        avg_loss = total_loss / len(dataloader)
        return avg_loss, pearson

    def fit(self, train_loader, val_loader, num_epochs=50, save_path="best_model.pt", patience=5):
        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(num_epochs):
            train_loss, train_r = self.train_epoch(train_loader)
            val_loss, val_r = self.validate(val_loader)

            print(f"[Epoch {epoch+1}] Train Loss: {train_loss:.4f}, R: {train_r:.3f} | Val Loss: {val_loss:.4f}, R: {val_r:.3f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(self.model.state_dict(), save_path)
                print("✅ Saved new best model.")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print("⏹️ Early stopping triggered.")
                    break