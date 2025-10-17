import os, sys, argparse, json, math, warnings
from pathlib import Path

# 项目根路径
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

import yaml
import numpy as np
import torch
import pandas as pd

# 第三方可选依赖
try:
    import pyBigWig
except Exception:
    pyBigWig = None
try:
    from scipy.stats import pearsonr
except Exception:
    pearsonr = None

from data.roadmap_dataset import create_dataloaders
from models.vae import VAE
from train.losses import total_vae_loss

warnings.filterwarnings("ignore", category=UserWarning)

def safe_t(t: torch.Tensor) -> torch.Tensor:
    return torch.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0)

def safe_np(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a)
    a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
    return a

def safe_pearson(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = safe_np(y_true).ravel()
    y_pred = safe_np(y_pred).ravel()
    m = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[m]; y_pred = y_pred[m]
    if y_true.size < 2 or np.std(y_true) == 0 or np.std(y_pred) == 0:
        return 0.0
    if pearsonr:
        try:
            return float(pearsonr(y_true, y_pred)[0])
        except Exception:
            pass
    r = np.corrcoef(y_true, y_pred)[0, 1]
    return float(r) if np.isfinite(r) else 0.0

def align_expr_shapes(pred_expr: torch.Tensor, y: torch.Tensor):
    # 统一为 [N, D] 形状，D 默认为1
    pe = pred_expr
    ty = y
    if pe.dim() == 1:
        pe = pe.view(pe.shape[0], 1)
    elif pe.dim() > 2:
        pe = pe.view(pe.shape[0], -1)
    if ty.dim() == 1:
        ty = ty.view(ty.shape[0], 1)
    elif ty.dim() > 2:
        ty = ty.view(ty.shape[0], -1)
    if pe.shape[1] != ty.shape[1]:
        # 若维度不一致，优先将 pred_expr 降到 1 维；否则截到最小公共维
        if pe.shape[1] > 1 and ty.shape[1] == 1:
            pe = pe.mean(dim=1, keepdim=True)
        else:
            d = min(pe.shape[1], ty.shape[1])
            pe = pe[:, :d]
            ty = ty[:, :d]
    return pe, ty

def parse_model_out(out, x: torch.Tensor, y: torch.Tensor):
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
        pred_expr = out.get('pred_expr') or out.get('expr') or out.get('y_pred') or torch.zeros((y.shape[0], 1), device=y.device, dtype=y.dtype)
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

    # 数值清洗
    x_hat = safe_t(x_hat)
    mu = safe_t(mu)
    logvar = safe_t(logvar)
    pred_expr = safe_t(pred_expr)
    # 表达形状对齐
    pred_expr, y = align_expr_shapes(pred_expr, y)
    return x_hat, mu, logvar, pred_expr, y

def call_loss_with_components(out, x, y):
    # 解析模型输出 -> 组装 total_vae_loss 的6个参数
    x_hat, mu, logvar, pred_expr, y_aligned = parse_model_out(out, x, y)
    total, parts = total_vae_loss(x_hat, x, mu, logvar, pred_expr, y_aligned)
    # parts 是 dict，键名可能是 'recon_loss','kl_loss','expr_loss'
    recon = float(parts.get('recon_loss', 0.0))
    kl    = float(parts.get('kl_loss', 0.0))
    expr  = float(parts.get('expr_loss', 0.0))
    return total, recon, kl, expr, pred_expr

def step1_config_check(cfg_path: str):
    print("== STEP1 | 配置与路径核对 ==")
    cfg = yaml.safe_load(open(cfg_path))
    paths = cfg['paths']
    for k, p in paths.items():
        if p:
            print(f"- {k}: {p} | exists={os.path.exists(p)}")
    print(f"- eids: {len(cfg['eids'])} -> {cfg['eids'][:5]} ...")
    print(f"- marks: {len(cfg['marks']['core'])} -> {cfg['marks']['core']}")
    return cfg

def step2_data_smoke(cfg_path: str, device: str):
    print("\n== STEP2 | 数据层冒烟（单 batch 前向） ==")
    # 强制单 worker，避免并发干扰
    os.environ['NUM_WORKERS'] = os.environ.get('NUM_WORKERS', '0')
    train_loader, val_loader, test_loader = create_dataloaders(cfg_path)
    ds = getattr(train_loader, 'dataset', None)
    base_ds = getattr(ds, 'dataset', ds)  # Subset -> RoadmapDataset
    in_ch = getattr(base_ds, 'input_channels', len(getattr(base_ds, 'marks', [])) or 7)
    seq_len = getattr(base_ds, 'seq_len', 2000)
    print(f"- dataset sizes: train={len(train_loader.dataset)}, val={len(val_loader.dataset)}, test={len(test_loader.dataset)}")
    print(f"- model dims: channels={in_ch}, seq_len={seq_len}")
    model = VAE(input_channels=in_ch, latent_dim=64, sequence_length=seq_len).to(device)
    xb, yb = next(iter(train_loader))
    xb, yb = safe_t(xb).to(device), safe_t(yb).to(device)
    with torch.no_grad():
        out = model(xb)
        loss, recon, kl, expr, y_pred = call_loss_with_components(out, xb, yb)
        parts = [loss, torch.as_tensor(recon), torch.as_tensor(kl), torch.as_tensor(expr)]
        ok = all(torch.isfinite(p if torch.is_tensor(p) else torch.as_tensor(p)) for p in parts)
        print(f"- batch: X={tuple(xb.shape)}, finite_loss={ok}, loss={float(loss):.6f}, recon={float(recon):.6f}, kl={float(kl):.6f}, expr={float(expr):.6f}")
        if not ok:
            raise SystemExit("存在 NaN/Inf 的损失项，请检查模型/损失的数值稳定性。")
    return train_loader, val_loader, test_loader, in_ch, seq_len, model

def step3_tiny_train(train_loader, device: str, in_ch: int, seq_len: int):
    print("\n== STEP3 | 10 步小训练（跳过坏 batch + 梯度裁剪） ==")
    model = VAE(input_channels=in_ch, latent_dim=64, sequence_length=seq_len).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=3e-5, weight_decay=1e-4)
    it = iter(train_loader)
    steps, skipped = 0, 0
    while steps < 10:
        xb, yb = next(it)
        xb, yb = safe_t(xb).to(device), safe_t(yb).to(device)
        opt.zero_grad(set_to_none=True)
        out = model(xb)
        loss, recon, kl, expr, _ = call_loss_with_components(out, xb, yb)
        if not torch.isfinite(loss):
            skipped += 1
            continue
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        steps += 1
        print(f"- step {steps}: loss={float(loss):.6f}, recon={float(recon):.6f}, kl={float(kl):.6f}, expr={float(expr):.6f}")
    print(f"- tiny-train done; skipped_bad_batches={skipped}")

def step4_metric_sanity(train_loader, device: str, in_ch: int, seq_len: int):
    print("\n== STEP4 | 指标健壮性（Pearson r 安全计算） ==")
    model = VAE(input_channels=in_ch, latent_dim=64, sequence_length=seq_len).to(device)
    yt_all, yp_all = [], []
    with torch.no_grad():
        for i, (xb, yb) in enumerate(train_loader):
            if i >= 20:
                break
            xb, yb = safe_t(xb).to(device), safe_t(yb).to(device)
            out = model(xb)
            # 直接解析 pred_expr，对齐 y
            _, _, _, pred_expr, y_aligned = parse_model_out(out, xb, yb)
            yp = safe_t(pred_expr).detach().cpu().numpy().ravel()
            yt = safe_t(y_aligned).detach().cpu().numpy().ravel()
            yt_all.append(yt); yp_all.append(yp)
    if yt_all and yp_all:
        yt = np.concatenate(yt_all); yp = np.concatenate(yp_all)
        r = safe_pearson(yt, yp)
        print(f"- pearson_r (first ~20 batches) = {r:.4f}")
    else:
        print("- 无 y_pred 可用于相关性计算（可忽略）")

def step5_signal_probe(cfg):
    print("\n== STEP5 | 信号与统计探针（q99>0 + 窗口可读） ==")
    stats_p = cfg['paths']['stats_json']
    try:
        stats = json.load(open(stats_p))
    except Exception as e:
        print(f"- 跳过：无法读取 stats_json ({e})")
        return
    eids = cfg['eids']
    marks = cfg['marks']['core']
    q99s = []
    for e in eids:
        for m in marks:
            v = stats.get(f"{e}:{m}")
            if isinstance(v, dict) and 'q99' in v:
                q99s.append(v['q99'])
    if q99s:
        q99_min = float(np.nanmin(q99s))
        print(f"- stats q99 min = {q99_min:.6f}  (应 > 0)")
    else:
        print("- stats 中未找到任何 q99 字段，请检查键格式 EID:MARK")

    if pyBigWig is None:
        print("- 未安装 pyBigWig，跳过窗口探针")
        return

    # 采样 3 个 promoter，并测试 1 个 EID × 2 个 marks
    bed_p = cfg['paths']['promoters_bed']
    try:
        bed = pd.read_csv(bed_p, sep='\t', header=None,
                          names=['chrom','start','end','gene_id','score','strand'])
        samp = bed.sample(n=min(3, len(bed)), random_state=0)
    except Exception as e:
        print(f"- 跳过：读取 promoters 失败 ({e})")
        return

    eid = eids[0]
    for m in marks[:2]:
        p = ROOT / "hist" / f"{eid}-{m}.bw"
        if not p.exists():
            print(f"- 缺少 BW: {p}")
            continue
        try:
            bw = pyBigWig.open(str(p))
            for _, r in samp.iterrows():
                v = np.array(bw.values(str(r.chrom), int(r.start), int(r.end)))
                cnt = int(np.isfinite(v).sum())
                print(f"- {eid}-{m} {r.chrom}:{int(r.start)}-{int(r.end)} finite_vals={cnt}")
            bw.close()
        except Exception as e:
            print(f"- 读取 BW 失败 {p}: {e}")

def main():
    ap = argparse.ArgumentParser(description="VAE-express 冒烟自检")
    ap.add_argument('--config', default='config/config.yaml')
    args = ap.parse_args()

    # 强制 DataLoader 使用单 worker
    os.environ.setdefault('NUM_WORKERS', '0')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    cfg = step1_config_check(args.config)
    tr, va, te, in_ch, seq_len, _ = step2_data_smoke(args.config, device)
    step3_tiny_train(tr, device, in_ch, seq_len)
    step4_metric_sanity(tr, device, in_ch, seq_len)
    step5_signal_probe(cfg)
    print("\n[OK] 冒烟自检完成")

if __name__ == '__main__':
    main()