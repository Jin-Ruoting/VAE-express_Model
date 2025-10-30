import numpy as np, torch, yaml
from data.roadmap_dataset import create_dataloaders

tr, _, _ = create_dataloaders('config/config.yaml')
xb, yb = next(iter(tr))
y = yb.detach().cpu().numpy().reshape(-1)
print('y stats: min=', float(y.min()), 'max=', float(y.max()), 'mean=', float(y.mean()))
# 粗判：log2(RPKM+1) 
q = np.quantile(y, [0, 0.5, 0.9, 0.99])
print('y quantiles:', q)
# 还原近似 RPKM 分布（仅用于 sanity check）
rpkm_hat = np.power(2.0, y) - 1.0
print('rpkm_hat stats: min=', float(rpkm_hat.min()), 'max=', float(rpkm_hat.max()), 'mean=', float(rpkm_hat.mean()))