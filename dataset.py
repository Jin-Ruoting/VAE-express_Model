# data/dataset.py

import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

class PromoterExpressionDataset(Dataset):
    def __init__(self, data_dir, label_csv, file_list=None, transform=None):
        """
        data_dir: 目录，包含 .npy 文件
        label_csv: 表达标签CSV，包含 'gene_id', 'expression'
        file_list: 限定使用的样本id列表
        transform: 数据预处理函数
        """
        self.data_dir = data_dir
        self.labels = pd.read_csv(label_csv)
        self.labels.set_index("gene_id", inplace=True)
        self.transform = transform

        # 自动识别.npy文件列表
        all_files = [f for f in os.listdir(data_dir) if f.endswith('.npy')]
        if file_list:
            with open(file_list, 'r') as f:
                selected_ids = set([line.strip() for line in f])
                self.files = [f for f in all_files if f.replace('.npy', '') in selected_ids]
        else:
            self.files = all_files

        # 保留既有输入又有标签的样本
        self.valid_files = [
            f for f in self.files
            if f.replace('.npy', '') in self.labels.index
        ]

    def __len__(self):
        return len(self.valid_files)

    def __getitem__(self, idx):
        file = self.valid_files[idx]
        gene_id = file.replace('.npy', '')
        x = np.load(os.path.join(self.data_dir, file))  # shape: [C, 60]
        y = self.labels.loc[gene_id, 'expression']

        if self.transform:
            x = self.transform(x)

        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)

        return x, y