import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RoadmapDataset(Dataset):
    def __init__(self, data_dir, sequence_length=2000, min_expr_threshold=0.1):
        self.data_dir = Path(data_dir)
        self.sequence_length = sequence_length
        self.min_expr_threshold = min_expr_threshold
        
        # 加载表达数据
        self.expression_data = self._load_expression_data()
        
        # 构建样本索引
        self.samples = self._build_sample_index()
        
        logger.info(f"Loaded {len(self.samples)} samples")

    def _load_expression_data(self):
        """加载基因表达数据"""
        exp_path = self.data_dir / "exp" / "raw_exp.tsv"
        exp_df = pd.read_csv(exp_path, sep='\t', index_col=0)
        
        # 过滤低表达基因
        exp_df = exp_df[exp_df.max(axis=1) > self.min_expr_threshold]
        
        logger.info(f"Loaded expression data: {exp_df.shape}")
        return exp_df

    def _load_histone_signal(self, eid, mark):
        """加载单个组蛋白修饰信号"""
        npz_path = self.data_dir / "hist" / f"{eid}-{mark}.npz"
        
        if npz_path.exists():
            data = np.load(npz_path)
            # 假设NPZ文件包含信号数组
            if 'signal' in data:
                signal = data['signal']
            elif len(data.files) == 1:
                signal = data[data.files[0]]
            else:
                # 如果有多个数组，取第一个
                signal = data[data.files[0]]
                
            return signal
        else:
            logger.warning(f"Missing file: {npz_path}")
            return np.zeros(self.sequence_length)

    def _build_sample_index(self):
        """构建样本索引"""
        samples = []
        
        cell_types = ['E003', 'E004', 'E005', 'E006', 'E007', 
                      'E016', 'E066', 'E087', 'E114', 'E116', 'E118']
        
        histone_marks = ['H3K4me1', 'H3K4me3', 'H3K9me3', 'H3K27me3', 
                         'H3K36me3', 'H3K27ac', 'H3K9ac']
        
        # 为每个基因和细胞类型组合创建样本
        for gene_id in self.expression_data.index[:1000]:  # 先测试1000个基因
            for eid in cell_types:
                if eid in self.expression_data.columns:
                    expr_val = self.expression_data.loc[gene_id, eid]
                    
                    if not np.isnan(expr_val) and expr_val > self.min_expr_threshold:
                        samples.append({
                            'gene_id': gene_id,
                            'eid': eid,
                            'expression': expr_val
                        })
        
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        gene_id = sample['gene_id']
        eid = sample['eid']
        expression = sample['expression']
        
        # 加载所有组蛋白修饰信号
        histone_marks = ['H3K4me1', 'H3K4me3', 'H3K9me3', 'H3K27me3', 
                         'H3K36me3', 'H3K27ac', 'H3K9ac']
        
        signals = []
        for mark in histone_marks:
            signal = self._load_histone_signal(eid, mark)
            
            # 调整信号长度
            if len(signal) > self.sequence_length:
                start = (len(signal) - self.sequence_length) // 2
                signal = signal[start:start + self.sequence_length]
            elif len(signal) < self.sequence_length:
                padding = self.sequence_length - len(signal)
                signal = np.pad(signal, (padding//2, padding - padding//2), 'constant')
            
            signals.append(signal)
        
        x = torch.FloatTensor(np.stack(signals, axis=0))  # [7, sequence_length]
        y = torch.FloatTensor([np.log1p(expression)])  # log变换
        
        return x, y


def create_dataloaders(config):
    """创建数据加载器"""
    dataset = RoadmapDataset(
        data_dir=config.REFERENCE_DATA_DIR,
        sequence_length=config.SEQUENCE_LENGTH
    )
    
    # 数据分割
    total_size = len(dataset)
    train_size = int(total_size * config.TRAIN_SPLIT)
    val_size = int(total_size * config.VAL_SPLIT)
    test_size = total_size - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size]
    )
    
    # 创建DataLoader
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY
    )
    
    logger.info(f"Created dataloaders - Train: {len(train_dataset)}, "
                f"Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    
    return train_loader, val_loader, test_loader