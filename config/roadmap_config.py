import torch
from pathlib import Path

class RoadmapVAEConfig:
    """
    Roadmap Epigenomics数据的VAE模型配置
    """
    
    # === 路径配置 ===
    PROJECT_ROOT = Path("/data/zqjinruoting/VAE-express_Model")
    REFERENCE_DATA_DIR = Path("/data/zqjinruoting/Promoter_Model/chromoformer/preprocessing")
    
    # === 数据参数 ===
    # Roadmap Epigenomics 数据规格
    INPUT_CHANNELS = 7  # 7种组蛋白修饰标记
    SEQUENCE_LENGTH = 2000  # 序列长度 (可根据实际数据调整)
    LATENT_DIM = 64  # 潜在空间维度
    
    # 细胞类型和组蛋白标记
    CELL_TYPES = [
        'E003', 'E004', 'E005', 'E006', 'E007',
        'E016', 'E066', 'E087', 'E114', 'E116', 'E118'
    ]
    
    HISTONE_MARKS = [
        'H3K4me1',   # Enhancer mark
        'H3K4me3',   # Promoter mark  
        'H3K9me3',   # Heterochromatin mark
        'H3K27me3',  # Polycomb repressive mark
        'H3K36me3',  # Gene body mark
        'H3K27ac',   # Active enhancer mark
        'H3K9ac'     # Active promoter mark
    ]
    
    # === 模型架构参数 ===
    ENCODER_HIDDEN_DIMS = [64, 128, 256]  # CNN通道数
    DECODER_HIDDEN_DIMS = [256, 128, 64]  # 解码器通道数
    REGRESSOR_HIDDEN_DIMS = [128, 64]     # 表达预测器隐藏层
    
    KERNEL_SIZE = 15     # 卷积核大小
    POOL_SIZE = 4        # 池化大小
    DROPOUT_RATE = 0.3   # Dropout比率
    
    # === 训练参数 ===
    BATCH_SIZE = 16      # 批大小 (根据GPU内存调整)
    LEARNING_RATE = 5e-5 # 学习率
    NUM_EPOCHS = 150     # 训练轮数
    PATIENCE = 20        # 早停耐心值
    
    # 优化器参数
    WEIGHT_DECAY = 1e-4
    BETAS = (0.9, 0.999)
    
    # 学习率调度
    LR_SCHEDULER = "ReduceLROnPlateau"
    LR_PATIENCE = 5
    LR_FACTOR = 0.5
    
    # === 损失函数权重 ===
    RECON_WEIGHT = 1.0      # 重建损失权重
    KL_WEIGHT = 1e-5      # KL散度权重 (防止后验坍塌)
    EXPR_WEIGHT = 15.0      # 表达预测损失权重
    
    # KL权重退火
    KL_ANNEALING = True
    KL_ANNEAL_EPOCHS = 50   # 退火轮数
    
    # === 数据处理参数 ===
    NORMALIZE_INPUT = True   # 输入标准化
    LOG_TRANSFORM_EXPR = True  # 表达值log变换
    
    # 数据分割比例
    TRAIN_SPLIT = 0.7
    VAL_SPLIT = 0.2  
    TEST_SPLIT = 0.1
    
    # === 保存和日志 ===
    MODEL_SAVE_DIR = PROJECT_ROOT / "results" / "models"
    LOG_DIR = PROJECT_ROOT / "results" / "logs"
    PLOT_DIR = PROJECT_ROOT / "results" / "plots"
    
    # 模型保存
    SAVE_BEST_ONLY = True
    SAVE_CHECKPOINT_EVERY = 10  # 每N轮保存检查点
    
    # 日志记录
    LOG_INTERVAL = 100      # 每N个batch记录一次
    EVAL_INTERVAL = 1       # 每N轮评估一次
    
    # === 计算资源 ===
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 4         # 数据加载进程数
    PIN_MEMORY = True       # 固定内存 (GPU加速)
    
    # === 实验配置 ===
    EXPERIMENT_NAME = "roadmap_vae_baseline"
    RANDOM_SEED = 42
    
    # 数据增强
    USE_DATA_AUGMENTATION = True
    NOISE_FACTOR = 0.01     # 添加噪声的强度
    
    @classmethod
    def create_directories(cls):
        """创建必要的目录"""
        dirs_to_create = [
            cls.MODEL_SAVE_DIR,
            cls.LOG_DIR, 
            cls.PLOT_DIR
        ]
        
        for directory in dirs_to_create:
            directory.mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def get_model_save_path(cls, epoch=None, best=False):
        """获取模型保存路径"""
        if best:
            return cls.MODEL_SAVE_DIR / f"{cls.EXPERIMENT_NAME}_best.pt"
        elif epoch is not None:
            return cls.MODEL_SAVE_DIR / f"{cls.EXPERIMENT_NAME}_epoch_{epoch}.pt"
        else:
            return cls.MODEL_SAVE_DIR / f"{cls.EXPERIMENT_NAME}_latest.pt"
    
    @classmethod
    def print_config(cls):
        """打印配置信息"""
        print("=== VAE Configuration ===")
        print(f"Experiment: {cls.EXPERIMENT_NAME}")
        print(f"Device: {cls.DEVICE}")
        print(f"Input channels: {cls.INPUT_CHANNELS}")
        print(f"Sequence length: {cls.SEQUENCE_LENGTH}")
        print(f"Latent dimension: {cls.LATENT_DIM}")
        print(f"Batch size: {cls.BATCH_SIZE}")
        print(f"Learning rate: {cls.LEARNING_RATE}")
        print(f"Loss weights - Recon: {cls.RECON_WEIGHT}, KL: {cls.KL_WEIGHT}, Expr: {cls.EXPR_WEIGHT}")
        print("=" * 30)

# 创建配置实例
config = RoadmapVAEConfig()

# 使用示例
if __name__ == "__main__":
    config.print_config()
    config.create_directories()
    print("Configuration loaded and directories created!")