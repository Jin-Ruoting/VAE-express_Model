#!/usr/bin/env python3
"""
项目设置和初始化脚本
"""
import sys
from pathlib import Path
import shutil

def setup_vae_project():
    """设置VAE项目结构"""
    project_root = Path("/data/zqjinruoting/VAE-express_Model")
    
    print("=== Setting up VAE Project ===\n")
    
    # 1. 创建目录结构
    directories = [
        "data",
        "config", 
        "models",
        "train",
        "scripts",
        "notebooks",
        "results/models",
        "results/plots", 
        "results/logs",
        "tests",
        "experiments"
    ]
    
    print("1. Creating project directories:")
    for directory in directories:
        dir_path = project_root / directory
        dir_path.mkdir(parents=True, exist_ok=True)
        print(f"   [CREATED] {directory}/")
    
    # 2. 创建__init__.py文件
    init_files = [
        "data/__init__.py",
        "config/__init__.py", 
        "models/__init__.py",
        "train/__init__.py",
        "tests/__init__.py"
    ]
    
    print("\n2. Creating Python package files:")
    for init_file in init_files:
        init_path = project_root / init_file
        if not init_path.exists():
            init_path.touch()
            print(f"   [CREATED] {init_file}")
        else:
            print(f"   [EXISTS] {init_file}")
    
    # 3. 创建requirements.txt
    requirements_content = """torch>=2.0.0
torchvision>=0.15.0
numpy>=1.21.0
pandas>=1.3.0
scipy>=1.7.0
matplotlib>=3.4.0
seaborn>=0.11.0
scikit-learn>=1.0.0
tqdm>=4.62.0
tensorboard>=2.8.0
h5py>=3.6.0
jupyter>=1.0.0
plotly>=5.0.0""".strip()
    
    requirements_path = project_root / "requirements.txt"
    if not requirements_path.exists():
        requirements_path.write_text(requirements_content)
        print(f"\n3. [CREATED] requirements.txt")
    else:
        print(f"\n3. [EXISTS] requirements.txt")
    
    # 4. 创建.gitignore
    gitignore_content = """# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# PyTorch
*.pth
*.pt

# Data files
*.h5
*.hdf5
*.npz
*.pkl
*.pickle

# Jupyter Notebook
.ipynb_checkpoints

# Environment
.env
.venv
env/
venv/

# IDE
.vscode/
.idea/
*.swp
*.swo

# macOS
.DS_Store

# Results
results/models/*.pt
results/logs/*.log
experiments/outputs/

# Large data files
data/raw/
data/processed/large_files/""".strip()
    
    gitignore_path = project_root / ".gitignore"
    if not gitignore_path.exists():
        gitignore_path.write_text(gitignore_content)
        print(f"4. [CREATED] .gitignore")
    else:
        print(f"4. [EXISTS] .gitignore")
    
    # 5. 创建README.md
    readme_content = """# VAE-express_Model

An improved Variational Autoencoder (VAE) for gene expression prediction based on ChromExpress.

## Overview

This project implements a VAE architecture that:
- Encodes histone modification signals using CNN layers adapted from ChromExpress
- Learns latent representations of chromatin states
- Predicts gene expression from the learned latent space
- Reconstructs histone modification patterns

## Project Structure

```
VAE-express_Model/
├── data/           # Data loading and preprocessing
├── models/         # VAE model definitions  
├── train/          # Training related code
├── config/         # Configuration files
├── scripts/        # Utility scripts
├── notebooks/      # Jupyter notebooks for analysis
├── results/        # Model outputs and plots
├── tests/          # Unit tests
└── experiments/    # Experimental configurations
```

## Quick Start

1. **Install dependencies:**
```bash
pip install -r requirements.txt
```

2. **Check data availability:**
```bash
python scripts/check_data_availability.py
```

3. **Prepare data (if needed):**
```bash
cd /path/to/Promoter_Model/chromoformer/preprocessing
snakemake --cores 4
```

4. **Start training:**
```bash
python scripts/train_vae.py --config config/roadmap_config.py
```

## Data Requirements

- Roadmap Epigenomics histone modification data (7 marks)
- Gene expression data (RPKM values)
- Processed through ChromExpress/ChromoFormer pipeline

## Model Architecture

- **Encoder**: CNN layers → Latent space (μ, σ)
- **Decoder**: Latent space → Reconstructed histone signals
- **Regressor**: Latent space → Gene expression prediction

## License

MIT License""".strip()
    
    readme_path = project_root / "README.md"
    if not readme_path.exists():
        readme_path.write_text(readme_content)
        print(f"5. [CREATED] README.md")
    else:
        print(f"5. [EXISTS] README.md")
    
    # 6. 创建示例配置文件
    config_template = """# Example configuration template
# Copy and modify this for your experiments

class ExampleConfig:
    # Model parameters
    INPUT_CHANNELS = 7
    LATENT_DIM = 64
    SEQUENCE_LENGTH = 2000
    
    # Training parameters  
    BATCH_SIZE = 16
    LEARNING_RATE = 5e-5
    NUM_EPOCHS = 100
    
    # Loss weights
    RECON_WEIGHT = 1.0
    KL_WEIGHT = 0.0001
    EXPR_WEIGHT = 10.0""".strip()
    
    config_template_path = project_root / "config" / "example_config.py"
    if not config_template_path.exists():
        config_template_path.write_text(config_template)
        print(f"6. [CREATED] config/example_config.py")
    
    print(f"\nProject setup completed!")
    print(f"Project root: {project_root}")
    print(f"\nNext steps:")
    print("1. Run: python scripts/check_data_availability.py")
    print("2. Prepare data if needed (run Snakefile)")
    print("3. Implement data loader")
    print("4. Test VAE model")
    print("5. Start training")

def verify_setup():
    """验证设置是否成功"""
    project_root = Path("/Users/steven/Documents/workspace/VAE-express_Model")
    
    print("\n=== Verifying Setup ===")
    
    required_dirs = ["data", "config", "models", "train", "scripts", "results"]
    required_files = ["requirements.txt", "README.md", ".gitignore"]
    
    all_good = True
    
    for directory in required_dirs:
        dir_path = project_root / directory
        if dir_path.exists():
            print(f"   [OK] {directory}/")
        else:
            print(f"   [MISSING] {directory}/")
            all_good = False
    
    for file_name in required_files:
        file_path = project_root / file_name
        if file_path.exists():
            print(f"   [OK] {file_name}")
        else:
            print(f"   [MISSING] {file_name}")
            all_good = False
    
    if all_good:
        print("\nSetup verification passed!")
    else:
        print("\nSome components are missing. Please re-run setup.")
    
    return all_good

if __name__ == "__main__":
    setup_vae_project()
    verify_setup()