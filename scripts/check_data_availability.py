#!/usr/bin/env python3
"""
检查Promoter_Model项目中的数据是否可用
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np

def check_promoter_model_data():
    """检查参考项目的数据状态"""
    base_path = Path("/data/zqjinruoting/Promoter_Model/chromoformer/preprocessing")
    
    print("=== Checking Promoter_Model Data Status ===\n")
    
    # 1. 检查基本文件
    files_to_check = [
        "train.csv",
        "exp/raw_exp.tsv", 
        "annotations/hg19.fa.sizes",
        "Snakefile"
    ]
    
    print("1. Basic Files Check:")
    missing_files = []
    for file_path in files_to_check:
        full_path = base_path / file_path
        if full_path.exists():
            print(f"   [OK] {file_path}: Found")
            
            # 检查CSV文件内容
            if file_path.endswith('.csv'):
                try:
                    df = pd.read_csv(full_path)
                    print(f"       -> Contains {len(df):,} rows")
                except Exception as e:
                    print(f"       -> Error reading: {e}")
            
            # 检查TSV文件内容
            elif file_path.endswith('.tsv'):
                try:
                    df = pd.read_csv(full_path, sep='\t')
                    print(f"       -> Contains {len(df):,} rows, {len(df.columns)} columns")
                except Exception as e:
                    print(f"       -> Error reading: {e}")
        else:
            print(f"   [MISSING] {file_path}")
            missing_files.append(file_path)
    
    # 2. 检查组蛋白数据
    print("\n2. Histone Modification Data Check:")
    hist_dir = base_path / "hist"
    if hist_dir.exists():
        npz_files = list(hist_dir.glob("*.npz"))
        bam_files = list(hist_dir.glob("*.bam"))
        bedgraph_files = list(hist_dir.glob("*.bedGraph"))
        
        print(f"   NPZ files: {len(npz_files)}")
        print(f"   BAM files: {len(bam_files)}")
        print(f"   BedGraph files: {len(bedgraph_files)}")
        
        if npz_files:
            print("   Sample NPZ files:")
            for f in npz_files[:3]:
                print(f"      - {f.name}")
                
        # 检查组蛋白标记覆盖情况
        expected_marks = ['H3K4me1', 'H3K4me3', 'H3K9me3', 'H3K27me3', 'H3K36me3', 'H3K27ac', 'H3K9ac']
        expected_eids = ['E003', 'E004', 'E005', 'E006', 'E007', 'E016', 'E066', 'E087', 'E114', 'E116', 'E118']
        
        mark_coverage = {}
        for mark in expected_marks:
            mark_files = list(hist_dir.glob(f"*-{mark}.npz"))
            mark_coverage[mark] = len(mark_files)
        
        print("\n   Histone Mark Coverage:")
        for mark, count in mark_coverage.items():
            status = "[OK]" if count > 0 else "[MISSING]"
            print(f"      {status} {mark}: {count}/{len(expected_eids)} cell types")
            
    else:
        print("   [ERROR] hist/ directory not found")
    
    # 3. 检查处理后的数据
    print("\n3. Processed Data Check:")
    data_dir = base_path / "data"
    if data_dir.exists():
        eid_dirs = [d for d in data_dir.iterdir() if d.is_dir()]
        print(f"   Found {len(eid_dirs)} cell type directories")
        
        total_gene_files = 0
        for eid_dir in eid_dirs:
            npz_files = list(eid_dir.glob("*.npz"))
            total_gene_files += len(npz_files)
            if len(eid_dirs) <= 5:  # 只显示前5个的详细信息
                print(f"      - {eid_dir.name}: {len(npz_files):,} gene files")
        
        if len(eid_dirs) > 5:
            print(f"      ... and {len(eid_dirs) - 5} more directories")
        
        print(f"   Total gene data files: {total_gene_files:,}")
    else:
        print("   [ERROR] data/ directory not found")
    
    # 4. 生成诊断和建议
    print("\n=== Diagnosis and Recommendations ===")
    
    if missing_files:
        print("[FIX NEEDED] Missing critical files. Need to run Snakefile pipeline:")
        print("Commands:")
        print("cd /Users/steven/Documents/workspace/Promoter_Model/chromoformer/preprocessing")
        print("conda activate chromoformer  # or conda env create -f environment.yaml")
        print("snakemake --cores 4 --keep-going")
        return False
    
    elif not data_dir.exists() or len(list(data_dir.glob("*/"))) == 0:
        print("[FIX NEEDED] Basic files exist but processed data is missing.")
        print("Continue running Snakefile to extract signals:")
        print("cd /Users/steven/Documents/workspace/Promoter_Model/chromoformer/preprocessing")
        print("snakemake --cores 4 --keep-going")
        return False
    
    else:
        print("[SUCCESS] Data appears ready for VAE model development!")
        print("Recommended next steps:")
        print("1. Implement data loader for Roadmap format")
        print("2. Test VAE model with sample data")
        print("3. Start training pipeline")
        print("4. Evaluate model performance")
        return True

def check_vae_project_structure():
    """检查VAE项目结构是否完整"""
    project_root = Path("/Users/steven/Documents/workspace/VAE-express_Model")
    
    print("\n=== VAE Project Structure Check ===")
    
    required_dirs = ["data", "config", "models", "train", "scripts", "results"]
    required_files = [
        "models/vae.py",
        "train/trainer.py", 
        "train/losses.py"
    ]
    
    print("Required directories:")
    for directory in required_dirs:
        dir_path = project_root / directory
        status = "[OK]" if dir_path.exists() else "[MISSING]"
        print(f"   {status} {directory}/")
    
    print("\nRequired files:")
    for file_path in required_files:
        full_path = project_root / file_path
        status = "[OK]" if full_path.exists() else "[MISSING]"
        print(f"   {status} {file_path}")

if __name__ == "__main__":
    # 检查数据状态
    data_ready = check_promoter_model_data()
    
    # 检查项目结构
    check_vae_project_structure()
    
    # 最终状态报告
    print(f"\n{'='*50}")
    if data_ready:
        print("STATUS: Ready to proceed with VAE development!")
    else:
        print("STATUS: Data preparation needed before VAE development")
    print("="*50)