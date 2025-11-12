#!/usr/bin/env python3
"""
列出和比较所有训练运行的结果
"""
import json
from pathlib import Path
import pandas as pd
from datetime import datetime

def list_runs():
    """列出所有运行及其关键指标"""
    
    results_dir = Path('results')
    
    # 查找所有运行目录（格式：YYYYMMDD-HHMM_name）
    run_dirs = sorted([d for d in results_dir.iterdir() 
                      if d.is_dir() and len(d.name) >= 13 and d.name[8] == '-'],
                      reverse=True)
    
    if not run_dirs:
        print("未找到任何运行目录")
        return
    
    print("="*100)
    print("  训练运行列表")
    print("="*100)
    print()
    
    # 收集所有运行的信息
    runs_info = []
    
    for run_dir in run_dirs:
        info = {
            'dir': run_dir.name,
            'timestamp': None,
            'name': None,
            'r2': None,
            'pearson_r': None,
            'pred_max': None,
            'pred_gt7': None,
            'has_model': False,
            'has_metrics': False
        }
        
        # 解析目录名
        parts = run_dir.name.split('_', 1)
        if len(parts) == 2:
            info['timestamp'] = parts[0]
            info['name'] = parts[1]
        else:
            info['timestamp'] = run_dir.name[:13]
            info['name'] = 'unknown'
        
        # 检查模型文件
        model_path = run_dir / 'models' / 'vae_best.pt'
        info['has_model'] = model_path.exists()
        
        # 读取指标
        metrics_path = run_dir / 'test_metrics.json'
        if metrics_path.exists():
            info['has_metrics'] = True
            try:
                with open(metrics_path) as f:
                    metrics = json.load(f)
                info['r2'] = metrics.get('r2_score')
                info['pearson_r'] = metrics.get('pearson_r')
            except:
                pass
        
        # 读取预测结果
        pred_path = run_dir / 'test_predictions.csv'
        if pred_path.exists():
            try:
                df = pd.read_csv(pred_path)
                info['pred_max'] = df['pred_expr'].max()
                info['pred_gt7'] = (df['pred_expr'] > 7).sum()
            except:
                pass
        
        runs_info.append(info)
    
    # 打印表格
    print(f"{'时间戳':<15} {'运行名称':<20} {'R²':<8} {'Pearson R':<10} {'预测最大值':<10} {'>7样本数':<10} {'状态'}")
    print("-"*100)
    
    for info in runs_info:
        timestamp = info['timestamp']
        name = info['name'][:18] + '..' if len(info['name']) > 20 else info['name']
        
        r2_str = f"{info['r2']:.4f}" if info['r2'] is not None else '-'
        r_str = f"{info['pearson_r']:.4f}" if info['pearson_r'] is not None else '-'
        max_str = f"{info['pred_max']:.2f}" if info['pred_max'] is not None else '-'
        gt7_str = f"{info['pred_gt7']}" if info['pred_gt7'] is not None else '-'
        
        status = []
        if info['has_model']:
            status.append('✓模型')
        if info['has_metrics']:
            status.append('✓指标')
        status_str = ' '.join(status) if status else '未完成'
        
        print(f"{timestamp:<15} {name:<20} {r2_str:<8} {r_str:<10} {max_str:<10} {gt7_str:<10} {status_str}")
    
    print()
    print("="*100)
    print(f"共找到 {len(runs_info)} 次运行")
    print()
    
    # 找出最佳运行
    valid_runs = [r for r in runs_info if r['r2'] is not None]
    if valid_runs:
        best_r2 = max(valid_runs, key=lambda x: x['r2'])
        best_r = max(valid_runs, key=lambda x: x['pearson_r'])
        
        print("最佳运行:")
        print(f"  R² 最高: {best_r2['dir']} (R²={best_r2['r2']:.4f})")
        print(f"  Pearson R 最高: {best_r['dir']} (R={best_r['pearson_r']:.4f})")
        print()
    
    # 使用说明
    print("评估特定运行:")
    if runs_info:
        latest = runs_info[0]
        print(f"  python scripts/evaluate_best_model.py --run_dir results/{latest['dir']}")
    print()
    print("查看详细指标:")
    if runs_info:
        latest = runs_info[0]
        print(f"  cat results/{latest['dir']}/test_metrics.json")
    print()

def compare_runs(run_dirs):
    """详细比较多个运行"""
    print("="*100)
    print("  运行对比")
    print("="*100)
    print()
    
    for run_dir_str in run_dirs:
        run_dir = Path(run_dir_str)
        if not run_dir.exists():
            print(f"警告: {run_dir} 不存在")
            continue
        
        print(f"运行: {run_dir.name}")
        print("-"*80)
        
        # 读取指标
        metrics_path = run_dir / 'test_metrics.json'
        if metrics_path.exists():
            with open(metrics_path) as f:
                metrics = json.load(f)
            
            print(f"  R²:           {metrics.get('r2_score', 'N/A'):.4f}")
            print(f"  Pearson R:    {metrics.get('pearson_r', 'N/A'):.4f}")
            print(f"  Spearman R:   {metrics.get('spearman_r', 'N/A'):.4f}")
            print(f"  MSE:          {metrics.get('mse', 'N/A'):.4f}")
            print(f"  RMSE:         {metrics.get('rmse', 'N/A'):.4f}")
            print(f"  MAE:          {metrics.get('mae', 'N/A'):.4f}")
            print(f"  样本数:       {metrics.get('n_samples', 'N/A')}")
        
        # 读取预测结果
        pred_path = run_dir / 'test_predictions.csv'
        if pred_path.exists():
            df = pd.read_csv(pred_path)
            print(f"  预测值范围:   [{df['pred_expr'].min():.2f}, {df['pred_expr'].max():.2f}]")
            print(f"  预测值 > 7:   {(df['pred_expr'] > 7).sum()} 个样本")
            print(f"  预测值 > 10:  {(df['pred_expr'] > 10).sum()} 个样本")
        
        print()

if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1:
        # 比较指定的运行
        compare_runs(sys.argv[1:])
    else:
        # 列出所有运行
        list_runs()
