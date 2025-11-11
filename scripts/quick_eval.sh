#!/bin/bash
# 快速评估脚本 - 查看VAE模型训练和评估结果

echo ""
echo "======================================"
echo "  VAE 模型快速评估"
echo "======================================"
echo ""

# 检查文件
echo "检查生成的文件..."
echo ""

# 检查图表文件
if ls results/plots/*.png 1> /dev/null 2>&1; then
    ls -lh results/plots/*.png 2>/dev/null | awk '{print "  ✓", $9, "\t", $5}'
else
    echo "  未找到图表文件 (results/plots/*.png)"
fi

echo ""

# 检查数据文件
if [ -f results/test_metrics.json ]; then
    echo "  ✓ results/test_metrics.json"
fi
if [ -f results/test_predictions.csv ]; then
    echo "  ✓ results/test_predictions.csv"
fi
if [ -f results/test_latent.npz ]; then
    echo "  ✓ results/test_latent.npz"
fi

echo ""
echo "======================================"

# 显示测试指标
echo "测试集性能指标:"
echo ""
if [ -f results/test_metrics.json ]; then
    python3 -c "
import json
with open('results/test_metrics.json') as f:
    metrics = json.load(f)
    print('  Pearson R:     {:.4f}'.format(metrics.get('pearson_r', 0)))
    print('  Spearman R:    {:.4f}'.format(metrics.get('spearman_r', 0)))
    print('  R² Score:      {:.4f}'.format(metrics.get('r2_score', 0)))
    print('  MSE:           {:.4f}'.format(metrics.get('mse', 0)))
    print('  MAE:           {:.4f}'.format(metrics.get('mae', 0)))
    print('  Samples:       {}'.format(metrics.get('num_samples', 0)))
    print('')
    r = metrics.get('pearson_r', 0)
    if r > 0.70:
        print('  性能评价: 优秀 (R > 0.70)')
    elif r > 0.60:
        print('  性能评价: 良好 (0.60 < R ≤ 0.70)')
    elif r > 0.50:
        print('  性能评价: 中等 (0.50 < R ≤ 0.60)')
    elif r > 0.40:
        print('  性能评价: 一般 (0.40 < R ≤ 0.50)')
    else:
        print('  性能评价: 较差 (R ≤ 0.40)')
" 2>/dev/null || python -c "
import json
with open('results/test_metrics.json') as f:
    metrics = json.load(f)
    print('  Pearson R:     {:.4f}'.format(metrics.get('pearson_r', 0)))
    print('  Spearman R:    {:.4f}'.format(metrics.get('spearman_r', 0)))
    print('  R² Score:      {:.4f}'.format(metrics.get('r2_score', 0)))
    print('  MSE:           {:.4f}'.format(metrics.get('mse', 0)))
    print('  MAE:           {:.4f}'.format(metrics.get('mae', 0)))
    print('  Samples:       {}'.format(metrics.get('num_samples', 0)))
    print('')
    r = metrics.get('pearson_r', 0)
    if r > 0.70:
        print('  性能评价: 优秀 (R > 0.70)')
    elif r > 0.60:
        print('  性能评价: 良好 (0.60 < R ≤ 0.70)')
    elif r > 0.50:
        print('  性能评价: 中等 (0.50 < R ≤ 0.60)')
    elif r > 0.40:
        print('  性能评价: 一般 (0.40 < R ≤ 0.50)')
    else:
        print('  性能评价: 较差 (R ≤ 0.40)')
"
else
    echo "  test_metrics.json 不存在"
    echo "  请运行: python scripts/evaluate_best_model.py"
fi
echo ""

echo "======================================"

# 显示训练总结
echo "训练总结:"
echo ""
if [ -f logs/train.log ]; then
    # 统计epoch数
    total_epochs=$(grep -c 'epoch' logs/train.log)
    echo "  总训练Epoch数: $total_epochs"
    echo ""
    
    # 找最佳验证R
    echo "  最佳验证性能:"
    grep "Val.*R:" logs/train.log | sort -t',' -k2 -gr | head -1 | sed 's/^/    /'
    echo ""
    
    # 显示最后几个epoch
    echo "  最后3个epoch:"
    tail -9 logs/train.log | grep -E "epoch|Train|Val" | tail -7 | sed 's/^/    /'
    
else
    echo "  train.log 不存在"
fi

echo ""
echo "======================================"

# 过拟合检查
echo "过拟合检查:"
echo ""
if [ -f logs/train.log ]; then
    # 提取最后一个epoch的训练和验证R
    last_train_r=$(grep "Train.*R:" logs/train.log | tail -1 | sed 's/.*R: \([0-9.]*\).*/\1/')
    last_val_r=$(grep "Val.*R:" logs/train.log | tail -1 | sed 's/.*R: \([0-9.]*\).*/\1/')
    
    if [ ! -z "$last_train_r" ] && [ ! -z "$last_val_r" ]; then
        echo "  最终训练R:   $last_train_r"
        echo "  最终验证R:   $last_val_r"
        
        # 简单比较
        python3 -c "
train_r = float('$last_train_r')
val_r = float('$last_val_r')
diff = train_r - val_r
print(f'  差值:       {diff:.3f}')
print('')
if diff < 0.05:
    print('  无过拟合 (差值 < 0.05)')
elif diff < 0.10:
    print('  轻微过拟合 (0.05 ≤ 差值 < 0.10)')
else:
    print('  明显过拟合 (差值 ≥ 0.10)')
" 2>/dev/null || python -c "
train_r = float('$last_train_r')
val_r = float('$last_val_r')
diff = train_r - val_r
print(f'  差值:       {diff:.3f}')
print('')
if diff < 0.05:
    print('  无过拟合 (差值 < 0.05)')
elif diff < 0.10:
    print('  轻微过拟合 (0.05 ≤ 差值 < 0.10)')
else:
    print('  明显过拟合 (差值 ≥ 0.10)')
"
    fi
else
    echo "  无法检查（缺少训练日志）"
fi

echo ""
echo "======================================"
echo ""
echo "查看图表:"
echo "  - results/plots/training_curves.png"
echo "  - results/plots/evaluation_report.png"
echo ""
echo "详细结果文件:"
echo "  - results/test_metrics.json (性能指标)"
echo "  - results/test_predictions.csv (预测结果)"
echo "  - results/test_latent.npz (潜空间表示)"
echo ""
echo "======================================"
echo ""

