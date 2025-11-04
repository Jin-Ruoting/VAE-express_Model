#!/bin/bash
# 两阶段训练脚本 - 自动保存日志
# 使用方法: ./scripts/train_with_log.sh

# 设置日期时间戳
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="logs"
mkdir -p $LOG_DIR

echo "========================================"
echo "  VAE 模型两阶段训练"
echo "  开始时间: $(date)"
echo "========================================"
echo ""

# ============================================
# 阶段 1: 仅训练表达回归器 (前8个epoch)
# ============================================
echo "========================================"
echo "  阶段 1: 训练表达回归器"
echo "  - 冻结编码器和解码器"
echo "  - 仅训练表达预测头"
echo "  - Epoch: 8"
echo "========================================"
echo ""

CUDA_VISIBLE_DEVICES=0,1 \
NUM_WORKERS=8 \
OMP_NUM_THREADS=8 \
MKL_NUM_THREADS=8 \
EXPR_ALPHA=0.0 \
EXPR_W=6.0 \
RECON_W=0.0 \
KL_W=0.0 \
FREEZE_EPOCHS=8 \
KL_SCHEDULE=linear \
KL_BETA_MAX=1e-5 \
python scripts/train_vae.py 2>&1 | tee ${LOG_DIR}/train_phase1_${TIMESTAMP}.log

echo ""
echo "阶段 1 完成！"
echo ""
echo "按 Enter 继续阶段 2，或 Ctrl+C 退出..."
read

# ============================================
# 阶段 2: 联合训练所有模块
# ============================================
echo ""
echo "========================================"
echo "  阶段 2: 联合训练"
echo "  - 解冻所有模块"
echo "  - 联合优化表达、重构和KL"
echo "  - Epoch: 继续训练"
echo "========================================"
echo ""

CUDA_VISIBLE_DEVICES=0,1 \
NUM_WORKERS=8 \
OMP_NUM_THREADS=8 \
MKL_NUM_THREADS=8 \
EXPR_ALPHA=0.2 \
EXPR_W=4.0 \
RECON_W=0.01 \
KL_W=0.01 \
KL_SCHEDULE=linear \
KL_BETA_MAX=1e-5 \
python scripts/train_vae.py 2>&1 | tee ${LOG_DIR}/train_phase2_${TIMESTAMP}.log

# ============================================
# 训练完成，处理日志
# ============================================
echo ""
echo "========================================"
echo "  训练完成！"
echo "  结束时间: $(date)"
echo "========================================"
echo ""

# 合并两个阶段的日志
echo "合并日志文件..."
cat ${LOG_DIR}/train_phase1_${TIMESTAMP}.log ${LOG_DIR}/train_phase2_${TIMESTAMP}.log > ${LOG_DIR}/train_complete_${TIMESTAMP}.log

# 创建软链接到最新日志
ln -sf train_complete_${TIMESTAMP}.log ${LOG_DIR}/train.log

echo ""
echo "日志文件已保存:"
echo "  - ${LOG_DIR}/train_phase1_${TIMESTAMP}.log (阶段1)"
echo "  - ${LOG_DIR}/train_phase2_${TIMESTAMP}.log (阶段2)"
echo "  - ${LOG_DIR}/train_complete_${TIMESTAMP}.log (完整)"
echo "  - ${LOG_DIR}/train.log (软链接到完整日志)"
echo ""
echo "现在可以运行:"
echo "  python scripts/visualize_training_results.py"
echo "  python scripts/evaluate_best_model.py"
echo ""

