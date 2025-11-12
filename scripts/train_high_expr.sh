#!/bin/bash
# 优化高表达基因预测的训练脚本

# 设置运行名称（可通过环境变量覆盖）
RUN_NAME=${RUN_NAME:-"raw_mode"}

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR=logs
mkdir -p ${LOG_DIR}

echo "========================================================================"
echo "  VAE 模型训练 - 优化高表达基因预测"
echo "========================================================================"
echo "运行名称: ${RUN_NAME}"
echo "时间戳: ${TIMESTAMP}"
echo "日志目录: ${LOG_DIR}"
echo ""
echo "关键配置:"
echo "  - EXPR_LOSS_MODE=raw  (在原始值上计算 MSE)"
echo "  - EXPR_ALPHA=0.8      (80% MSE + 20% 相关损失)"
echo "  - FREEZE_EPOCHS=10    (前 10 轮仅训练回归器)"
echo ""
echo "结果将保存到: results/YYYYMMDD-HHMM_${RUN_NAME}/"
echo "========================================================================"
echo ""

# Phase 1: 训练表达回归器（冻结编码器/解码器）
echo "[Phase 1] 训练表达回归器"
echo "------------------------------------------------------------------------"
echo "开始时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

CUDA_VISIBLE_DEVICES=0,1 \
NUM_WORKERS=32 OMP_NUM_THREADS=32 MKL_NUM_THREADS=32 \
RUN_NAME="${RUN_NAME}" \
EXPR_ALPHA=0.8 EXPR_W=6.0 RECON_W=0.0 KL_W=0.0 \
EXPR_LOSS_MODE=raw \
FREEZE_EPOCHS=10 KL_SCHEDULE=linear KL_BETA_MAX=1e-5 \
python scripts/train_vae.py 2>&1 | tee ${LOG_DIR}/train_phase1_${TIMESTAMP}.log

PHASE1_EXIT=$?
echo ""
echo "Phase 1 完成 (退出码: ${PHASE1_EXIT})"
echo "结束时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

if [ ${PHASE1_EXIT} -ne 0 ]; then
    echo "错误: Phase 1 训练失败"
    exit 1
fi

# Phase 2: 联合训练（解冻所有层）
echo "[Phase 2] 联合训练"
echo "------------------------------------------------------------------------"
echo "开始时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

CUDA_VISIBLE_DEVICES=0,1 \
NUM_WORKERS=32 OMP_NUM_THREADS=32 MKL_NUM_THREADS=32 \
RUN_NAME="${RUN_NAME}" \
EXPR_ALPHA=0.8 EXPR_W=4.0 RECON_W=0.01 KL_W=0.01 \
EXPR_LOSS_MODE=raw \
KL_SCHEDULE=linear KL_BETA_MAX=1e-5 \
python scripts/train_vae.py 2>&1 | tee ${LOG_DIR}/train_phase2_${TIMESTAMP}.log

PHASE2_EXIT=$?
echo ""
echo "Phase 2 完成 (退出码: ${PHASE2_EXIT})"
echo "结束时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

if [ ${PHASE2_EXIT} -ne 0 ]; then
    echo "错误: Phase 2 训练失败"
    exit 1
fi

# 获取实际创建的运行目录
RUN_DIR=$(ls -td results/2*_${RUN_NAME} 2>/dev/null | head -1)

# 训练完成
echo "========================================================================"
echo "  训练完成！"
echo "========================================================================"
echo ""
echo "运行目录: ${RUN_DIR}"
echo ""
echo "生成的文件:"
echo "  - 最佳模型: ${RUN_DIR}/models/vae_best.pt"
echo "  - 最终模型: ${RUN_DIR}/models/vae_last.pt"
echo "  - 兼容路径: results/models/vae_promoter_only_best.pt"
echo "  - 日志: ${LOG_DIR}/train_phase1_${TIMESTAMP}.log"
echo "  - 日志: ${LOG_DIR}/train_phase2_${TIMESTAMP}.log"
echo ""
echo "下一步:"
echo "  1. 评估模型（使用本次运行的结果）:"
echo "     python scripts/evaluate_best_model.py --run_dir ${RUN_DIR}"
echo ""
echo "  2. 或使用兼容路径评估（默认）:"
echo "     python scripts/evaluate_best_model.py"
echo ""
echo "  3. 检查预测值范围:"
echo "     python -c \"import pandas as pd; df = pd.read_csv('${RUN_DIR}/test_predictions.csv'); print('预测值范围:', df['pred_expr'].min(), '-', df['pred_expr'].max())\""
echo ""
echo "========================================================================"

