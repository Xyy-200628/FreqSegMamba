#!/usr/bin/env bash
# ============================================================================
# 损失消融 —— 4 个 Dice-heavy 比例（7:3, 8:2, 9:1, 1:1），edge_weight=0.1，跑满 35 轮
# （修正：之前误设 edge_weight=1.0，现改为 0.1）
#
# 服务器: BFSU58   环境: FreqSegamba   代码: /app/XiangYiYang/SCSegamba-main/
# 在 tmux 里跑（关电脑不中断）:
#   tmux new -s loss_ablation2
#   conda activate FreqSegamba
#   cd /app/XiangYiYang/SCSegamba-main
#   bash run_loss_ablation.sh
#   Ctrl+b 再 d 分离
# ============================================================================

DATA_PATH="/app/XiangYiYang/data/Crack500"   # ✅ 已确认
NO_ADP_FLAG=""                                # ✅ no_adp=False，边缘分支开
EPOCHS=35
SEED=42
BASE_DIR="./checkpoints/loss_ablation"        # 每个实验一个子目录

cd /app/XiangYiYang/SCSegamba-main || exit 1
export PYTHONUNBUFFERED=1

# ── 4 个配置：tag  BCE  Dice  edge  GPU（edge_weight=0.1）─────────────────
# 同 tag 会覆盖之前 edge=1.0 的错误结果
CONFIGS=(
  "d7_b3  0.3  0.7  0.1  7"
  "d8_b2  0.2  0.8  0.1  7"
  "d9_b1  0.1  0.9  0.1  7"
  "d1_b1  0.5  0.5  0.1  7"
)

# ── 并行启动，每个实验写到自己独立目录 ──────────────────────────────────
for row in "${CONFIGS[@]}"; do
  read -r tag bce dice edge gpu <<< "${row}"
  logdir="${BASE_DIR}/${tag}"
  mkdir -p "${logdir}"
  (
    export CUDA_VISIBLE_DEVICES="${gpu}"
    python main.py \
        --dataset_path "${DATA_PATH}" \
        --epochs "${EPOCHS}" \
        --seed "${SEED}" \
        --BCELoss_ratio "${bce}" \
        --DiceLoss_ratio "${dice}" \
        --edge_weight "${edge}" \
        --tag "${tag}" \
        --device cuda \
        ${NO_ADP_FLAG} \
        > "${logdir}/stdout.log" 2>&1
    echo "exit=$?" > "${logdir}/DONE"
  ) &
  echo "▶ 已启动 [${tag}]  GPU=${gpu}  →  ${logdir}/stdout.log"
done

echo ""
echo "4 个进程已全部启动，等待全部结束（不限时）..."
wait

echo ""
echo "════════ 全部结束，汇总最终测试 mIoU ════════"
for row in "${CONFIGS[@]}"; do
  read -r tag bce dice edge gpu <<< "${row}"
  log="${BASE_DIR}/${tag}/stdout.log"
  m=$(grep -oE "\[最终测试\] mIoU = [0-9.]+" "${log}" 2>/dev/null | tail -n 1)
  echo "  ${tag}: ${m:-（无结果，见 ${log}）}"
done
