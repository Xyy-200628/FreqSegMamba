#!/usr/bin/env bash
# ============================================================================
# TPAS 模式消融 —— Crack500，3 个 mode 各占一张卡并行
#   adaptive4 = 真自适应（锚定弹性 SASS，逐样本沿裂缝拐弯）
#   theta_dir = SASS 固定路径 + 逐样本无向朝向码（★推荐）
#   sass_dir  = SASS 固定路径 + 逐像素移动方向码（基线）
#
# 统一条件: 50 epochs, seed 42, 损失 9:1(BCE0.1/Dice0.9), --no_adp(隔离边缘分支)
# 目的: 只看 --tpas_mode 这一个变量，判断"自适应路径"到底换来多少 mIoU
#
# 服务器: BFSU-GPU55   代码: /home/gpu/XiangYiYang/SCSegamba-main/
# 在 tmux 里跑:
#   tmux new -s tpas_mode
#   conda activate SCSegamba
#   cd /home/gpu/XiangYiYang/SCSegamba-main
#   bash run_tpas_mode_ablation.sh
#   Ctrl+b 再 d 分离
# ============================================================================

export PYTHONUNBUFFERED=1

DATA_PATH="/home/gpu/XiangYiYang/data/Crack500"
EPOCHS=50
SEED=42
BASE_DIR="./checkpoints/tpas_mode"

cd /home/gpu/XiangYiYang/SCSegamba-main || exit 1

# mode  gpu
CONFIGS=(
  "adaptive4   1"
  "theta_dir   2"
  "sass_dir    0"
)

for row in "${CONFIGS[@]}"; do
  read -r mode gpu <<< "${row}"
  logdir="${BASE_DIR}/${mode}"
  mkdir -p "${logdir}"
  (
    export CUDA_VISIBLE_DEVICES="${gpu}"
    python main.py \
        --dataset_path "${DATA_PATH}" \
        --tpas_mode "${mode}" \
        --epochs "${EPOCHS}" \
        --seed "${SEED}" \
        --BCELoss_ratio 0.1 --DiceLoss_ratio 0.9 --edge_weight 0.1 \
        --no_adp \
        --tag "crack500_${mode}" \
        --device cuda \
        > "${logdir}/stdout.log" 2>&1
    echo "exit=$?" > "${logdir}/DONE"
  ) &
  echo "▶ 已启动 [${mode}]  GPU=${gpu}  →  ${logdir}/stdout.log"
done

echo ""
echo "3 个进程已全部启动，等待全部结束..."
wait

echo ""
echo "════════ 汇总最终测试指标 ════════"
printf "  %-12s  %-10s  %-10s  %-10s  %-10s\n" "mode" "mIoU" "F1" "ODS" "OIS"
printf "  %-12s  %-10s  %-10s  %-10s  %-10s\n" "────" "────" "──" "───" "───"
for row in "${CONFIGS[@]}"; do
  read -r mode gpu <<< "${row}"
  log="${BASE_DIR}/${mode}/stdout.log"
  miou=$(grep -oE "\[最终测试\] mIoU = [0-9.]+" "${log}" 2>/dev/null | tail -1 | grep -oE "[0-9.]+")
  f1=$(  grep -oE "\[最终测试\] F1 = [0-9.]+"   "${log}" 2>/dev/null | tail -1 | grep -oE "[0-9.]+")
  ods=$( grep -oE "\[最终测试\] ODS = [0-9.]+"  "${log}" 2>/dev/null | tail -1 | grep -oE "[0-9.]+")
  ois=$( grep -oE "\[最终测试\] OIS = [0-9.]+"  "${log}" 2>/dev/null | tail -1 | grep -oE "[0-9.]+")
  printf "  %-12s  %-10s  %-10s  %-10s  %-10s\n" "${mode}" "${miou:--}" "${f1:--}" "${ods:--}" "${ois:--}"
done
