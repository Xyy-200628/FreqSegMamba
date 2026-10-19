#!/usr/bin/env bash
# ============================================================================
# 组件消融 —— 5 行（累计式 + 孤立行），三组件 TPAS / WAVSS / Freq-ADB
#   base         (×××)   SASS 基线
#   tpas         (✓××)   +TPAS
#   tpas_wavss   (✓✓×)   +WAVSS
#   tpas_freqadb (✓×✓)   +Freq-ADB（孤立行：验证 Freq-ADB 不依赖 WAVSS）
#   full         (✓✓✓)   全模型
#
# 统一条件: Crack500, 50 epochs, seed 42, 损失 9:1(BCE0.1/Dice0.9), edge 0.1
# 服务器: BFSU-GPU55（三卡全空）  代码: /home/gpu/XiangYiYang/SCSegamba-main/
#
# 在 tmux 里跑:
#   tmux new -s comp_abl
#   conda activate SCSegamba
#   cd /home/gpu/XiangYiYang/SCSegamba-main
#   bash run_component_ablation.sh
#   Ctrl+b 再 d 分离
# ============================================================================

export PYTHONUNBUFFERED=1

DATA_PATH="/home/gpu/XiangYiYang/data/Crack500"
EPOCHS=50
SEED=42
TPAS_MODE="adaptive4"         # ★ 真自适应（锚定弹性 SASS，逐样本沿裂缝拐弯）
BASE_DIR="./checkpoints/component_ablation"

cd /home/gpu/XiangYiYang/SCSegamba-main || exit 1

# tag  no_tpas no_wavelet no_adp  gpu
CONFIGS=(
  "base          1       1         1       1"
  "tpas          0       1         1       1"
  "tpas_wavss    0       0         1       2"
  "tpas_freqadb  0       1         0       2"
  "full          0       0         0       1"
)

for row in "${CONFIGS[@]}"; do
  read -r tag nt nw na gpu <<< "${row}"
  logdir="${BASE_DIR}/${tag}"
  mkdir -p "${logdir}"

  flags=""
  [ "$nt" = "1" ] && flags="$flags --no_tpas"
  [ "$nw" = "1" ] && flags="$flags --no_wavelet"
  [ "$na" = "1" ] && flags="$flags --no_adp"

  (
    export CUDA_VISIBLE_DEVICES="${gpu}"
    python main.py \
        --dataset_path "${DATA_PATH}" \
        --epochs "${EPOCHS}" \
        --seed "${SEED}" \
        --BCELoss_ratio 0.1 --DiceLoss_ratio 0.9 --edge_weight 0.1 \
        --tpas_mode "${TPAS_MODE}" \
        --tag "crack500_${tag}" \
        --device cuda \
        ${flags} \
        > "${logdir}/stdout.log" 2>&1
    echo "exit=$?" > "${logdir}/DONE"
  ) &
  echo "▶ 已启动 [${tag}]  GPU=${gpu}  →  ${logdir}/stdout.log"
done

echo ""
echo "5 个进程已全部启动，等待全部结束..."
wait

echo ""
echo "════════ 汇总最终测试指标 ════════"
printf "  %-14s  %-10s  %-10s  %-10s  %-10s\n" "实验" "mIoU" "F1" "ODS" "OIS"
printf "  %-14s  %-10s  %-10s  %-10s  %-10s\n" "────" "────" "──" "───" "───"
for row in "${CONFIGS[@]}"; do
  read -r tag nt nw na gpu <<< "${row}"
  log="${BASE_DIR}/${tag}/stdout.log"
  miou=$(grep -oE "\[最终测试\] mIoU = [0-9.]+" "${log}" 2>/dev/null | tail -1 | grep -oE "[0-9.]+")
  f1=$(  grep -oE "\[最终测试\] F1 = [0-9.]+"   "${log}" 2>/dev/null | tail -1 | grep -oE "[0-9.]+")
  ods=$( grep -oE "\[最终测试\] ODS = [0-9.]+"  "${log}" 2>/dev/null | tail -1 | grep -oE "[0-9.]+")
  ois=$( grep -oE "\[最终测试\] OIS = [0-9.]+"  "${log}" 2>/dev/null | tail -1 | grep -oE "[0-9.]+")
  printf "  %-14s  %-10s  %-10s  %-10s  %-10s\n" "${tag}" "${miou:--}" "${f1:--}" "${ods:--}" "${ois:--}"
done
