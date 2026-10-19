#!/usr/bin/env bash
# ============================================================================
# 实时显示 4 个 Dice-heavy 消融实验的训练进度（当前 epoch + 最新 mIoU）
# 用法：bash show_progress.sh   （Ctrl+C 退出，不影响训练本身）
# ============================================================================
cd /app/XiangYiYang/SCSegamba-main || exit 1

TAGS="d7_b3 d8_b2 d9_b1 d1_b1"
BASE="./checkpoints/loss_ablation"

while true; do
  clear
  echo "════════ Dice:BCE 消融实时进度  $(date '+%m-%d %H:%M:%S') ════════"
  echo ""
  printf "  %-12s  %-22s  %s\n" "实验" "当前 epoch" "最新 mIoU"
  printf "  %-12s  %-22s  %s\n" "────" "──────────" "────────"
  for tag in ${TAGS}; do
    log="${BASE}/${tag}/stdout.log"
    if [ -f "${log}" ]; then
      ep=$(grep -oE '训练轮次 [0-9]+ 开始' "${log}" 2>/dev/null | tail -1 | grep -oE '[0-9]+')
      miou=$(grep -oE 'mIoU = [0-9.]+' "${log}" 2>/dev/null | tail -1 | grep -oE '[0-9.]+')
      done_flag=""
      [ -f "${BASE}/${tag}/DONE" ] && done_flag=" ✅$(cat "${BASE}/${tag}/DONE")"
      printf "  %-12s  %-22s  %s%s\n" "${tag}" "epoch ${ep:-0}" "${miou:--}" "${done_flag}"
    else
      printf "  %-12s  %-22s  %s\n" "${tag}" "（未启动）" "-"
    fi
  done
  echo ""
  echo "  （每 10 秒刷新，Ctrl+C 退出）"
  sleep 10
done
