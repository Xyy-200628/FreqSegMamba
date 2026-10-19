"""
engine.py -- train one epoch
FP-ADB version: supports boundary output
"""
# engine.py 第19行: for i, data in enumerate(data_loader):
#   │
#   │  ★ 从DataLoader取一个batch（1张图 + 1张标签）
#   │  samples = (1, 3, 512, 512)    ← 裂缝照片
#   │  targets = (1, 1, 512, 512)    ← 人工标注（0=背景, 1=裂缝）
#   │
#   ▼
# engine.py 第22行: output = model(samples, return_boundary=True)
#   │
#   │  ★ 前向传播：图片穿过整个网络
#   │  samples → decoder.py:forward()
#   │    ├─ backbone → [F4,F3,F2,F1]    ← WAVSS.py 编码器
#   │    ├─ adp_stream → [D4,D3,D2,D1]  ← edge_utils.py ADP-Stream
#   │    ├─ CFAF×4 → 融合               ← edge_utils.py CFAF
#   │    └─ MFS → mask + feat           ← MFS.py
#   │        mask:  (1, 1, 512, 512)    ← 裂缝预测(logits)
#   │        boundary: (1, 1, 512, 512) ← 边缘预测(logits, 仅训练)
#   │
#   ▼
# engine.py 第24行: loss = criterion(mask, targets, boundary)
#   │
#   │  ★ 计算总损失 = L_main + L_boundary
#   │  decoder.py bce_dice_edge.forward():
#   │
#   │  ┌─ L_main（第138-141行）─────────────────────────┐
#   │  │ BCE(mask, targets)       ← 逐像素: 每个像素对不对？   │
#   │  │   + Dice(mask, targets)  ← 区域级: 裂缝区域重叠够吗？ │
#   │  │ L_main = 0.83×BCE + 0.17×Dice                      │
#   │  │ 例: L_main = 0.12                                  │
#   │  └──────────────────────────────────────────────────┘
#   │
#   │  ┌─ L_boundary（第143-145行）─────────────────────┐
#   │  │ Sobel(boundary) vs Sobel(targets) ← 边缘对不对？  │
#   │  │ L_boundary = BCE(边缘预测, 边缘标签) × 0.05       │
#   │  │ 例: L_boundary = 0.04                             │
#   │  └──────────────────────────────────────────────────┘
#   │
#   │  总损失: loss = 0.12 + 0.04 = 0.16
#   │
#   ▼
# engine.py 第34行: loss.backward()
#   │
#   │  ★ 反向传播：计算每个参数的梯度
#   │  梯度从 loss 往回传 → BoundaryHead → MFS → CFAF → Encoder
#   │                     → MainHead → MFS → CFAF → Encoder
#   │                     → ADP-Stream
#   │
#   ▼
# engine.py 第35行: optimizer.step()
#   │
#   │  ★ 更新参数：根据梯度微调所有 2.8M 个参数
#   │  参数 = 参数 - lr × 梯度
#   │
#   ▼
# engine.py 第33行: optimizer.zero_grad()
#   │
#   │  ★ 清零梯度，准备下一个batch
#   │
#   ▼
# 下一个 batch（重复 300 次 = 1 个 epoch）

from typing import Iterable
import torch
import time
from tqdm import tqdm

def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                     epoch: int, args = None, logger = None):
    model.train()
    criterion.train()

    pbar = tqdm(total=len(data_loader.dataloader), desc=f"Initial Loss Fused: Pending")
    for i, data in enumerate(data_loader):
        samples = data['image'].to(torch.device(args.device))
        targets = data['label'].to(torch.device(args.device))

        # FP-ADB: request boundary output during training
        output = model(samples, return_boundary=True)

        # Handle tuple return (mask, boundary) or single return (mask only)
        if isinstance(output, tuple):
            mask, boundary = output
        else:
            mask = output
            boundary = None

        loss_final = criterion(mask, targets.float(), boundary)
        cur_time = time.strftime('%Y_%m_%d_%H:%M:%S', time.localtime(time.time()))

        loss_final_str = '{:.4f}'.format(loss_final.item())
        lr = optimizer.param_groups[0]['lr']
        logger.info(f"time -> {cur_time} | Epoch -> {epoch} | "
                    f"image_num -> {data['A_paths']} | loss final -> {loss_final_str} | lr -> {lr}")

        pbar.set_description(f"Loss: {loss_final.item():.4f}")
        pbar.update(1)
        optimizer.zero_grad()
        loss_final.backward()
        optimizer.step()

    pbar.close()
