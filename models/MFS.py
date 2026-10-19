'''
Author: Hui Liu
Github: https://github.com/Karl1109
Email: liuhui@ieee.org

================================================================================
models/MFS.py —— ③ 多尺度特征融合（Multi-scale Feature Fusion）
================================================================================

┌─────────────────────────────────────────────────────────────────────────┐
│ 本文件在网络中的位置                                                       │
│                                                                          │
│   编码器输出 [F4, F3, F2, F1]                                             │
│     │ F4:(B,128,64,64)  F3:(B,64,128,128)                                │
│     │ F2:(B,32,256,256) F1:(B,16,512,512)                                │
│     ▼                                                                     │
│   ┌─────────────────────────────────────────────────┐                    │
│   │ ③ MFS 多尺度融合 ★ 本文件                          │                    │
│   │                                                   │                   │
│   │  F4 → MLP(128→8) → DySample(×8) → 512×512        │                   │
│   │  F3 → MLP(64→8)  → DySample(×4) → 512×512        │                   │
│   │  F2 → MLP(32→8)  → DySample(×2) → 512×512        │                   │
│   │  F1 → MLP(16→8)  → (保持512×512)                   │                   │
│   │                    ↓                               │                   │
│   │  Concat(8,8,8,8)→32维 → GBC全局上下文→BottConv→8维  │                   │
│   │                    ↓                               │                   │
│   │  Dropout → BottConv(8→1) → Conv(1→1)              │                   │
│   │                    ↓                               │                   │
│   │  输出: (B, 1, 512, 512) 单通道裂缝预测图            │                   │
│   └─────────────────────────────────────────────────┘                    │
└─────────────────────────────────────────────────────────────────────────┘

关键设计：DySample 动态上采样（轻量高效，比传统插值更好）
'''

import torch.nn as nn
import torch
from models.GBC import GBC, BottConv          # → GBC.py
from models.DySample import DySample          # → DySample.py：动态上采样


# ============================================================================
# MLP —— 单层线性投影
# ============================================================================
class MLP(nn.Module):
    '''简单的2层感知机：input_dim → embed_dim'''
    def __init__(self, input_dim=2048, embed_dim=768):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        x = self.proj(x)
        return x


# ============================================================================
# ③ MFS —— 多尺度特征融合模块
# ============================================================================
class MFS(nn.Module):
    '''
    MFS = Multi-Scale Feature Fusion

    将编码器输出的4个不同尺度特征图：
      F4 (128, 64²)  →  深层语义（全局结构）
      F3 (64, 128²)  →  中高层特征
      F2 (32, 256²)  →  中低层特征
      F1 (16, 512²)  →  浅层细节（边缘纹理）

    融合为一张 512×512 的单通道预测图。

    流程：
      1. MLP投影：各尺度的通道数统一到8维
      2. DySample：F4/F3/F2 分别上采样8×/4×/2×到512×512
      3. Concat：4个 (B,8,512,512) → (B,32,512,512)
      4. GBC：全局上下文建模（32→32）
      5. BottConv：32→8 降维
      6. 输出：8→1→1 → (B,1,512,512) 分割图
    '''

    def __init__(self, embedding_dim):
        super(MFS, self).__init__()
        self.embedding_dim = embedding_dim  # 8

        # === 4个MLP：把不同通道数的特征投影到统一维度(8) ===
        self.linear_c4 = MLP(input_dim=128, embed_dim=embedding_dim)  # 128→8
        self.linear_c3 = MLP(input_dim=64,  embed_dim=embedding_dim)  # 64→8
        self.linear_c2 = MLP(input_dim=32,  embed_dim=embedding_dim)  # 32→8
        self.linear_c1 = MLP(input_dim=16,  embed_dim=embedding_dim)  # 16→8

        # === GBC：处理拼接后的32维特征，提取全局上下文 ===
        self.GBC_C = GBC(embedding_dim * 4)      # 输入32维
        self.GBC_8 = GBC(8, norm_type='IN')      # InstanceNorm版本
        self.GN_C = nn.GroupNorm(num_channels=embedding_dim*4,
                                 num_groups=embedding_dim*4//16)  # 32维→2组

        # === 融合卷积：32→8 降维 ===
        self.linear_fuse = BottConv(embedding_dim*4, embedding_dim,
                                     embedding_dim//8, kernel_size=1, padding=0, stride=1)

        # === 输出层：8→1→1 生成单通道分割图 ===
        self.linear_pred = BottConv(embedding_dim, 1, 1, kernel_size=1)  # 8→1
        self.linear_pred_1 = nn.Conv2d(1, 1, kernel_size=1)             # 1→1

        self.dropout = nn.Dropout(p=0.1)  # 防止过拟合

        # === DySample 动态上采样：轻量高效的学习型上采样 ===
        self.DySample_C_2 = DySample(embedding_dim, scale=2)    # 2倍上采样
        self.DySample_C_4 = DySample(embedding_dim, scale=4)    # 4倍上采样
        self.DySample_C_8 = DySample(embedding_dim, scale=8)    # 8倍上采样

    def forward(self, inputs, return_feat=False):
        '''
        输入: [F4, F3, F2, F1]
          F4: (B, 128, 64, 64)
          F3: (B, 64,  128, 128)
          F2: (B, 32,  256, 256)
          F1: (B, 16,  512, 512)

        输出: x (B, 1, 512, 512) —— 裂缝概率图(logits)
        '''
        c4, c3, c2, c1 = inputs

        # === F4: 128→8 + 8倍上采样 (64×64→512×512) ===
        b, c, h, w = c4.shape
        out_c4 = self.linear_c4(
            c4.reshape(b, c, h*w).permute(0, 2, 1)    # (B,4096,128) → MLP → (B,4096,8)
        ).permute(0, 2, 1).reshape(b, self.embedding_dim, h, w)  # → (B,8,64,64)
        out_c4 = self.DySample_C_8(out_c4)            # → (B,8,512,512)

        # === F3: 64→8 + 4倍上采样 (128×128→512×512) ===
        b, c, h, w = c3.shape
        out_c3 = self.linear_c3(
            c3.reshape(b, c, h*w).permute(0, 2, 1)
        ).permute(0, 2, 1).reshape(b, self.embedding_dim, h, w)  # → (B,8,128,128)
        out_c3 = self.DySample_C_4(out_c3)            # → (B,8,512,512)

        # === F2: 32→8 + 2倍上采样 (256×256→512×512) ===
        b, c, h, w = c2.shape
        out_c2 = self.linear_c2(
            c2.reshape(b, c, h*w).permute(0, 2, 1)
        ).permute(0, 2, 1).reshape(b, self.embedding_dim, h, w)  # → (B,8,256,256)
        out_c2 = self.DySample_C_2(out_c2)            # → (B,8,512,512)

        # === F1: 16→8 (已经是512×512，不用上采样) ===
        b, c, h, w = c1.shape
        out_c1 = self.linear_c1(
            c1.reshape(b, c, h*w).permute(0, 2, 1)
        ).permute(0, 2, 1).reshape(b, self.embedding_dim, h, w)  # → (B,8,512,512)

        # === 拼接4个尺度 → GBC → 融合 ===
        # (8+8+8+8) = 32维
        out_c = self.GBC_C(torch.cat([out_c4, out_c3, out_c2, out_c1], dim=1))  # GBC(32→32)
        out_c = self.linear_fuse(out_c)  # BottConv(32→8)

        # === 输出：生成单通道分割图 ===
        out_c = self.dropout(out_c)                      # Dropout防过拟合
        x = self.linear_pred_1(self.linear_pred(out_c))  # 8→1→1

        if return_feat:
            return x, out_c   # (B,1,512,512), (B,8,512,512) 中间特征给BoundaryHead
        return x  # (B, 1, 512, 512)
