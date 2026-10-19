'''
Author: Hui Liu
Github: https://github.com/Karl1109
Email: liuhui@ieee.org

================================================================================
models/PAF.py —— 像素注意力融合（Pixel Attention Fusion）
================================================================================

PAF 在 WAVSS_Layer 的末尾被调用，作用是将 SSM 输出与原始特征融合。

┌─────────────────────────────────────────────────────────────────┐
│ PAF 的工作流程:                                                   │
│                                                                  │
│   base_feat (原始特征)    guidance_feat (SSM输出/引导特征)         │
│        │                        │                                 │
│        │            ┌───────────┴───────────┐                    │
│        │        feature_transform      feature_transform          │
│        │        (BottConv→BN)          (BottConv→BN)             │
│        │            │                       │                    │
│        │        base_key              guidance_query             │
│        │            │                       │                    │
│        │            └──────────×───────────┘                     │
│        │                     │                                    │
│        │              channel_adapter                             │
│        │              (BottConv→BN)                               │
│        │                     │                                    │
│        │              sigmoid → similarity_map                    │
│        │                     │                                    │
│   ┌──────────────────────────┼──────────────────────────┐        │
│   │                          │                          │        │
│   ▼                          ▼                          ▼        │
│  (1 - similarity_map)  ×  base_feat   +   similarity_map  ×  guidance_feat
│                          └────────────────┬────────────────┘
│                                           ▼
│                                     fused_feature（输出）
└─────────────────────────────────────────────────────────────────┘

直观理解：
  - similarity_map 是一个 [0,1] 的权重图，表示每个位置相信 guidance 的程度
  - 裂缝区域 → similarity_map 接近1 → 更多采纳SSM输出（强调裂缝连续性）
  - 背景区域 → similarity_map 接近0 → 更多保留原始特征（保持背景稳定）
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.GBC import BottConv  # → GBC.py


class PAF(nn.Module):
    '''
    PAF = Pixel Attention Fusion（像素注意力融合）

    输入: base_feat（原始特征图）和 guidance_feat（SSM引导特征）
    输出: 融合后的特征图

    核心思想: 学一个相似度图(similarity_map)，用它来混合两个特征:
      fused = (1 - map) × base + map × guidance

    这类似于一个可学习的"alpha blending"
    '''

    def __init__(self,
                 in_channels: int,         # 输入通道数 = 256
                 mid_channels: int,        # 中间通道数 = 128 (=256//2)
                 after_relu: bool = False, # 是否先用ReLU激活（默认不用）
                 mid_norm: nn.Module = nn.BatchNorm2d,
                 in_norm: nn.Module = nn.BatchNorm2d):
        super().__init__()
        self.after_relu = after_relu

        # 特征变换：把输入特征映射到低维空间进行比较
        # BottConv(256→128, 瓶颈=16, kernel=1)
        self.feature_transform = nn.Sequential(
            BottConv(in_channels, mid_channels, mid_channels=16, kernel_size=1),
            mid_norm(mid_channels)
        )

        # 通道适配器：将比较结果映射回原始通道数
        self.channel_adapter = nn.Sequential(
            BottConv(mid_channels, in_channels, mid_channels=16, kernel_size=1),
            in_norm(in_channels)
        )

        if after_relu:
            self.relu = nn.ReLU(inplace=True)

    def forward(self, base_feat: torch.Tensor, guidance_feat: torch.Tensor) -> torch.Tensor:
        '''
        参数:
          base_feat:     原始特征 (B, 256, 64, 64)  — WAVSS_Layer输入
          guidance_feat: 引导特征 (B, 256, 64, 64)  — SSM扫描输出

        返回:
          fused_feature: (B, 256, 64, 64)
        '''
        base_shape = base_feat.size()

        # 可选ReLU预处理
        if self.after_relu:
            base_feat = self.relu(base_feat)
            guidance_feat = self.relu(guidance_feat)

        # 步骤1: 把两个特征都映射到128维 → base_key 和 guidance_query
        guidance_query = self.feature_transform(guidance_feat)  # (B, 128, H, W)
        base_key = self.feature_transform(base_feat)             # (B, 128, H, W)

        # 步骤2: 对齐尺寸（如果guidance_feat和base_feat尺寸不同）
        guidance_query = F.interpolate(
            guidance_query,
            size=[base_shape[2], base_shape[3]],
            mode='bilinear', align_corners=False
        )

        # 步骤3: 计算相似度图（逐元素乘积 → 通道适配 → sigmoid）
        # base_key * guidance_query: 衡量两个特征在每个位置的相似程度
        # sigmoid → [0,1]: 得到每个位置的融合权重
        similarity_map = torch.sigmoid(
            self.channel_adapter(base_key * guidance_query)
        )  # (B, 256, H, W)

        # 步骤4: 对齐guidance_feat尺寸
        resized_guidance = F.interpolate(
            guidance_feat,
            size=[base_shape[2], base_shape[3]],
            mode='bilinear', align_corners=False
        )

        # 步骤5: ★ 加权融合 = (1-α)×base + α×guidance
        # 裂缝像素: α接近1 → 输出靠近guidance（SSM的连续性增强结果）
        # 背景像素: α接近0 → 输出靠近base（原始特征保持稳定）
        fused_feature = (1 - similarity_map) * base_feat + similarity_map * resized_guidance

        return fused_feature
