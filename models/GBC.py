'''
Author: Hui Liu
Github: https://github.com/Karl1109
Email: liuhui@ieee.org

================================================================================
models/GBC.py —— Gated Bottleneck Convolution（门控瓶颈卷积）
================================================================================

GBC 是论文提出的轻量卷积模块，在 WAVSS 网络中被多次调用：
  - WAVSS_Layer 的开头（2次GBC：建模裂缝形态）
  - MFS 的融合阶段（GBC：全局上下文建模）
  - WAVSS 的输出转换层（BottConv：256→128/64/32/16）

包含两个组件：
  ┌─────────────────────────────────────────────────────────┐
  │ BottConv —— 瓶颈卷积（核心积木）                          │
  │   1×1(pointwise) → Depthwise → 1×1(pointwise)           │
  │   例：256 → 32(瓶颈) → 32 → 256                          │
  │   意义：比标准卷积参数少约8倍，计算量小得多                  │
  ├─────────────────────────────────────────────────────────┤
  │ GBC —— 门控瓶颈卷积块                                     │
  │   Block1(3×3卷积) → Block2(3×3卷积)                      │
  │   Block3(1×1卷积) ────────────┐                          │
  │           ↓                    ↓                          │
  │   Block1→Block2 输出 × Block3 输出 ←（门控：1×1门控3×3）  │
  │           ↓                                              │
  │   Block4(1×1卷积) + 残差连接                              │
  └─────────────────────────────────────────────────────────┘
'''

import torch.nn as nn


# ============================================================================
# BottConv —— 瓶颈卷积（Depthwise Separable的变体）
# ============================================================================
class BottConv(nn.Module):
    '''
    瓶颈卷积 = 1×1降维 → Depthwise卷积 → 1×1升维

    例：BottConv(256, 128, 32, kernel_size=3)
      in=256 → 1×1→32(瓶颈) → 3×3深度卷积 → 1×1→128(out)

    为什么用瓶颈？
      - 标准 3×3 卷积参数量: 256×128×3×3 = 294,912
      - 瓶颈卷积参数量:   256×32 + 32×3×3 + 32×128 = 12,320
      - 减少了约96%的参数！

    瓶颈层(mid_channels)越小，压缩越狠，参数越少。
    论文在裂缝分割这个小模型里用 mid_channels = in_channels//8。
    '''
    def __init__(self, in_channels, out_channels, mid_channels,
                 kernel_size, stride=1, padding=0, bias=True):
        super(BottConv, self).__init__()

        # 1×1降维：in_channels → mid_channels（压缩阶段）
        self.pointwise_1 = nn.Conv2d(in_channels, mid_channels, 1, bias=bias)

        # Depthwise卷积：每个通道单独卷积，groups=mid_channels
        self.depthwise = nn.Conv2d(mid_channels, mid_channels,
                                    kernel_size, stride, padding,
                                    groups=mid_channels, bias=False)

        # 1×1升维：mid_channels → out_channels（扩张阶段）
        self.pointwise_2 = nn.Conv2d(mid_channels, out_channels, 1, bias=False)

    def forward(self, x):
        x = self.pointwise_1(x)   # 1×1压缩
        x = self.depthwise(x)     # 空间卷积（保持通道数不变）
        x = self.pointwise_2(x)   # 1×1扩张到目标通道
        return x


# ============================================================================
# 归一化层工具函数
# ============================================================================
def get_norm_layer(norm_type, channels, num_groups):
    '''
    根据类型创建归一化层
    - 'GN': GroupNorm（分组归一化，小batch稳定）
    - 其他: InstanceNorm2d（逐样本归一化）
    '''
    if norm_type == 'GN':
        return nn.GroupNorm(num_groups=num_groups, num_channels=channels)
    else:
        return nn.InstanceNorm2d(channels)


# ============================================================================
# GBC —— 门控瓶颈卷积块（Gated Bottleneck Convolution Block）
# ============================================================================
class GBC(nn.Module):
    '''
    GBC = Gated Bottleneck Convolution Block

    结构（共4个子块）:
      ┌─────────────────────────────────────────┐
      │                输入 x                     │
      │         ┌──────────┬──────────┐          │
      │     Block1       Block3               │
      │  (3×3卷积+GN+ReLU) (1×1卷积+GN+ReLU)    │
      │         │           │                    │
      │     Block2          │                    │
      │  (3×3卷积+GN+ReLU)  │                    │
      │         │           │                    │
      │         └────×──────┘  ← 逐元素乘法(门控)  │
      │              │                           │
      │          Block4                          │
      │       (1×1卷积+GN+ReLU)                   │
      │              │                           │
      │         └── + ──┘  ← 残差连接(x + out)    │
      │              │                           │
      │            输出                           │
      └─────────────────────────────────────────┘

    门控机制：
      Block1→Block2 是 3×3 路径（大感受野，感知形态）
      Block3 是 1×1 路径（小感受野，逐点门控）
      两者逐元素相乘 = 用1×1的输出门控3×3的输出
      → 模型学会自适应地选择哪些空间位置的特征重要
    '''

    def __init__(self, in_channels, norm_type='GN'):
        super(GBC, self).__init__()

        # Block1: 3×3瓶颈卷积（空间特征提取）
        self.block1 = nn.Sequential(
            BottConv(in_channels, in_channels, in_channels // 8, 3, 1, 1),
            get_norm_layer(norm_type, in_channels, in_channels // 16),
            nn.ReLU()
        )

        # Block2: 3×3瓶颈卷积（继续提取空间特征）
        self.block2 = nn.Sequential(
            BottConv(in_channels, in_channels, in_channels // 8, 3, 1, 1),
            get_norm_layer(norm_type, in_channels, in_channels // 16),
            nn.ReLU()
        )

        # Block3: 1×1瓶颈卷积（门控路径）
        self.block3 = nn.Sequential(
            BottConv(in_channels, in_channels, in_channels // 8, 1, 1, 0),
            get_norm_layer(norm_type, in_channels, in_channels // 16),
            nn.ReLU()
        )

        # Block4: 1×1瓶颈卷积（融合输出）
        self.block4 = nn.Sequential(
            BottConv(in_channels, in_channels, in_channels // 8, 1, 1, 0),
            get_norm_layer(norm_type, in_channels, 16),
            nn.ReLU()
        )

    def forward(self, x):
        '''
        输入:  x (B, C, H, W)
        输出:  x + GBC(x) (B, C, H, W)，带残差连接

        具体步骤：
          1. Block1(x) → Block2 → f1（3×3特征路径）
          2. Block3(x) → f2（1×1门控路径）
          3. f1 × f2（门控：3×3特征被1×1门控加权）
          4. Block4(f1×f2) + x（残差连接）
        '''
        residual = x                    # 保存原始输入用于残差

        # 3×3 路径：空间特征提取
        x1 = self.block1(x)            # Block1: 3×3卷积
        x1 = self.block2(x1)           # Block2: 3×3卷积

        # 1×1 路径：门控
        x2 = self.block3(x)            # Block3: 1×1卷积

        # 门控融合
        x = x1 * x2                     # ★ 逐元素乘法（门控机制）

        # 输出 + 残差
        x = self.block4(x)             # Block4: 1×1融合

        return x + residual             # ★ 残差连接（帮助梯度传播）
