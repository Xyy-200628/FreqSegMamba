"""
patch_embed.py —— 把一张大图切成小方块，方便模型处理

做什么: (3, 512, 512) 的照片 → (4096, 256) 的"拼图片段"
        每张图切成 64×64=4096 个小块，每个块用 256 个数字描述

数据变化: 512×512 → stem缩一半(256×256) → convs保持(256×256)
                  → projection缩4倍(64×64) → flatten拍平(4096, 256)
"""
import torch
from mmcv.cnn.bricks.transformer import AdaptivePadding
from mmcv.cnn import (build_conv_layer, build_norm_layer)
from mmcv.runner.base_module import BaseModule
from mmcv.utils import to_2tuple
from torch import nn

class ConvPatchEmbed(BaseModule):
    """图片→序列的转换器。用卷积层把大图切成小方块"""

    def __init__(self,
                 in_channels=3,      # 输入：RGB，3个颜色通道
                 embed_dims=768,     # 输出：每个小方块用几个数字表示（实际用256）
                 num_convs=0,        # 中间多几层卷积（实际用2）
                 conv_type='Conv2d', # 卷积类型
                 patch_size=16,      # 每个小方块多大（实际用8）
                 stride=16,          # 每次跨几步（等于patch_size）
                 padding='corner',   # 边缘怎么补
                 dilation=1,         # 卷积膨胀率
                 bias=True,          # 是否加偏置
                 norm_cfg=None,      # 归一化设置
                 input_size=None,    # 输入图片尺寸
                 init_cfg=None):     # 初始化设置
        super(ConvPatchEmbed, self).__init__(init_cfg=init_cfg)

        # stride实际上用 patch_size // 2
        # 因为下面stem已经缩了2倍，project再缩 (patch_size//2) 倍
        # 总缩放 = 2 × (8//2) = 8倍 (512→64)
        self.embed_dims = embed_dims
        if stride is None:
            stride = patch_size // 2
        else:
            stride = stride // 2          # 8 // 2 = 4

        # ===== 第1层: Stem =====
        # 7×7大窗, 2步一跳 → 图缩小一半(512→256), 通道3→64
        self.stem = torch.nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=(7,7), stride=(2,2),
                      padding=3, bias=False),
            nn.GroupNorm(num_channels=64, num_groups=4),  # 归一化: 稳定数值
            nn.ReLU(True))                                 # 激活: 负数变成0

        # ===== 第2-3层: 额外卷积(num_convs=2) =====
        # 3×3小窗, 1步一跳 → 图大小不变, 64→64
        # 这两层不改变大小, 只是"打磨"特征
        if num_convs > 0:
            convs = []
            for _ in range(num_convs):
                convs.append(nn.Conv2d(64, 64, (3,3), (1,1), padding=1, bias=False))
                convs.append(nn.GroupNorm(num_channels=64, num_groups=4))
                convs.append(nn.ReLU(True))
            self.convs = torch.nn.Sequential(*convs)
        else:
            self.convs = None

        # ===== 第4层: Projection =====
        # 4×4窗, 4步一跳 → 图缩小4倍(256→64), 通道64→256
        kernel_size = to_2tuple(patch_size//2)     # (4, 4)
        stride = to_2tuple(stride)                  # (4, 4)
        dilation = to_2tuple(dilation)              # (1, 1) 修复：dilation也要转tuple

        # 自适应边距（如果需要的话，本项目中没用到）
        if isinstance(padding, str):
            self.adaptive_padding = AdaptivePadding(
                kernel_size=kernel_size, stride=stride,
                dilation=dilation, padding=padding)
            padding = (0, 0)  # 因为用了adaptive_padding，conv本身的padding设为0
        else:
            self.adaptive_padding = None
            padding = to_2tuple(padding)  # 不用adaptive_padding时，padding可能是int需要转tuple

        self.projection = nn.Conv2d(
            in_channels=64,          # 输入64维
            out_channels=embed_dims, # 输出256维
            kernel_size=kernel_size, # 窗口4×4
            stride=stride,           # 4步一跳
            bias=bias
        )

        # 可选归一化（本项目没用到）
        if norm_cfg is not None:
            self.norm = build_norm_layer(norm_cfg, embed_dims)[1]
        else:
            self.norm = None

        # ===== 算一下输出尺寸 =====
        if input_size:
            input_size = to_2tuple(input_size)  # (512, 512)
            self.init_input_size = input_size
            _input_size = (input_size[0] // 2, input_size[1] // 2)
            # stem 之后 = (256, 256)

            if self.adaptive_padding:
                pad_h, pad_w = self.adaptive_padding.get_pad_shape(_input_size)
                _input_size = (_input_size[0] + pad_h, _input_size[1] + pad_w)

            # 卷积输出大小公式: (输入 + 2×边距 - 膨胀×(核-1) - 1) / 步长 + 1
            h_out = (_input_size[0] + 2 * padding[0] - dilation[0] *
                     (kernel_size[0] - 1) - 1) // stride[0] + 1
            w_out = (_input_size[1] + 2 * padding[1] - dilation[1] *
                     (kernel_size[1] - 1) - 1) // stride[1] + 1
            self.init_out_size = (h_out, w_out)  # = (64, 64)
        else:
            self.init_input_size = None
            self.init_out_size = None

    def forward(self, x):
        """
        输入: x = (B, 3, 512, 512)
              B = 一次喂几张图
              3 = RGB三个颜色

        输出: x = (B, 4096, 256)  ← 4096个小方块，每个用256个数字描述
              out_size = (64, 64)  ← 空间上是64行×64列
        """
        # 第1步: stem → 512→256, 3→64维
        x = self.stem(x)

        # 第2步: convs → 保持256×256, 加强特征
        if self.convs is not None:
            x = self.convs(x)

        # 第3步: 自适应边距
        if self.adaptive_padding:
            x = self.adaptive_padding(x)

        # 第4步: projection → 256→64, 64→256维
        x = self.projection(x)
        out_size = (x.shape[2], x.shape[3])      # (64, 64)

        # 第5步: 拍平 → (B,256,64,64) 变成 (B,4096,256)
        #       像把64×64的棋盘一格一格排成一排
        x = x.flatten(2).transpose(1, 2)

        if self.norm is not None:
            x = self.norm(x)

        return x, out_size
