"""
wavelet_utils.py —— Haar 小波的 DWT/IDWT（用卷积实现，GPU友好）

作用: 把特征图拆成4个子带 → 低频走SSM → 高频走Conv → 合并回来

DWT 分解:
  LL (低频近似):  [[1, 1], [1, 1]] / 2     → 保留裂缝主体轮廓
  LH (水平高频):  [[1, -1], [1, -1]] / 2   → 保留水平方向边缘
  HL (垂直高频):  [[1, 1], [-1, -1]] / 2   → 保留垂直方向边缘
  HH (对角高频):  [[1, -1], [-1, 1]] / 2   → 保留对角线方向边缘

IDWT 重构: 用转置卷积把4个子带合并回原始尺寸

为什么用卷积而不是 pywt？
  pywt 在 CPU 上逐张图算，太慢。
  卷积直接跑在 GPU 上，和网络其他部分无缝衔接。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ==============================================================================
# Haar 小波滤波器核
# ==============================================================================
def _make_haar_kernels():
    """
    创建4个2×2的Haar小波卷积核，每个核处理"逐通道"卷积（groups=C）。

    返回: (LL, LH, HL, HH) 四个核, shape都是 (1, 1, 2, 2)
    """
    ll = torch.tensor([[[[1., 1.], [1., 1.]]]]) / 2.   # 低通平均
    lh = torch.tensor([[[[1., -1.], [1., -1.]]]]) / 2. # 水平差异
    hl = torch.tensor([[[[1., 1.], [-1., -1.]]]]) / 2. # 垂直差异
    hh = torch.tensor([[[[1., -1.], [-1., 1.]]]]) / 2. # 对角差异
    return ll, lh, hl, hh


class HaarDWT(nn.Module):
    """
    Haar 小波分解: (B, C, H, W) → (B, 4C, H/2, W/2)

    4个通道组: [LL, LH, HL, HH]
    """
    def __init__(self):
        super().__init__()
        ll, lh, hl, hh = _make_haar_kernels()
        # 每个核扩展为 (C, 1, 2, 2) 以支持 groups=C 逐通道卷积
        self.register_buffer('ll', ll)
        self.register_buffer('lh', lh)
        self.register_buffer('hl', hl)
        self.register_buffer('hh', hh)

    def forward(self, x):
        """
        输入: (B, C, H, W)
        输出: (B, 4*C, H//2, W//2) → [LL, LH, HL, HH] 在通道维拼接
        """
        B, C, H, W = x.shape
        # 每个核做 groups=C 卷积: 每个通道用自己的2×2核，stride=2 → 尺寸减半
        LL = F.conv2d(x, self.ll.expand(C, 1, 2, 2), stride=2, groups=C)
        LH = F.conv2d(x, self.lh.expand(C, 1, 2, 2), stride=2, groups=C)
        HL = F.conv2d(x, self.hl.expand(C, 1, 2, 2), stride=2, groups=C)
        HH = F.conv2d(x, self.hh.expand(C, 1, 2, 2), stride=2, groups=C)
        return torch.cat([LL, LH, HL, HH], dim=1)


class HaarIDWT(nn.Module):
    """
    Haar 小波重构: (B, 4*C, H/2, W/2) → (B, C, H, W)

    用转置卷积把4个子带合并回原始分辨率。每个通道组的贡献相加。
    """
    def __init__(self):
        super().__init__()
        ll, lh, hl, hh = _make_haar_kernels()
        self.register_buffer('ll', ll)
        self.register_buffer('lh', lh)
        self.register_buffer('hl', hl)
        self.register_buffer('hh', hh)

    def forward(self, x):
        """
        输入: (B, 4*C, H_half, W_half) → 4*C = LL_C + LH_C + HL_C + HH_C
        输出: (B, C, H_half*2, W_half*2)
        """
        B, total_C, H_half, W_half = x.shape
        C = total_C // 4

        # 拆分4个通道组
        LL = x[:, 0*C:1*C, :, :]
        LH = x[:, 1*C:2*C, :, :]
        HL = x[:, 2*C:3*C, :, :]
        HH = x[:, 3*C:4*C, :, :]

        # 转置卷积: 每个子带用对应的核 "反卷积" 回原始尺寸，stride=2 上采样2倍
        out_LL = F.conv_transpose2d(LL, self.ll.expand(C, 1, 2, 2), stride=2, groups=C)
        out_LH = F.conv_transpose2d(LH, self.lh.expand(C, 1, 2, 2), stride=2, groups=C)
        out_HL = F.conv_transpose2d(HL, self.hl.expand(C, 1, 2, 2), stride=2, groups=C)
        out_HH = F.conv_transpose2d(HH, self.hh.expand(C, 1, 2, 2), stride=2, groups=C)

        # 4个贡献相加 = 原始特征图
        return out_LL + out_LH + out_HL + out_HH


class HighFreqEnhance(nn.Module):
    """
    高频子带增强模块: 对 LH/HL/HH 做轻量卷积保留边缘细节

    每个子带: BottConv(C → C//16 → C, 3×3) + SiLU

    极轻量，几乎不增加参数。
    """
    def __init__(self, channels):
        super().__init__()
        bottleneck = channels // 16

        # 3个子带各自独立增强
        self.conv_lh = nn.Sequential(
            nn.Conv2d(channels, bottleneck, 1),
            nn.Conv2d(bottleneck, bottleneck, 3, padding=1, groups=bottleneck),
            nn.Conv2d(bottleneck, channels, 1),
            nn.SiLU()
        )
        self.conv_hl = nn.Sequential(
            nn.Conv2d(channels, bottleneck, 1),
            nn.Conv2d(bottleneck, bottleneck, 3, padding=1, groups=bottleneck),
            nn.Conv2d(bottleneck, channels, 1),
            nn.SiLU()
        )
        self.conv_hh = nn.Sequential(
            nn.Conv2d(channels, bottleneck, 1),
            nn.Conv2d(bottleneck, bottleneck, 3, padding=1, groups=bottleneck),
            nn.Conv2d(bottleneck, channels, 1),
            nn.SiLU()
        )

    def forward(self, lh, hl, hh):
        """
        输入: lh(B,C,H,W), hl(B,C,H,W), hh(B,C,H,W)
        输出: 增强后的三个张量
        """
        return self.conv_lh(lh), self.conv_hl(hl), self.conv_hh(hh)
