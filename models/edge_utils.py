"""
edge_utils.py —— FP-ADB 非对称双分支模块（技术贡献一）

包含4个类：
  EdgePriorExtractor : 细节/边缘物理先验提取（内部用4方向Sobel算子做高通滤波）
  DetailPyramid  : 把边缘图逐层缩小，生成D1~D4，匹配编码器4个尺度
  ADPStream      : 上面两个的组合——主动细节保持流
  CFAF           : 把主干特征F_i和边缘特征D_i融合在一起
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.wavelet_utils import HaarDWT


# ==============================================================================
# EdgePriorExtractor —— 细节/边缘物理先验提取
# ==============================================================================
class EdgePriorExtractor(nn.Module):
    """
    细节/边缘物理先验提取模块（Detail/Edge Prior Extractor）。

    内部用4个方向的Sobel算子作为显式几何先验来提取图像边缘：
      0°（水平边缘）  45°（对角线边缘）
      90°（垂直边缘） 135°（反对角线边缘）

    每个方向对RGB三个通道分别做，得到12通道边缘图。
    先验核是固定的、不参与训练——因为边缘检测是物理先验，不需要学；
    只有末尾的1×1通道扩展是可学习的。

    输入: (B, 3, H, W)  RGB图像
    输出: (B, 12, H, W)  4方向×3通道的边缘图
    """
    def __init__(self, out_ch=12):
        super().__init__()
        # 4个Sobel卷积核（3×3，权重固定不变）
        s0   = torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=torch.float32)/4.   # 0°水平
        s90  = s0.t()                                                                  # 90°垂直
        s45  = torch.tensor([[-2,-1,0],[-1,0,1],[0,1,2]], dtype=torch.float32)/4.     # 45°对角
        s135 = torch.tensor([[0,1,2],[-1,0,1],[-2,-1,0]], dtype=torch.float32)/4.     # 135°反对角

        # 把4个核扩展到12个（每方向×RGB3通道）
        kernels = []
        for s in [s0, s45, s90, s135]:
            for _ in range(3):      # RGB三个通道各用一个
                kernels.append(s)
        weight = torch.stack(kernels).unsqueeze(1)  # (12, 1, 3, 3)
        self.register_buffer('weight', weight)       # 注册为buffer，不参与梯度

        # 1×1扩展通道（12→out_ch），让特征更丰富
        self.expand = nn.Conv2d(12, out_ch, 1)

    def forward(self, x):
        # groups=3：RGB三个通道各自独立做Sobel
        edge = F.conv2d(x, self.weight, padding=1, groups=3)  # (B, 12, H, W)
        return self.expand(edge)                                # (B, out_ch, H, W)


# ==============================================================================
# DetailStage —— 细节金字塔的一层（带残差）
# ==============================================================================
class DetailStage(nn.Module):
    """第一段(改通道/下采样) + 第二段(打磨) + 残差连接"""
    def __init__(self, first, second):
        super().__init__()
        self.first = first
        self.second = second

    def forward(self, x):
        y = self.first(x)    # 第一段：改通道 / 下采样
        z = self.second(y)   # 第二段：同通道打磨
        return y + z         # ★ 残差：保护第一段的边缘信息不被冲淡


# ==============================================================================
# DetailPyramid —— 4层细节金字塔
# ==============================================================================
class DetailPyramid(nn.Module):
    """
    从边缘图生成4个不同尺度的细节特征，和编码器4个层的输出一一对应：

      D1 (16,  512×512) ← 最高分辨率，浅层细节（边缘最清晰）
      D2 (32,  256×256) ← 缩小2倍
      D3 (64,  128×128) ← 缩小4倍
      D4 (128,  64×64)  ← 最低分辨率，深层语义边缘（粗糙轮廓）

    第一层(stride=1)保持 512²；后面各层用 Haar 小波 DWT 下采样（尺寸减半、
    通道翻 4 倍保留低频+高频），再卷积调整通道。用 GroupNorm 而不是
    BatchNorm，因为 batch_size=1 时 BN 不稳定。
    """
    def __init__(self, in_ch=12):
        super().__init__()

        # stage1：不下采样（保持 512²），带残差
        self.stage1 = DetailStage(
            first=nn.Sequential(
                nn.Conv2d(in_ch, 16, 3, padding=1, bias=False),
                nn.GroupNorm(4, 16),
                nn.ReLU(True),
            ),
            second=nn.Sequential(
                nn.Conv2d(16, 16, 3, padding=1, bias=False),
                nn.GroupNorm(4, 16),
                nn.ReLU(True),
            ),
        )

        # stage2/3/4：小波 DWT 下采样，带残差
        self.stage2 = self._make_stage(16, 32, 8)     # 512→256, 16→32
        self.stage3 = self._make_stage(32, 64, 8)     # 256→128, 32→64
        self.stage4 = self._make_stage(64, 128, 16)   # 128→64,  64→128

    def _make_stage(self, in_ch, out_ch, groups):
        """DWT 下采样 + 残差的一层：HaarDWT → 卷积调整通道 → 卷积打磨 → 残差"""
        return DetailStage(
            first=nn.Sequential(
                HaarDWT(),                              # in_ch → 4*in_ch，尺寸减半
                nn.Conv2d(4 * in_ch, out_ch, 3, padding=1, bias=False),
                nn.GroupNorm(groups, out_ch),
                nn.ReLU(True),
            ),
            second=nn.Sequential(
                nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
                nn.GroupNorm(groups, out_ch),
                nn.ReLU(True),
            ),
        )

    def forward(self, x):
        d1 = self.stage1(x)   # (B, 16,  512, 512)
        d2 = self.stage2(d1)  # (B, 32,  256, 256)
        d3 = self.stage3(d2)  # (B, 64,  128, 128)
        d4 = self.stage4(d3)  # (B, 128, 64,  64)
        # 返回 [D4, D3, D2, D1]，顺序和编码器输出[F4,F3,F2,F1]对齐
        return [d4, d3, d2, d1]


# ==============================================================================
# ADPStream —— 主动细节保持流
# ==============================================================================
class ADPStream(nn.Module):
    """
    ADP-Stream = 边缘先验提取 + 细节金字塔

    这就是论文里的"主动细节保持流"——一条和主干编码器并行的轻量分支，
    专门提取裂缝的边缘细节信息。

    参数总量 < 15K，极轻量。

    输入: 原始RGB图像 (B, 3, H, W)
    输出: [D4, D3, D2, D1] —— 4个尺度的边缘细节特征
    """
    def __init__(self, in_ch=12, pyramid_in_ch=12):
        super().__init__()
        self.sobel = EdgePriorExtractor(in_ch)      # 第1步：提取细节/边缘先验
        self.pyramid = DetailPyramid(pyramid_in_ch) # 第2步：生成多尺度

    def forward(self, x):
        edge = self.sobel(x)       # → 12通道边缘图
        return self.pyramid(edge)  # → [D4, D3, D2, D1]


# ==============================================================================
# CFAF —— 交叉特征对齐融合
# ==============================================================================
class CFAF(nn.Module):
    """
    把主干特征 F_i 和边缘细节 D_i 融合在一起。

    核心公式: F'_i = F_i + alpha × D_i

    alpha 是学出来的权重图（每个像素一个权重）：
      - 裂缝区域 → alpha 大 → 采纳边缘细节，边界更清晰
      - 背景区域 → alpha 小 → 保持主干语义，不被边缘噪声干扰

    安全设计: bias初始=-2.0 → sigmoid(-2.0)≈0.12 → 边缘一开始只贡献12%
    这样主干先学会基本分割，边缘增强慢慢加上来。
    """
    def __init__(self, channels):
        """
        channels: F_i和D_i的通道数（各尺度不同: 128/64/32/16）
        """
        super().__init__()
        mid = channels // 4  # 中间通道数缩小4倍，省参数

        # 把主干特征和边缘特征都投影到同一个低维空间
        self.proj_F = nn.Sequential(
            nn.Conv2d(channels, mid, 1, bias=False),
            nn.GroupNorm(max(1, mid//4), mid),
        )
        self.proj_D = nn.Sequential(
            nn.Conv2d(channels, mid, 1, bias=False),
            nn.GroupNorm(max(1, mid//4), mid),
        )

        # 从拼接后的特征算出每个像素的边缘可信度alpha
        self.attn = nn.Sequential(
            nn.Conv2d(mid * 2, mid, 3, padding=1, bias=False),  # 3×3看周围像素
            nn.GroupNorm(max(1, mid//4), mid),
            nn.ReLU(True),
            nn.Conv2d(mid, 1, 1),                                 # → 单通道权重图
        )

        # bias=-2: sigmoid(-2)=0.12 → 边缘最初只贡献12%
        self.bias = nn.Parameter(torch.tensor(-2.0))

    def forward(self, F_i, D_i):
        """
        输入:
          F_i: (B, C, H, W) 主干特征（编码器某层的输出）
          D_i: (B, C, H, W) 边缘细节（ADP-Stream同尺度的输出）
        输出:
          (B, C, H, W) 融合后的特征
        """
        # 步骤1: 把主干和边缘都投影到低维空间
        f = self.proj_F(F_i)   # F_i → 低维
        d = self.proj_D(D_i)   # D_i → 低维
        # 第一步：拼接得到“待审阅”的数据
        # 假设在某个像素位置(h, w)，它有
        # f（主干特征）和
        # d（边缘特征）两套数据，拼接后得到
        # mid * 2
        # 个通道。此时该像素拥有
        # mid * 2
        # 个数值。
        #
        # 第二步：3×3
        # 卷积 —— 收集“周围邻居”的意见
        # 代码中的
        # nn.Conv2d(mid * 2, mid, 3, padding=1)
        # 会生成
        # mid
        # 个输出通道。
        #
        # 怎么算的？ 对于输出的每一个通道，都有一个独立的
        # 3×3
        # 卷积核（厚度为
        # mid * 2）。
        #
        # 在像素(h, w)
        # 处：这个卷积核会覆盖以(h, w)
        # 为中心的
        # 9
        # 个像素（即该像素本身 + 8
        # 个邻居）。
        #
        # 数学操作：把覆盖到的
        # 9
        # 个像素，每个像素的
        # mid * 2
        # 个数值，与卷积核里对应的权重相乘后累加，得到一个单一数值。
        #
        # 结果：mid
        # 个卷积核各自算出一个数值，所以输出变成了
        # mid
        # 个通道。
        #
        # 作用：这一步告诉网络——“要判断这个像素重不重要，不能只看它自己，还要看看它周围的
        # 8
        # 个邻居是不是也像裂缝边缘。” 这引入了空间上下文。
        #
        # 第三步：1×1
        # 卷积 —— 综合意见，拍板打分
        # 代码中的
        # nn.Conv2d(mid, 1, 1)
        # 会把
        # mid
        # 个通道压缩成
        # 1
        # 个通道。
        #
        # 怎么算的？ 此时卷积核大小是
        # 1×1，意味着它只看当前像素，不看邻居。
        #
        # 数学操作：它把刚才算出来的
        # mid
        # 个数值（来自
        # 3×3
        # 卷积的结果），按照各自的权重线性加权求和，得到一个最终的标量分数
        # raw_score（原始分数）。
        #
        # 作用：这一步相当于“领导签字”——把之前收集的所有局部证据（mid
        # 个特征）汇总成一个结论：“此像素是裂缝边缘的可信度高低”。
        #
        # 第四步：加偏置 + Sigmoid → 得到
        # alpha
        # 代码中最后是
        # alpha = torch.sigmoid(self.attn(feat) + self.bias)。
        #
        # 加偏置：raw_score + (-2.0)。因为
        # bias
        # 初始化为 - 2.0，所以原始分数被强行拉低。
        #
        # Sigmoid
        # 压缩：将分数映射到(0, 1)
        # 区间。
        #
        # 如果得分极大，alpha
        # 趋近于
        # 1（采纳边缘）。
        #
        # 如果得分极小（被 - 2.0
        # 压制），alpha
        # 趋近于
        # 0（忽略边缘）。
        # 步骤2: 拼接 → 3×3卷积 → sigmoid → 得到权重图alpha
        feat = torch.cat([f, d], dim=1)                   # 拼接
        alpha = torch.sigmoid(self.attn(feat) + self.bias) # alpha ∈ [0,1]

        # 步骤3: 残差融合 ← 主干原样保留，边缘按需增强
        return F_i + alpha * D_i
