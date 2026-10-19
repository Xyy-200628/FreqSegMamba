'''
Author: Hui Liu
Github: https://github.com/Karl1109
Email: liuhui@ieee.org

====================================================================================
WAVSS_layer.py —— ② 每层核心实现（WAVSS 网络的计算单元）
====================================================================================

本文件包含两个关键类：

┌──────────────────────────────────────────────────────────────────────────────┐
│                                                                              │
│  WAVSS_2D  —— 2D选择性状态空间模型（论文核心：SASS扫描 + SSM计算）             │
│    ├── sass()           : 结构感知扫描策略 —— 4个方向的扫描顺序               │
│    ├── in_proj          : 输入投影 (d_model → 2×d_inner)                     │
│    ├── conv2d           : GBC卷积（3×3瓶颈卷积，建模裂缝形态）                 │
│    ├── x_proj / dt_proj : SSM参数投影                                        │
│    └── selective_scan_fn: 选择性扫描（mamba-ssm算子，核心计算）                 │
│                                                                              │
│  WAMambaBlock —— 一层的完整封装                                                │
│    ├── GBC_C ×2         : 全局上下文建模（开头2次GBC卷积）                     │
│    ├── LayerNorm        : 归一化                                             │
│    ├── WAVSS_2D         : SSM扫描+计算                                       │
│    ├── PAF              : 像素注意力融合 → 增强裂缝连续性                      │
│    ├── GN + Linear      : 归一化+线性变换                                    │
│    └── 残差连接          : mixed_x + mixed_x_res                             │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘

SSM 计算流程（WAVSS_2D.forward）:
  输入 x: (B, L=4096, C=256)
  ┌─────────────────────────────────────────────────────────────────┐
  │ 1. in_proj: x → (x, z) 各 d_inner=512维                         │
  │ 2. GBC Conv: x → 2D reshape → conv2d(3×3) → SiLU → 展平        │
  │ 3. x_proj: x → (dt, B, C) 三个SSM参数                           │
  │ 4. dt_proj: dt → Δ (时间步长)                                   │
  │ 5. SASS: 生成4个方向的扫描索引序列                                │
  │ 6. selective_scan_fn ×4: 沿4个方向做选择性扫描（并行）             │
  │ 7. y = sum(y1,y2,y3,y4) × SiLU(z)  ← 门控融合4方向结果           │
  │ 8. out_proj: y → output                                         │
  └─────────────────────────────────────────────────────────────────┘
'''

import math
from einops import repeat                          # 重复张量：用于创建A矩阵
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn.bricks.transformer import build_dropout
from mmcv.cnn.utils.weight_init import trunc_normal_
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn  # ★ SSM核心算子
from mamba_ssm.ops.triton.layer_norm import RMSNorm                    # RMS归一化
from models.GBC import GBC, BottConv                                   # → GBC.py：瓶颈卷积
from models.PAF import PAF                                             # → PAF.py：像素注意力融合
from models.wavelet_utils import HaarDWT, HaarIDWT, HighFreqEnhance    # ★ 小波模块
from models.tpas import TPAS, sass_scan, scan_orders                # ★ 拓扑保持扫描 / 固定SASS扫描 / 固定扫描策略消融


# ============================================================================
# ============================================================================
class WAVSS(nn.Module):
    """
    ┌─────────────────────────────────────────────────────────────────┐
    │  WAVSS：Wavelet-Aware Visual State Space                       │
    │  论文第2个创新点 —— 小波域时频双域协同                            │
    ├─────────────────────────────────────────────────────────────────┤
    │                                                                  │
    │  核心思想:                                                        │
    │    DWT 把特征分成 4 个子带，各走各的路：                            │
    │    - LL(低频, 裂缝轮廓) → SSM 扫描长程依赖                          │
    │    - LH/HL/HH(高频, 裂缝边缘) → 各自 Conv 增强                      │
    │    - 最后 IDWT 拼回来                                             │
    │                                                                  │
    │  好处:                                                            │
    │    1. SSM 只跑在 LL 上，尺寸减半 → 序列长度 1/4，计算量 1/4          │
    │    2. 高频单独保留 → 微细边缘不会被 SSM 抹平                        │
    │    3. α 从 0 开始学 → 不会崩，epoch 0 和原始模型一模一样             │
    │                                                                  │
    └─────────────────────────────────────────────────────────────────┘
    """
    def __init__(self, d_model, d_state=16, expand=2, dt_rank="auto",
                 dt_min=0.001, dt_max=0.1, dt_init="random", dt_scale=1.0,
                 dt_init_floor=1e-4, conv_size=7, bias=False, conv_bias=False,
                 init_layer_scale=None, default_hw_shape=None,
                 use_wavelet=True, use_tpas=True, scan_strategy='sass',
                 freq_mode='default', tpas_mode='sass_dir',
                 tpas_tau_coh=0.35, tpas_tau_win=12):
        super().__init__()
        # ---- 基础参数 ----
        self.d_model = d_model          # 输入通道 = 256
        self.d_state = d_state          # SSM 状态维度 = 16
        self.expand = expand            # 内部扩展倍数 = 2
        self.d_inner = int(expand * d_model)    # 内部工作通道 = 256×2 = 512
        self.dt_rank = math.ceil(d_model / 16) if dt_rank == "auto" else dt_rank  # dt的秩 = 16
        self.n_directions = 4           # 4个扫描方向
        self.use_wavelet = use_wavelet  # 小波开关（消融）
        self.use_tpas = use_tpas        # TPAS/SASS开关（消融）
        self.scan_strategy = scan_strategy  # 固定扫描策略（仅 use_tpas=False 时生效）
        self.freq_mode = freq_mode      # 频域-空域角色分配 [default|swap|all_ssm|all_conv]
        self.tpas_mode = tpas_mode      # TPAS方向码模式 [sass_dir|theta_dir|theta_soft|theta_path|adaptive4|greedy|hybrid]

        # ===== 第1部分：和原始 WAVSS_2D 一样的 SSM 参数 =====
        # 输入投影：256 → 1024（=2×512, x和z各一半）
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=bias)

        # GBC卷积：在2D特征上做瓶颈卷积，感知裂缝形态
        self.conv2d = BottConv(self.d_inner, self.d_inner, self.d_inner // 16,
                                kernel_size=3, padding=1, stride=1)

        self.act = nn.SiLU()   # 激活函数

        # SSM参数生成: 从特征算出 dt(时间步长), B(输入投影), C(输出投影)
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2, bias=False)
        # dt的投影层：把16维的粗dt映射到512维
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        # ---- dt(Δ) 初始化：控制时间步长的初始范围 ----
        dt_init_std = self.dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        dt = torch.exp(torch.rand(self.d_inner) *
                       (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
                       ).clamp(min=dt_init_floor)
        with torch.no_grad():
            self.dt_proj.bias.copy_(dt + torch.log(-torch.expm1(-dt)))
        self.dt_proj.bias._no_reinit = True

        # ---- A 矩阵(状态转移矩阵)：控制历史信息衰减速度 ----
        # 形状 (512, 16)，每行是 1,2,3,...,16 的对数值
        A = repeat(torch.arange(1, self.d_state + 1, dtype=torch.float32),
                   "n -> d n", d=self.d_inner).contiguous()
        self.A_log = nn.Parameter(torch.log(A))
        self.A_log._no_weight_decay = True     # A不参与权重衰减

        # ---- D 参数(直接跳跃连接)：允许输入跳过SSM ----
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.D._no_weight_decay = True

        # 输出投影：512 → 256 恢复原始维度
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=bias)

        # 方向偏置：每个扫描方向有独立的B偏置
        # 形状 (5, 16)：4个方向 + 1个默认
        self.direction_Bs = nn.Parameter(torch.zeros(self.n_directions + 1, self.d_state))
        trunc_normal_(self.direction_Bs, std=0.02)

        # ===== 第2部分：小波模块（论文新增，可消融）=====
        if self.use_wavelet:
            self.dwt  = HaarDWT()     # 分解：把2D特征拆成4个子带
            self.idwt = HaarIDWT()    # 重构：把4个子带拼回2D特征

            # ===== 第3部分：高频子带增强 =====
            # 每个方向(LH水平/HL垂直/HH对角)各自一个轻量BottConv
            b = self.d_inner // 16    # 瓶颈通道 = 512/16 = 32
            self.enh_lh = nn.Sequential(    # 水平边缘增强
                nn.Conv2d(self.d_inner, b, 1),
                nn.Conv2d(b, b, 3, padding=1, groups=b),
                nn.Conv2d(b, self.d_inner, 1),
                nn.SiLU())
            self.enh_hl = nn.Sequential(    # 垂直边缘增强
                nn.Conv2d(self.d_inner, b, 1),
                nn.Conv2d(b, b, 3, padding=1, groups=b),
                nn.Conv2d(b, self.d_inner, 1),
                nn.SiLU())
            self.enh_hh = nn.Sequential(    # 对角边缘增强
                nn.Conv2d(self.d_inner, b, 1),
                nn.Conv2d(b, b, 3, padding=1, groups=b),
                nn.Conv2d(b, self.d_inner, 1),
                nn.SiLU())

            # ===== 第4部分：归一化层 =====
            self.norm_pre_dwt   = nn.LayerNorm(self.d_inner)       # DWT之前
            self.norm_ll        = nn.LayerNorm(self.d_inner)       # LL进SSM之前
            self.norm_lh        = nn.GroupNorm(16, self.d_inner)
            self.norm_hl        = nn.GroupNorm(16, self.d_inner)
            self.norm_hh        = nn.GroupNorm(16, self.d_inner)
            self.norm_post_idwt = nn.GroupNorm(16, self.d_inner)   # IDWT之后

            # ===== 融合权重 α：sigmoid(α) 压缩到 [0,1]，α=4→≈0.98 =====
            self.alpha = nn.Parameter(torch.tensor(4.0))

            # ===== LL适配器：把小波域分布"翻译"回SSM熟悉的分布 =====
            self.ll_adapter = nn.Sequential(
                nn.Linear(self.d_inner, self.d_inner),
                nn.SiLU(),
                nn.Linear(self.d_inner, self.d_inner),
            )

            # ===== 频域-空域对调消融的额外模块 =====
            # 'swap'/'all_conv' 时 LL 需要走 Conv 路径（结构与高频 enh 一致）
            if freq_mode in ('swap', 'all_conv'):
                self.enh_ll = nn.Sequential(
                    nn.Conv2d(self.d_inner, b, 1),
                    nn.Conv2d(b, b, 3, padding=1, groups=b),
                    nn.Conv2d(b, self.d_inner, 1),
                    nn.SiLU())
                self.norm_conv_ll = nn.GroupNorm(16, self.d_inner)
            # 'swap'/'all_ssm' 时高频子带需要走 SSM 路径（三个子带共享一套 norm+adapter）
            if freq_mode in ('swap', 'all_ssm'):
                self.norm_ssm_hi = nn.LayerNorm(self.d_inner)
                self.adapter_hi = nn.Sequential(
                    nn.Linear(self.d_inner, self.d_inner),
                    nn.SiLU(),
                    nn.Linear(self.d_inner, self.d_inner),
                )
        else:
            # 非小波路径：SSM 前只加一层 LayerNorm
            self.norm_ssm = nn.LayerNorm(self.d_inner)

        # 最终输出归一化（两种路径共用）
        self.norm_post = nn.LayerNorm(d_model)

        # ===== 扫描策略：TPAS（自适应）或 SASS（固定），可消融 =====
        if self.use_tpas:
            self.tpas = TPAS(n_directions=4, mode=tpas_mode,
                             tau_coh=tpas_tau_coh, tau_win=tpas_tau_win)
            # theta_soft: 连续方向条件化的可学习投影 (cos2θ,sin2θ,coh) → d_state
            if tpas_mode == 'theta_soft':
                self.orient_proj = nn.Linear(3, d_state)
                nn.init.normal_(self.orient_proj.weight, std=0.02)
                nn.init.zeros_(self.orient_proj.bias)

    def forward(self, x, hw_shape, orient=None):
        """
        ┌─────────────────────────────────────────────────────────────┐
        │ 输入:  x = (B, L=4096, C=256)   ← 4096个小方块               │
        │        hw_shape = (64, 64)       ← 每个小方块的空间位置       │
        │                                                              │
        │ 流程(7步):                                                    │
        │ ① in_proj: 256→1024, 拆成 x(512维)和 z(512维,门控)          │
        │ ② GBC Conv: 序列→2D→3×3瓶颈卷积→SiLU                        │
        │ ③ ★ DWT: 2D特征→4个子带(LL/LH/HL/HH), 各尺寸减半            │
        │ ④ ★ 分频处理:                                                │
        │     LL→flatten(1024)→SASS→selective_scan×4→reshape          │
        │     LH/HL/HH→各自Conv+SiLU                                   │
        │ ⑤ ★ IDWT: 4个子带→拼回2D特征                                 │
        │ ⑥ 混合: x_orig + α×(x_wavelet − x_orig)                     │
        │    α从0开始→epoch0=纯原始, 训练中α可能变大                    │
        │ ⑦ 门控: ×SiLU(z)→out_proj→输出                              │
        └─────────────────────────────────────────────────────────────┘
        """
        B, L, C = x.shape     # B=batch, L=序列长(4096), C=通道(256)
        H, W = hw_shape       # H=W=64
        E = self.d_inner      # E=512 (内部工作维度)

        # ===== 步骤①: in_proj 输入投影 =====
        # (B,4096,256) → (B,4096,1024) → 劈成两半
        xz = self.in_proj(x)                      # (B, L, 1024)
        x_feat, z = xz.chunk(2, dim=-1)           # x_feat=(B,4096,512), z=(B,4096,512)
        A = -torch.exp(self.A_log.float())        # A取反（负数=衰减）

        # ===== 步骤②: GBC卷积（感知裂缝形态）=====
        # 序列→2D图→卷积→激活→2D图
        x_2d = x_feat.reshape(B, H, W, E).permute(0, 3, 1, 2)   # (B,512,64,64)
        x_2d = self.act(self.conv2d(x_2d))                        # BottConv(512→32→512)

        if self.use_wavelet:
            # =================================================================
            # ★ 步骤③: DWT小波分解（论文核心创新）
            #   把特征分成4个子带: LL(低频轮廓) + LH/HL/HH(方向性高频边缘)
            # =================================================================
            x_2d_norm = x_2d.permute(0, 2, 3, 1)                      # (B,64,64,512)
            x_2d_norm = self.norm_pre_dwt(x_2d_norm)                   # LayerNorm稳定数值
            x_2d_norm = x_2d_norm.permute(0, 3, 1, 2)                 # (B,512,64,64)

            coeffs = self.dwt(x_2d_norm)                                # (B,2048,32,32)
            LL = coeffs[:, 0*E:1*E, :, :]    # 低频     (B,512,32,32)
            LH = coeffs[:, 1*E:2*E, :, :]    # 水平边缘  (B,512,32,32)
            HL = coeffs[:, 2*E:3*E, :, :]    # 垂直边缘  (B,512,32,32)
            HH = coeffs[:, 3*E:4*E, :, :]    # 对角边缘  (B,512,32,32)

            Hh, Wh = H // 2, W // 2            # Hh=Wh=32

            # --- ④: 子带处理（freq_mode 决定 SSM/Conv 在频域上的角色分配） ---
            if self.freq_mode == 'swap':
                # 对调：LL走Conv，高频走SSM（检验设计假设）
                LL_out = self.enh_ll(self.norm_conv_ll(LL)) + LL
                LH_out = self._ssm_subband(LH, A, orient=orient)
                HL_out = self._ssm_subband(HL, A, orient=orient)
                HH_out = self._ssm_subband(HH, A, orient=orient)
            elif self.freq_mode == 'all_ssm':
                # 全SSM：四个子带都走SSM
                LL_out = self._ssm_subband(LL, A, use_ll=True, orient=orient)
                LH_out = self._ssm_subband(LH, A, orient=orient)
                HL_out = self._ssm_subband(HL, A, orient=orient)
                HH_out = self._ssm_subband(HH, A, orient=orient)
            elif self.freq_mode == 'all_conv':
                # 全Conv：四个子带都走Conv
                LL_out = self.enh_ll(self.norm_conv_ll(LL)) + LL
                LH_out = self.enh_lh(self.norm_lh(LH)) + LH
                HL_out = self.enh_hl(self.norm_hl(HL)) + HL
                HH_out = self.enh_hh(self.norm_hh(HH)) + HH
            else:
                # default：LL走SSM，高频走Conv（原设计）
                LL_out = self._ssm_subband(LL, A, use_ll=True, orient=orient)
                LH_out = self.enh_lh(self.norm_lh(LH)) + LH
                HL_out = self.enh_hl(self.norm_hl(HL)) + HL
                HH_out = self.enh_hh(self.norm_hh(HH)) + HH

            # =================================================================
            # ★ 步骤⑤: IDWT重构
            # =================================================================
            coeffs_out = torch.cat([LL_out, LH_out, HL_out, HH_out], dim=1)  # (B,2048,32,32)
            x_merged = self.idwt(coeffs_out)                                  # (B,512,64,64)
            x_merged = self.norm_post_idwt(x_merged)                          # GroupNorm稳定

            # ===== 步骤⑥: α 加权的混合输出 =====
            alpha_clamped = torch.sigmoid(self.alpha)     # [0, 1]
            x_wavelet = x_merged.permute(0, 2, 3, 1).reshape(B, L, E)  # (B,4096,512)
            x_orig = x_2d.permute(0, 2, 3, 1).reshape(B, L, E)         # (B,4096,512)
            x_conv = x_orig + alpha_clamped * (x_wavelet - x_orig)      # 平滑插值
        else:
            # ===== 非小波路径（消融）：SSM 直接在全分辨率 64×64 上扫描 =====
            x_seq = x_2d.permute(0, 2, 3, 1).reshape(B, H*W, E)        # (B,4096,512)
            x_seq = self.norm_ssm(x_seq)
            x_conv = self._ssm_scan(x_seq, A, x_2d, orient=orient)      # (B,4096,512)

        # ===== 步骤⑦: 门控 + 输出 =====
        # z: 步骤①分出来的门控信号
        # SiLU(z): 门控值(裂缝→开, 背景→关)
        # x_conv × SiLU(z): 门控选择哪些信息通过
        y = x_conv * self.act(z)                                     # 门控 (B,4096,512)
        out = self.out_proj(y)                                       # 512→256 (B,4096,256)
        out = self.norm_post(out)                                    # 最终LayerNorm

        return out

    def _ssm_subband(self, sb, A, use_ll=False, orient=None):
        """
        把一个子带 (B, E, Hh, Wh) 走 SSM 路径。
        use_ll=True: 用 LL 的 norm_ll + ll_adapter；否则用高频共享的 norm_ssm_hi + adapter_hi。
        """
        B_, E_, Hh_, Wh_ = sb.shape
        seq = sb.permute(0, 2, 3, 1).reshape(B_, Hh_ * Wh_, E_)
        if use_ll:
            seq = self.norm_ll(seq)
            seq = self.ll_adapter(seq)
        else:
            seq = self.norm_ssm_hi(seq)
            seq = self.adapter_hi(seq)
        y = self._ssm_scan(seq, A, sb, orient=orient)
        return y.reshape(B_, Hh_, Wh_, E_).permute(0, 3, 1, 2)

    def _ssm_scan(self, seq, A, feat_2d, orient=None):
        """
        在序列 seq 上执行 4 方向选择性扫描（TPAS 自适应 或 SASS 固定）。

        输入: seq (B, L_seq, E), A (E, d_state), feat_2d (B, E, H2, W2) —— feat_2d 供 TPAS 算方向场
              orient (B, 3, H0, W0): 输入图像方向场 [cos2θ,sin2θ,coh] (theta_dir/theta_soft 用)
        输出: (B, L_seq, E)
        """
        B_, L_, E_ = seq.shape
        H2, W2 = feat_2d.shape[2], feat_2d.shape[3]

        # 从特征算出 dt(Δ)/B/C 三个SSM参数
        x_dbl = self.x_proj(seq)
        dt, B_ssm, C_ssm = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = self.dt_proj(dt)
        dt = dt.permute(0, 2, 1).contiguous()
        B_ssm = B_ssm.permute(0, 2, 1).contiguous()
        C_ssm = C_ssm.permute(0, 2, 1).contiguous()

        # 扫描顺序：TPAS（数据驱动自适应）或固定扫描策略（SASS/parallel/diag/parasna/diagsna）
        if self.use_tpas:
            orders, inverse_orders, directions = self.tpas(feat_2d, (H2, W2), orient=orient)
        else:
            orders, inverse_orders, directions = scan_orders(self.scan_strategy, H2, W2)

        # 方向偏置: 兼容三种形式
        #   ① list[int] 长度 L        → 全 batch 共享的静态码 (sass_dir / 固定策略)
        #   ② LongTensor (B, L)       → 逐样本数据驱动码 (theta_dir)
        #   ③ None                    → 连续方向条件化 (theta_soft, 用 orient_proj)
        direction_Bs = []
        for o, d in zip(orders, directions):
            if d is None:
                # theta_soft: (cos2θ,sin2θ,coh) 按扫描顺序取出 → 线性投影成逐样本 B 偏置
                if orient is None:
                    # 兜底: 没传输入图像方向场时从当前特征现算(逐样本, 会漂移, 仅应急)
                    theta, coh = self.tpas.compute_direction_field(feat_2d)
                    orient = torch.stack([torch.cos(2 * theta), torch.sin(2 * theta),
                                          coh], dim=1).detach()
                of = F.interpolate(orient, size=(H2, W2), mode='bilinear',
                                   align_corners=False)
                of = of.flatten(2).permute(0, 2, 1)                    # (B, L, 3)
                of = of[:, list(o), :]                                 # 扫描顺序
                dB = self.orient_proj(of).permute(0, 2, 1)             # (B, d_state, L)
                direction_Bs.append(dB.contiguous().to(B_ssm.dtype))
                continue
            if not torch.is_tensor(d):
                d = torch.as_tensor(d, dtype=torch.long,
                                    device=self.direction_Bs.device)
            else:
                d = d.to(self.direction_Bs.device)
            dB = self.direction_Bs[d]      # (L, d_state) 或 (B, L, d_state)
            if dB.dim() == 2:
                dB = dB[None, :, :].expand(B_, -1, -1)
            direction_Bs.append(dB.permute(0, 2, 1).contiguous().to(B_ssm.dtype))

        y_scan = []
        for o, inv_order, dB in zip(orders, inverse_orders, direction_Bs):
            if torch.is_tensor(o):
                # ★ 逐样本自适应路径(adaptive4): gather 重排 → scan → scatter 回原位
                o_ = o.to(seq.device)
                inv_ = inv_order.to(seq.device)
                x_s = seq.gather(1, o_.unsqueeze(-1).expand(-1, -1, E_))
                y = selective_scan_fn(
                    x_s.permute(0, 2, 1).contiguous(),
                    dt, A, (B_ssm + dB).contiguous(), C_ssm,
                    self.D.float(), z=None,
                    delta_bias=self.dt_proj.bias.float(),
                    delta_softplus=True, return_last_state=False,
                ).permute(0, 2, 1)
                y = y.gather(1, inv_.unsqueeze(-1).expand(-1, -1, E_))
            else:
                # 全 batch 共享的固定路径(SASS/sass_dir/theta_*)
                y = selective_scan_fn(
                    seq[:, o, :].permute(0, 2, 1).contiguous(),
                    dt, A, (B_ssm + dB).contiguous(), C_ssm,
                    self.D.float(), z=None,
                    delta_bias=self.dt_proj.bias.float(),
                    delta_softplus=True, return_last_state=False,
                ).permute(0, 2, 1)[:, inv_order, :]
            y_scan.append(y)
        return sum(y_scan)


# ============================================================================
# WAMambaBlock —— 一层的完整封装（GBC + SSM + PAF + 残差）
# ============================================================================
class convFFN(nn.Module):
    """
    convFFN = 卷积前馈网络（Convolutional Feed-Forward Network）

    对应 VMamba block 里的 FFN 子块。
    内部 = 同一个 GBC 调用两次（权重共享），做卷积形态建模。
    这样封装成独立模块，且计算与原 GBC×2 完全一致（保证效果不变）。
    """
    def __init__(self, channels):
        super().__init__()
        self.gbc = GBC(channels)

    def forward(self, x):
        x = self.gbc(x)
        x = self.gbc(x)
        return x


class WAMambaBlock(nn.Module):
    '''
    WAMambaBlock 的一个完整层 = convFFN + WAVSS + PAF融合 + 残差连接

    一层的内部流程:
      输入 x: (B, H×W=4096, C=256)

      ┌── 预处理 ──┐
      │ reshape → 2D → GBC_C ×2    ← GBC全局上下文建模（建模裂缝形态）
      │ flatten → 序列形式
      ├── SSM扫描 ──┤
      │ LayerNorm → WAVSS_2D       ← 4方向选择性扫描（建模长距离依赖）
      ├── 融合 ─────┤
      │ PAF(x, mixed_x)            ← 像素注意力融合（增强裂缝连续性）
      │ GN → Linear → 残差连接     ← 归一化 + 线性变换 + 残差
      └─────────────┘

      输出: mixed_x + residual (B, 4096, 256)
    '''

    def __init__(
            self,
            embed_dims,        # 256：嵌入维度
            use_rms_norm,      # 是否用RMSNorm（该配置为False，用LayerNorm）
            with_dwconv,        # 是否额外加深度可分离卷积（该配置为False）
            drop_path_rate,     # DropPath正则化概率
            mamba_cfg,          # SSM参数配置（传给 WAVSS_2D）
            use_wavelet=True,   # 是否使用小波模块（消融用）
            use_tpas=True,      # 是否使用TPAS扫描（消融用）
            scan_strategy='sass',  # 固定扫描策略（消融用）
            freq_mode='default',   # 频域-空域角色分配（消融用）
            tpas_mode='sass_dir',  # TPAS方向码模式 [sass_dir|...|adaptive4|greedy|hybrid]
            tpas_tau_coh=0.35,   # adaptive4 门控①: 候选最小相干性
            tpas_tau_win=12,     # adaptive4 门控②: 偏离锚路径最大窗口
    ):
        super(WAMambaBlock, self).__init__()

        # 把embed_dims传给mamba配置 → WAVSS_2D知道维度
        mamba_cfg.update({'d_model': embed_dims})

        # ===== 归一化层 =====
        if use_rms_norm:
            self.norm = RMSNorm(embed_dims)    # RMS归一化（mamba-ssm提供）
        else:
            self.norm = nn.LayerNorm(embed_dims) # 标准LayerNorm

        # ===== 可选深度可分离卷积（本配置不用）=====
        self.with_dwconv = with_dwconv
        if self.with_dwconv:
            self.dw = nn.Sequential(
                nn.Conv2d(embed_dims, embed_dims, kernel_size=(3, 3),
                          padding=(1, 1), bias=False, groups=embed_dims),
                nn.BatchNorm2d(embed_dims),
                nn.GELU(),
            )

        self.wavss = WAVSS(use_wavelet=use_wavelet, use_tpas=use_tpas, scan_strategy=scan_strategy, freq_mode=freq_mode, tpas_mode=tpas_mode, tpas_tau_coh=tpas_tau_coh, tpas_tau_win=tpas_tau_win, **mamba_cfg)  # ★ WD-SSB 小波域状态空间块

        # ===== DropPath：随机丢掉一些路径（正则化）=====
        self.drop_path = build_dropout(dict(type='DropPath', drop_prob=drop_path_rate))

        # ===== 输出变换：256维线性 + GN归一化 =====
        self.linear_256 = nn.Linear(in_features=256, out_features=256, bias=True)
        self.GN_256 = nn.GroupNorm(num_channels=256, num_groups=16)

        # ===== 全局上下文 + 像素注意力 =====
        self.conv_ffn = convFFN(embed_dims)                # → convFFN：卷积前馈网络（对应 FFN）
        self.PAF_256 = PAF(embed_dims, embed_dims // 2)    # → PAF.py：像素注意力融合

    def forward(self, x, hw_shape, orient=None):
        '''
        输入:  x (B, L=4096, C=256), hw_shape=(64,64), orient=输入图像方向场(可选)
        输出:  mixed_x (B, 4096, 256)
        '''
        B, L, C = x.shape
        H = W = int(math.sqrt(L))  # 64

        # ===== 步骤1: GBC全局上下文预处理 ×2 =====
        # 序列 → 2D → GBC → GBC → 序列
        # GBC帮助提取裂缝形态信息（宽度、边缘等）
        x = x.reshape(B, H, W, C).permute(0, 3, 1, 2)  # → (B, 256, 64, 64)
        x = self.conv_ffn(x)  # convFFN：卷积形态建模（内部 GBC×2，权重共享）
        x = x.permute(0, 2, 3, 1).reshape(B, H * W, C)  # → (B, 4096, 256)

        # ===== 步骤2: SSM选择性扫描（核心）=====
        # LayerNorm → WAVSS_2D（4方向SSM扫描）→ DropPath
        mixed_x = self.drop_path(
            self.wavss(self.norm(x), hw_shape, orient=orient)
        )  # (B, 4096, 256)

        # ===== 步骤3: PAF 像素注意力融合 =====
        # PAF让模型关注裂缝像素间的连续性，减少断裂
        b, l, c = mixed_x.shape
        h = w = int(math.sqrt(l))
        mixed_x = self.PAF_256(
            x.permute(0, 2, 1).reshape(b, c, h, w),         # 原始特征 (B,256,64,64)
            mixed_x.permute(0, 2, 1).reshape(b, c, h, w)     # SSM输出 (B,256,64,64)
        )  # PAF融合 → (B, 256, 64, 64)

        # GN归一化 → 序列
        mixed_x = self.GN_256(mixed_x).reshape(b, c, h * w).permute(0, 2, 1)

        # ===== 步骤4: 残差连接 + 线性变换 =====
        mixed_x_res = self.linear_256(
            self.GN_256(mixed_x.permute(0, 2, 1)).permute(0, 2, 1)
        )  # GN → Linear → (B, 4096, 256)

        return mixed_x + mixed_x_res  # ★ 残差连接：帮助梯度传播，稳定训练
