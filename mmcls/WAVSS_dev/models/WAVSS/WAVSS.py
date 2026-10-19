'''
Author: Hui Liu
Github: https://github.com/Karl1109
Email: liuhui@ieee.org

====================================================================================
WAVSS.py —— ② WAVSS 主干网络（编码器，论文核心）
====================================================================================

┌─────────────────────────────────────────────────────────────────────────────┐
│ 本文件在网络中的位置                                                          │
│                                                                              │
│   输入(RGB图像)                                                               │
│     │                                                                         │
│     ▼                                                                         │
│   ┌───────────────────┐                                                      │
│   │ ① ConvPatchEmbed  │ ← patch_embed.py：Conv层把图像切成图块                  │
│   │   + pos_embed     │ ← 本文件第98-111行：可学习位置编码                      │
│   └────────┬──────────┘                                                      │
│            ▼                                                                  │
│   ┌───────────────────┐                                                      │
│   │ ② WAVSS 主干 ★    │ ← ★ 本文件：4层 WAMambaBlock + 每层间输出不同尺度特征     │
│   │                   │                                                     │
│   │  Block 0 (第0层)  │ → 256通道 → BottConv→GN→Upsample→ F4(128, 64×64)    │
│   │  Block 1 (第1层)  │ → 256通道 → BottConv→GN→Upsample→ F3(64, 128×128)   │
│   │  Block 2 (第2层)  │ → 256通道 → BottConv→GN→Upsample→ F2(32, 256×256)   │
│   │  Block 3 (第3层)  │ → 256通道 → BottConv→GN→Upsample→ F1(16, 512×512)   │
│   │                   │                                                     │
│   │  每层内部 = SSM状态空间模型 + GBC卷积 + RMSNorm + 注意力                    │
│   └────────┬──────────┘                                                      │
│            ▼                                                                  │
│   输出 [F4, F3, F2, F1] → 送入 MFS 多尺度融合                                  │
└─────────────────────────────────────────────────────────────────────────────┘

关键架构参数 (arch='Crack'):
  - patch_size = 8          : 每8×8像素为一个patch（输入512→64个patch）
  - embed_dims = 256        : 每个patch嵌入到256维
  - num_layers = 4          : 4层WAVSS块（4个不同感受野层级）
  - d_state = 16            : SSM状态维度
  - expand = 2              : 通道扩展倍数
  - conv_size = 7           : GBC卷积核大小(7×7)

流程:
  1. Patch Embedding: (3, 512, 512) → conv切块 → (4096, 256), patch_resolution=(64,64)
  2. + Position Embedding: 可学习的位置编码
  3. 经过4层 WAMambaBlock（SSM + GBC + 归一化）
  4. 每层输出经过 BottConv(256→C) + GN + Upsample → 4个不同尺度的特征图
  5. 返回 [F4(128,64²), F3(64,128²), F2(32,256²), F1(16,512²)]
'''

from typing import Sequence
import copy
import numpy as np
import torch
import torch.nn as nn
from timm.models.layers import DropPath, trunc_normal_
DropPath.__repr__ = lambda self: f"timm.DropPath({self.drop_prob})"
from mmcv.cnn import build_norm_layer
from mmcv.cnn.utils.weight_init import trunc_normal_
from mmcv.runner.base_module import ModuleList
from mmcls.models.builder import BACKBONES
from mmcls.models.utils import resize_pos_embed, to_2tuple
from mmcls.models.backbones.base_backbone import BaseBackbone
from mmcls.WAVSS_dev.models.modules.patch_embed import ConvPatchEmbed  # → patch_embed.py：图块嵌入
from mmcls.WAVSS_dev.models.WAVSS.WAVSS_layer import WAMambaBlock      # → WAVSS_layer.py：每层核心(SSM算子)
from models.GBC import BottConv                                        # → GBC.py：瓶颈卷积
from models.tpas import image_orientation_field                        # ★ 输入图像稳定方向场(theta_dir/theta_soft用)


@BACKBONES.register_module()
class WAMamba(BaseBackbone):
    '''
    WAMamba = Wavelet-Aware Mamba（小波感知 Mamba）主干网络

    这是论文的核心贡献：用 Mamba SSM 替代传统的 Transformer 自注意力。
    相比 Vision Transformer：
      - SSM可建模长距离依赖（和Transformer一样好）
      - 但计算量小得多（O(N) vs O(N²)）
      - 配合GBC卷积感知裂缝形态结构
    '''

    # =========================================================================
    # 架构配置字典：arch='Crack' 对应的超参数
    # =========================================================================
    arch_zoo = {
        'Crack': {
            'patch_size': 8,              # 每个patch 8×8像素
            'embed_dims': 256,            # 嵌入维度（所有层统一用256）
            'num_layers': 4,              # 4层WAVSS块（对应4个尺度）
            'num_convs_patch_embed': 2,   # Patch Embed用2层卷积
            'layers_with_dwconv': [],     # 哪几层额外用深度可分离卷积（空=都不用）
            'layer_cfgs': {               # 每一层的配置（所有4层共用）
                'use_rms_norm': False,    # 不用RMSNorm（用LayerNorm）
                'mamba_cfg': {            # SSM核心算子配置
                    'd_state': 16,        # 状态空间维度
                    'expand': 2,          # 通道扩展倍数（256→512→扫描→256）
                    'conv_size': 7,       # GBC卷积核7×7
                    'dt_init': "random",  # 时间步长参数随机初始化
                    'conv_bias': True,    # 卷积带偏置
                    'bias': True,         # 线性层带偏置
                    'default_hw_shape': (512 // 8, 512 // 8)  # 默认特征图尺寸 (64, 64)
                }
            }
        }
    }

    def __init__(self,
                 # === 以下是构建 WAVSS 时传入的参数（由 decoder.py::build() 提供）===
                 img_size=224,               # 输入尺寸（会被Crack配置覆盖）
                 in_channels=3,              # RGB三通道
                 arch=None,                  # 'Crack' → 使用上面arch_zoo['Crack']的配置
                 patch_size=16,              # patch大小（会被Crack配置覆盖）
                 embed_dims=192,             # 嵌入维度（会被Crack配置覆盖=256）
                 num_layers=20,              # 层数（会被Crack配置覆盖=4）
                 num_convs_patch_embed=1,    # Patch Embed卷积数（会被覆盖=2）
                 with_pos_embed=True,        # 是否使用位置编码
                 out_indices=-1,             # 输出哪些层的特征（会被覆盖=[0,1,2,3]）
                 drop_rate=0.,               # Dropout概率
                 drop_path_rate=0.,          # DropPath概率（会被覆盖=0.2）
                 norm_cfg=dict(type='LN', eps=1e-6),  # 归一化配置：LayerNorm
                 final_norm=True,            # 最后一层后加归一化
                 interpolate_mode='bicubic', # 位置编码插值方式
                 layer_cfgs=dict(),          # 层配置（会被Crack配置覆盖）
                 layers_with_dwconv=[],      # 深度可分离卷积层索引
                 init_cfg=None,              # 初始化配置
                 test_cfg=dict(),
                 convert_syncbn=False,       # 是否转换SyncBN
                 freeze_patch_embed=False,   # 是否冻结Patch Embed
                 use_wavelet=True,           # 是否使用小波模块（消融用）
                 use_tpas=True,              # 是否使用TPAS扫描（消融用）
                 scan_strategy='sass',       # 固定扫描策略（消融用）
                 freq_mode='default',        # 频域-空域角色分配（消融用）
                 tpas_mode='sass_dir',       # TPAS方向码模式 [sass_dir|...|adaptive4|greedy|hybrid]
                 tpas_tau_coh=0.35,          # adaptive4 门控①: 候选最小相干性
                 tpas_tau_win=12,            # adaptive4 门控②: 偏离锚路径最大窗口
                 **kwargs):
        super(WAMamba, self).__init__(init_cfg)

        # === 存储基础参数 ===
        self.test_cfg = test_cfg
        self.img_size = to_2tuple(img_size)
        self.convert_syncbn = convert_syncbn
        self.arch = arch
        self.use_wavelet = use_wavelet
        self.use_tpas = use_tpas
        self.tpas_mode = tpas_mode

        # === 加载架构配置 ===
        # 如果指定了arch（如'Crack'），就用arch_zoo里的预设参数覆盖默认值
        if self.arch is None:
            self.embed_dims = embed_dims
            self.num_layers = num_layers
            self.patch_size = patch_size
            self.num_convs_patch_embed = num_convs_patch_embed
            self.layers_with_dwconv = layers_with_dwconv
            _layer_cfgs = layer_cfgs
        else:
            assert self.arch in self.arch_zoo.keys()
            self.embed_dims = self.arch_zoo[self.arch]['embed_dims']                       # 256
            self.num_layers = self.arch_zoo[self.arch]['num_layers']                       # 4
            self.patch_size = self.arch_zoo[self.arch]['patch_size']                       # 8
            self.num_convs_patch_embed = self.arch_zoo[self.arch]['num_convs_patch_embed'] # 2
            self.layers_with_dwconv = self.arch_zoo[self.arch]['layers_with_dwconv']       # []
            _layer_cfgs = self.arch_zoo[self.arch]['layer_cfgs']  # SSM参数

        self.with_pos_embed = with_pos_embed
        self.interpolate_mode = interpolate_mode
        self.freeze_patch_embed = freeze_patch_embed
        _drop_path_rate = drop_path_rate  # 0.2 (DropPath正则化概率)

        # ====================================================================
        # ① Patch Embedding —— 用卷积把图像切分成图块
        #    输入: (B, 3, 512, 512)
        #    输出: (B, 64×64, 256) = (B, 4096, 256)
        #    patch_resolution = (64, 64) —— 即64×64=4096个patch
        # ====================================================================
        self.patch_embed = ConvPatchEmbed(
            in_channels=in_channels,              # 3 (RGB)
            input_size=img_size,
            embed_dims=self.embed_dims,           # 256
            num_convs=self.num_convs_patch_embed, # 2层卷积
            patch_size=self.patch_size,           # 8
            stride=self.patch_size                # stride=8: 每隔8像素取一个patch
        )
        self.patch_resolution = self.patch_embed.init_out_size   # (64, 64)
        num_patches = self.patch_resolution[0] * self.patch_resolution[1]  # 4096

        # ====================================================================
        # ① 位置编码 —— 让模型知道每个patch在图像中的位置
        #    shape: (1, 4096, 256)，可学习参数
        # ====================================================================
        if with_pos_embed:
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, self.embed_dims))
            trunc_normal_(self.pos_embed, std=0.02)  # 截断正态分布初始化
        self.drop_after_pos = nn.Dropout(p=drop_rate)

        # ====================================================================
        # 解析 out_indices —— 哪些层输出特征图
        #   out_indices=[0,1,2,3] → 4层全部输出
        # ====================================================================
        if isinstance(out_indices, int):
            out_indices = [out_indices]
        for i, index in enumerate(out_indices):
            if index < 0:
                out_indices[i] = self.num_layers + index
        self.out_indices = out_indices  # [0, 1, 2, 3]

        # DropPath概率：4层间线性分布 [0, 0.2]
        dpr = np.linspace(0, _drop_path_rate, self.num_layers)
        self.drop_path_rate = _drop_path_rate

        # ====================================================================
        # ② 构建4层 WAMambaBlock —— 网络的核心计算单元
        #    每层内部 = SSM(选择性扫描) + GBC(瓶颈卷积) + LayerNorm
        #    4层的区别仅在于 DropPath 概率不同
        # ====================================================================
        self.layer_cfgs = _layer_cfgs
        self.layers = ModuleList()
        if isinstance(layer_cfgs, dict):
            # 同一个配置复制4份（每层独立一份配置）
            layer_cfgs = [copy.deepcopy(_layer_cfgs) for _ in range(self.num_layers)]

        for i in range(self.num_layers):
            _layer_cfg_i = layer_cfgs[i]
            _layer_cfg_i.update({
                "embed_dims": self.embed_dims,   # 256
                "drop_path_rate": dpr[i],        # 每层不同的DropPath概率
                "use_wavelet": use_wavelet,      # 小波开关
                "use_tpas": use_tpas,            # TPAS/SASS开关
                "scan_strategy": scan_strategy,  # 固定扫描策略
                "freq_mode": freq_mode,          # 频域-空域角色分配
                "tpas_mode": tpas_mode,          # TPAS方向码模式
                "tpas_tau_coh": tpas_tau_coh,    # adaptive4 门控①
                "tpas_tau_win": tpas_tau_win,    # adaptive4 门控②
            })
            if i in self.layers_with_dwconv:
                _layer_cfg_i.update({"with_dwconv": True})
            else:
                _layer_cfg_i.update({"with_dwconv": False})
            # 创建一层 WAMambaBlock → WAVSS_layer.py
            self.layers.append(
                WAMambaBlock(**_layer_cfg_i)
            )

        # ====================================================================
        # 最终归一化层（第3层输出后加一层LayerNorm）
        # ====================================================================
        self.final_norm = final_norm
        if final_norm:
            self.norm1_name, norm1 = build_norm_layer(
                norm_cfg, self.embed_dims, postfix=1)
            self.add_module(self.norm1_name, norm1)

        # 为每个输出层创建归一化层（非最后一层的输出需要额外归一化）
        for i in out_indices:
            if i != self.num_layers - 1:
                if norm_cfg is not None:
                    norm_layer = build_norm_layer(norm_cfg, self.embed_dims)[1]
                else:
                    norm_layer = nn.Identity()
                self.add_module(f'norm_layer{i}', norm_layer)

        # ====================================================================
        # ② 输出转换层 —— 把每层的256维特征 → 不同通道数 → 不同分辨率
        #
        #    第0层 → BottConv(256→128) + GN + Upsample(64×64)  → F4: (128, 64, 64)
        #    第1层 → BottConv(256→64)  + GN + Upsample(128×128) → F3: (64, 128, 128)
        #    第2层 → BottConv(256→32)  + GN + Upsample(256×256) → F2: (32, 256, 256)
        #    第3层 → BottConv(256→16)  + GN + Upsample(512×512) → F1: (16, 512, 512)
        #
        #    通道数逐渐减半，分辨率逐渐加倍 —— 形成特征金字塔
        # ====================================================================
        self.conv256to128 = BottConv(256, 128, 32, kernel_size=1, stride=1, padding=0)
        self.conv256to64  = BottConv(256, 64,  16, kernel_size=1, stride=1, padding=0)
        self.conv256to32  = BottConv(256, 32,   8, kernel_size=1, stride=1, padding=0)
        self.conv256to16  = BottConv(256, 16,   4, kernel_size=1, stride=1, padding=0)

        # GroupNorm 分组归一化（对每组通道分别归一化）
        self.gn128 = nn.GroupNorm(num_channels=128, num_groups=8)
        self.gn64  = nn.GroupNorm(num_channels=64,  num_groups=4)
        self.gn32  = nn.GroupNorm(num_channels=32,  num_groups=2)
        self.gn16  = nn.GroupNorm(num_channels=16,  num_groups=2)

    @property
    def norm1(self):
        return getattr(self, self.norm1_name)

    def init_weights(self):
        super(WAMamba, self).init_weights()
        if not (isinstance(self.init_cfg, dict)
                and self.init_cfg['type'] == 'Pretrained'):
            if self.with_pos_embed:
                trunc_normal_(self.pos_embed, std=0.02)

    def set_freeze_patch_embed(self):
        if self.freeze_patch_embed:
            self.patch_embed.eval()
            for param in self.patch_embed.parameters():
                param.requires_grad = False

    # =========================================================================
    # 前向传播 —— 整个网络的数据流
    # =========================================================================
    def forward(self, x):
        '''
        输入:  x, shape = (B, 3, 512, 512)  RGB裂缝图像
        输出:  [F4, F3, F2, F1]
               F4: (B, 128, 64,  64 )  ← 深层语义（大感受野）
               F3: (B, 64,  128, 128)
               F2: (B, 32,  256, 256)
               F1: (B, 16,  512, 512)  ← 浅层细节（小感受野）
        '''
        # =================================================================
        # ★ 步骤0: 稳定方向场（theta_dir / theta_soft 模式）
        #   从原始输入图像(训练中恒定)算 θ/coh —— 不从学习特征算,
        #   避免 greedy/方案B 的 θ 漂移问题。打包为 [cos2θ, sin2θ, coh],
        #   各 WAVSS 层按需插值到自己的扫描分辨率。
        # =================================================================
        orient = None
        if self.use_tpas and self.tpas_mode in ('theta_dir', 'theta_soft',
                                                'theta_path', 'adaptive4'):
            orient = image_orientation_field(x)          # (B, 3, 512, 512), detached

        # =================================================================
        # 步骤1: Patch Embedding — 卷积切块
        #   (B, 3, 512, 512) → patch_embed → (B, 4096, 256)
        #   patch_resolution = (64, 64)
        # =================================================================
        x, patch_resolution = self.patch_embed(x)

        # =================================================================
        # 步骤2: 加位置编码 — 让模型知道每个patch在哪儿
        #   x = x + pos_embed （形状都是 (B, 4096, 256)）
        #   resize_pos_embed: 保证位置编码尺寸匹配实际patch数量
        # =================================================================
        if self.with_pos_embed:
            pos_embed = resize_pos_embed(
                self.pos_embed,
                self.patch_resolution,
                patch_resolution,
                mode=self.interpolate_mode,
                num_extra_tokens=0
            )
            x = x + pos_embed                # 逐元素相加
        x = self.drop_after_pos(x)           # Dropout正则化

        # =================================================================
        # 步骤3: 逐层通过 WAMambaBlock（SSM+GBC+归一化）× 4层
        #
        #   每层输入/输出都是 (B, 4096, 256)（序列形式）
        #   但不同层学到不同抽象层次的特征：
        #     第0层 → 低级纹理/边缘
        #     第1层 → 中级形状
        #     第2层 → 高级结构
        #     第3层 → 语义信息
        #
        #   ※ 注意：这4层不改变特征图和尺寸（256维, 64×64分辨率）
        #     分辨率改变发生在下面的Upsample阶段
        # =================================================================
        outs_before = []  # 归一化后的原始输出（备用）
        outs = []         # 最终输出的多尺度特征图

        for i, layer in enumerate(self.layers):
            # 经过一层 WAMambaBlock（SSM扫描 + GBC卷积）
            x = layer(x, hw_shape=patch_resolution, orient=orient)

            # 最后一层后加最终归一化
            if i == len(self.layers) - 1 and self.final_norm:
                x = self.norm1(x)

            # =============================================================
            # 步骤4: 如果当前层在out_indices中 → 转换并输出一个尺度的特征图
            #
            #   每层都做：256维 → BottConv降维 → GN归一化 → Upsample上采样
            #
            #   第0层 (i=0): (B,4096,256) → reshape(64,64) → BottConv(256→128)
            #               → GN → Upsample(64,64) = F4 (128ch, 64²)
            #
            #   第1层 (i=1): 同上 → BottConv(256→64) → GN → Upsample(128,128)
            #               = F3 (64ch, 128²)
            #
            #   第2层 (i=2): 同上 → BottConv(256→32) → GN → Upsample(256,256)
            #               = F2 (32ch, 256²)
            #
            #   第3层 (i=3): 同上 → BottConv(256→16) → GN → Upsample(512,512)
            #               = F1 (16ch, 512²)
            #
            #   效果：越深层 → 通道越少、分辨率越高
            #         深层语义 + 浅层细节 → 送入MFS融合
            # =============================================================
            if i in self.out_indices:
                B, _, C = x.shape
                # 序列 → 2D特征图: (B, 4096, 256) → (B, 64, 64, 256)
                patch_token = x.reshape(B, *patch_resolution, C)

                # 非最后一层的输出先过归一化
                if i != self.num_layers - 1:
                    norm_layer = getattr(self, f'norm_layer{i}')
                    patch_token = norm_layer(patch_token)

                # 通道维提前: (B, 64, 64, 256) → (B, 256, 64, 64)
                patch_token = patch_token.permute(0, 3, 1, 2)
                outs_before.append(patch_token)

                # 根据层索引选择对应的转换通道和上采样尺寸
                if i == self.out_indices[0]:
                    # 第0层 → F4: 256→128通道，分辨率64×64
                    patch_token_mid = self.gn128(self.conv256to128(patch_token))
                    patch_token_mid = nn.Upsample(size=(64, 64), mode="bilinear")(patch_token_mid)
                    outs.append(patch_token_mid)

                elif i == self.out_indices[1]:
                    # 第1层 → F3: 256→64通道，分辨率128×128
                    patch_token_mid = self.gn64(self.conv256to64(patch_token))
                    patch_token_mid = nn.Upsample(size=(128, 128), mode="bilinear")(patch_token_mid)
                    outs.append(patch_token_mid)

                elif i == self.out_indices[2]:
                    # 第2层 → F2: 256→32通道，分辨率256×256
                    patch_token_mid = self.gn32(self.conv256to32(patch_token))
                    patch_token_mid = nn.Upsample(size=(256, 256), mode="bilinear")(patch_token_mid)
                    outs.append(patch_token_mid)

                elif i == self.out_indices[3]:
                    # 第3层 → F1: 256→16通道，分辨率512×512
                    patch_token_mid = self.gn16(self.conv256to16(patch_token))
                    patch_token_mid = nn.Upsample(size=(512, 512), mode="bilinear")(patch_token_mid)
                    outs.append(patch_token_mid)

        # 返回4个尺度的特征图: [F4, F3, F2, F1]
        return outs
