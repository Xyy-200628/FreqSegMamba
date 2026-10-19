"""
decoder.py -- Decoder + Loss + Model Assembly
FP-ADB version: ADP-Stream + CFAF fusion + BoundaryHead + edge loss
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcls.WAVSS_dev.models.WAVSS.WAVSS import WAMamba
from models.MFS import MFS
from models.edge_utils import ADPStream, CFAF


# ==============================================================================
# Decoder -- Backbone + ADP-Stream + CFAF + MFS + BoundaryHead
# ==============================================================================
class Decoder(nn.Module):
    def __init__(self, backbone, args=None):
        super().__init__()
        self.args = args
        self.backbone = backbone
        self.use_adp = False  # ★ 公开版降级：边缘先验分支(Freq-ADB)未随此版本发布

        if self.use_adp:
            # ADP-Stream: 并行的边缘先验分支
            self.adp_stream = ADPStream(in_ch=12, pyramid_in_ch=12)

            # CFAF: 4个尺度的交叉融合模块
            self.cfaf1 = CFAF(128)  # F4
            self.cfaf2 = CFAF(64)   # F3
            self.cfaf3 = CFAF(32)   # F2
            self.cfaf4 = CFAF(16)   # F1

        # MFS: still used
        self.MFS = MFS(8)

        if self.use_adp:
            # Boundary Head: 1x1 conv on MFS intermediate features
            self.boundary_head = nn.Conv2d(8, 1, 1)

    def forward(self, samples, return_boundary=False):
        # samples: (B, 3, 512, 512)  RGB裂缝图片

        # === 步骤1: 编码器主干 → 4个尺度的语义特征 ===
        outs_WAVSS = self.backbone(samples)
        # 返回 [F4, F3, F2, F1]（由深层到浅层）
        #   F4: (B, 128, 64,  64)   ← 深层：全局结构，低分辨率
        #   F3: (B, 64,  128, 128)  ← 中高层
        #   F2: (B, 32,  256, 256)  ← 中低层
        #   F1: (B, 16,  512, 512)  ← 浅层：细节纹理，高分辨率

        if self.use_adp:
            # === 步骤2: ADP-Stream → 4个尺度的边缘细节 ===
            edge_feats = self.adp_stream(samples)

            # === 步骤3: CFAF融合 —— 每个尺度把语义+边缘合成一张图 ===
            fused = [
                self.cfaf1(outs_WAVSS[0], edge_feats[0]),  # F4+D4
                self.cfaf2(outs_WAVSS[1], edge_feats[1]),  # F3+D3
                self.cfaf3(outs_WAVSS[2], edge_feats[2]),  # F2+D2
                self.cfaf4(outs_WAVSS[3], edge_feats[3]),  # F1+D1
            ]
        else:
            # 消融：关闭边缘先验分支，主干特征直接进 MFS
            fused = outs_WAVSS

        # === 步骤4: MFS多尺度融合 → 单通道预测图 ===
        mask, feat = self.MFS(fused, return_feat=True)
        # mask: (B, 1, 512, 512)  ← 裂缝预测图(logits)
        # feat: (B, 8, 512, 512)  ← 中间特征，给BoundaryHead用

        # === 步骤5: 边界头（仅训练时，且启用边缘先验分支）===
        if return_boundary and self.use_adp:
            boundary = self.boundary_head(feat)  # (B, 1, 512, 512) 边缘预测
            return mask, boundary
        return mask


# ==============================================================================
# DiceLoss
# ==============================================================================
class DiceLoss(nn.Module):
    def __init__(self, smooth=1., dims=(-2, -1)):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.dims = dims

    def forward(self, x, y):
        tp = (x * y).sum(self.dims)
        fp = (x * (1 - y)).sum(self.dims)
        fn = ((1 - x) * y).sum(self.dims)
        dc = (2 * tp + self.smooth) / (2 * tp + fp + fn + self.smooth)
        dc = dc.mean()
        return 1 - dc


# ==============================================================================
# SobelEdgeLoss -- compute edge from GT and pred, then BCE
# ==============================================================================
class SobelEdgeLoss(nn.Module):
    """用Sobel算子从GT和预测中提取边缘，然后算BCE"""
    def __init__(self):
        super().__init__()
        # 3×3 Sobel核（固定的，不参与训练）
        kx = torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=torch.float32) / 4.
        ky = kx.t()
        self.register_buffer('kx', kx.view(1,1,3,3))
        self.register_buffer('ky', ky.view(1,1,3,3))
        self.bce = nn.BCEWithLogitsLoss()

    def sobel(self, x):
        """x: (B, 1, H, W), returns (B, 1, H, W) edge magnitude"""
        gx = F.conv2d(x, self.kx, padding=1)
        gy = F.conv2d(x, self.ky, padding=1)
        return torch.sqrt(gx**2 + gy**2 + 1e-6)

    def forward(self, pred, target):
        edge_pred = self.sobel(pred)      # 预测图的边缘
        edge_gt = self.sobel(target)       # 标签的边缘
        return self.bce(edge_pred, edge_gt)

# ==============================================================================
# bce_dice_edge -- BCE + Dice + Edge Loss
# ==============================================================================
class bce_dice_edge(nn.Module):
    """
    组合损失 = 0.83×BCE + 0.17×Dice + edge_weight×EdgeLoss
    edge_weight 默认 0.05，提供弱监督边缘引导
    """
    def __init__(self, args):
        super(bce_dice_edge, self).__init__()
        self.bce_fn = nn.BCEWithLogitsLoss()
        self.dice_fn = DiceLoss()
        self.edge_fn = SobelEdgeLoss()
        self.args = args
        self.edge_weight = getattr(args, 'edge_weight', 0.05)

    def forward(self, y_pred, y_true, boundary_pred=None):
        bce = self.bce_fn(y_pred, y_true)
        dice = self.dice_fn(y_pred.sigmoid(), y_true)
        loss = self.args.BCELoss_ratio * bce + self.args.DiceLoss_ratio * dice

        if boundary_pred is not None:
            edge_loss = self.edge_fn(boundary_pred.sigmoid(), y_true)
            loss = loss + self.edge_weight * edge_loss

        return loss

# ==============================================================================
# build() -- Model Assembly
# ==============================================================================
def build(args):
    device = torch.device(args.device)
    args.device = torch.device(args.device)

    use_wavelet = not getattr(args, 'no_wavelet', False)
    use_tpas = not getattr(args, 'no_tpas', False)
    scan_strategy = getattr(args, 'scan_strategy', 'sass')
    freq_mode = getattr(args, 'freq_mode', 'default')
    tpas_mode = getattr(args, 'tpas_mode', 'sass_dir')
    tpas_tau_coh = getattr(args, 'tpas_tau_coh', 0.35)
    tpas_tau_win = getattr(args, 'tpas_tau_win', 12)

    backbone = WAMamba(arch='Crack',
                     out_indices=(0, 1, 2, 3),
                     drop_path_rate=0.2,
                     final_norm=True,
                     convert_syncbn=True,
                     use_wavelet=use_wavelet,
                     use_tpas=use_tpas,
                     scan_strategy=scan_strategy,
                     freq_mode=freq_mode,
                     tpas_mode=tpas_mode,
                     tpas_tau_coh=tpas_tau_coh,
                     tpas_tau_win=tpas_tau_win)

    model = Decoder(backbone, args)
    criterion = bce_dice_edge(args)
    criterion.to(device)

    return model, criterion
