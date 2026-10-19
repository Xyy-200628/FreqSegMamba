"""
tpas.py -- Topology-Preserving Anisotropic Scanning (技术贡献三)

Replaces the fixed 4-direction SASS scanning with adaptive, data-driven scan paths
that follow the local crack orientation.

How it works on the LL subband (32x32 = 1024 pixels):
  1. Compute gradient field [Gx, Gy] via finite differences
  2. Build 2x2 structure tensor T = [Gx^2, GxGy; GxGy, Gy^2] at each pixel
  3. Gaussian smooth T for robustness against noise
  4. Eigenvalue decomposition -> principal direction theta = atan2(v1_y, v1_x)
  5. Generate N adaptive scan paths: greedily follow the direction field
  6. Each path is a sequence of pixel indices for selective_scan_fn

Why this helps:
  - SASS: fixed serpentine paths ignore crack geometry -> pixels across a crack
    get separated in the scan sequence -> SSM state decay severs the connection
  - TPAS: paths follow the crack manifold -> adjacent crack pixels stay adjacent
    in the scan sequence -> SSM maintains spatial continuity
"""

import hashlib

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# 高斯核缓存: image_orientation_field 每次调用都建核太浪费
_ORIENT_KERNEL_CACHE = {}


def image_orientation_field(img, sigma=2.0):
    """
    ★ 稳定 TPAS 信号源: 从原始输入图像(而非学习特征)计算方向场。

    为什么用它: greedy/方案B 失败的根因是 θ 从"每轮训练都在变"的特征里算,
    导致方向码漂移、模型学不到稳定偏置。原始输入图像在训练中恒定
    (数据集固定) → θ/coh 恒定 → 数据驱动且稳定, 这是超过 sass_dir 的关键。

    Input:  img (B, 3, H, W) 原始输入(归一化与否均可, 结构张量只看相对梯度)
    Output: orient (B, 3, H, W) = [cos2θ, sin2θ, coh], 已 detach。
            用 2θ 打包是因为裂缝朝向是"无向"的(θ 与 θ+π 等价),
            cos2θ/sin2θ 在 θ±π 处连续, 插值/量化都不会在 ±π 边界跳变。
    """
    with torch.no_grad():
        B, _, H, W = img.shape
        key = (sigma, img.device, img.dtype)
        if key not in _ORIENT_KERNEL_CACHE:
            ksize = int(4 * sigma + 1) | 1
            xs = torch.arange(ksize, dtype=img.dtype, device=img.device) - ksize // 2
            g = torch.exp(-0.5 * (xs / sigma) ** 2)
            g = g / g.sum()
            _ORIENT_KERNEL_CACHE[key] = (g[:, None] * g[None, :]).view(1, 1, ksize, ksize)
        gk = _ORIENT_KERNEL_CACHE[key]

        gray = img.mean(dim=1, keepdim=True)                       # (B,1,H,W)
        x_pad = F.pad(gray, (1, 1, 1, 1), mode='replicate')
        Gx = (x_pad[:, :, 1:H+1, 2:W+2] - x_pad[:, :, 1:H+1, 0:W]) / 2.0
        Gy = (x_pad[:, :, 2:H+2, 1:W+1] - x_pad[:, :, 0:H, 1:W+1]) / 2.0

        T_xx = F.conv2d(Gx * Gx, gk, padding='same')
        T_xy = F.conv2d(Gx * Gy, gk, padding='same')
        T_yy = F.conv2d(Gy * Gy, gk, padding='same')

        a, b, c = T_xx.squeeze(1), T_xy.squeeze(1), T_yy.squeeze(1)
        trace = a + c
        disc = torch.sqrt(torch.clamp((a - c) ** 2 + 4 * b * b, min=1e-12))
        lambda1 = (trace + disc) / 2.0

        axis = torch.abs(b) < 1e-8
        theta = torch.where(
            axis,
            torch.where(a > c, torch.full_like(a, torch.pi / 2), torch.zeros_like(a)),
            torch.atan2(b, a - lambda1))
        coh = torch.where(trace > 1e-8,
                          disc / torch.clamp(trace, min=1e-8),
                          torch.zeros_like(trace))

        orient = torch.stack([torch.cos(2 * theta), torch.sin(2 * theta), coh], dim=1)
    return orient.detach()


def quantize_orientation_codes(theta, coh, orders, W, coh_threshold=0.3):
    """
    ★ 逐样本"无向朝向"量化码 (theta_dir 模式用)。

    与旧 quantize_direction_codes 的关键区别:
      旧版按 θ 的符号分左/右、上/下 —— 但裂缝朝向是无向的,
      θ≈+π 和 θ≈0 是同一条水平裂缝, 旧版会给成码 2 和码 1(抖动)。
      新版按 2θ 分 4 个无向类: 水平→1, 主对角→2, 垂直→4, 反对角→3;
      θ 与 θ+π 必然同码, 不会在 ±π 边界抖动。
      低相干(背景) → 码 0(direction_Bs 的默认行), 方向条件化只作用在裂缝上。

    Input:  theta, coh (B, H, W) torch tensor
            orders: SASS 固定路径(tuple of lists)
    Output: list[LongTensor (B, L)], 每条路径一个, 逐样本逐位置方向码
    """
    B, H, Wt = theta.shape
    # 2θ 分 4 桶, 桶心 = 0/π/2/π/3π/2 (即 θ 的 水平/主对角/垂直/反对角),
    # 桶宽 π/2 → 需先偏移半桶(π/4)再取整, 最后 %4 处理回绕。
    # 否则桶边界会以 0 为中心, θ=-0.1(近似水平)会被错分进反对角桶。
    k = torch.floor(
        (torch.remainder(2 * theta, 2 * np.pi) + np.pi / 4) / (np.pi / 2)
    ).long() % 4                                                     # (B,H,W) ∈{0,1,2,3}
    code_table = torch.tensor([1, 2, 4, 3], dtype=torch.long, device=theta.device)
    codes = code_table[k].reshape(B, -1)                       # (B, L) ∈ {1,2,4,3}
    bg = (coh.reshape(B, -1) < coh_threshold)
    codes = torch.where(bg, torch.zeros_like(codes), codes)    # 背景 → 0
    idx = [list(o) for o in orders]
    return [codes[:, o].contiguous() for o in idx]             # 每条路径 (B, L)


# 自适应路径缓存: 路径只依赖(输入图方向场, 网格, 阈值) → 训练全程可安全缓存
# key = (blake2b(θ,coh), H, W, tau_coh, tau_win) → (orders, inverses, dirs) numpy
_ADAPTIVE_CACHE = {}
_ADAPTIVE_CACHE_MAX = 1024        # DeepCrack 训练集约 300 张, 足够放下


def generate_anchored_adaptive_path(theta, coh, anchor_order, H, W,
                                    tau_coh=0.35, tau_win=12,
                                    coh_weight=2.0, cont_weight=1.0):
    """
    ★ 锚定自适应路径("会拐弯的 SASS"): 以一条 SASS 蛇形路径为锚,
      局部沿裂缝方向场 θ 偏离, 阈值门控, 数学上保证全覆盖。

    每步决策(三重回退):
      1) 自适应步: 8 邻域中"未访问 且 在锚路径前方 tau_win 步以内
         且 coh ≥ tau_coh"的候选里, 选方向对齐分最高者(贴缝拐弯)。
         —— coh < tau_coh 的像素不跟(背景不引入路径, "低于阈值不算");
         —— 超出 tau_win 窗口的不跟(偏离锚太远, "大于阈值不算")。
      2) 刚性步: 无合格候选 → 回退到锚路径下一个未访问像素(正常推进)。
      3) 回退目标不相邻 → 记跳步(码 0, 与 scan_direction_codes 语义一致)。

    性质:
      全覆盖   —— 刚性回退沿锚路径单调推进, 每个像素恰好被访问一次;
      偏离有界 —— 自适应步只在锚路径 ±tau_win 窗口内重排, 不会丢失全局结构;
      稳定     —— 输入场恒定(来自原始图像) → 路径逐轮一致。

    Input:  theta, coh (H, W) numpy; anchor_order: 一条 SASS 锚路径(list, 长 L)
    Output: (order, inverse, dirs) 与 generate_greedy_path 同格式
    """
    L = H * W
    anchor_order = list(anchor_order)
    pos_in_anchor = np.argsort(np.asarray(anchor_order))   # 原索引 → 锚路径位置
    visited = np.zeros(L, dtype=bool)
    order = []
    inverse = np.full(L, -1, dtype=int)
    dirs = []

    # 8 邻域: 偏移 / 角度 / 方向码(与 greedy 同款, 已对齐)
    nb_di = [0, 1, 1, 1, 0, -1, -1, -1]
    nb_dj = [1, 1, 0, -1, -1, -1, 0, 1]
    nb_angle = np.array([0, np.pi/4, np.pi/2, 3*np.pi/4,
                         np.pi, -3*np.pi/4, -np.pi/2, -np.pi/4])
    nb_code = [1, 1, 4, 4, 2, 2, 3, 3]
    delta2k = {(0, 1): 0, (1, 1): 1, (1, 0): 2, (1, -1): 3,
               (0, -1): 4, (-1, -1): 5, (-1, 0): 6, (-1, 1): 7}

    coh_flat = coh.ravel()
    theta_flat = theta.ravel()

    i, j = anchor_order[0] // W, anchor_order[0] % W
    anchor_ptr = 0                       # 锚路径上最小的未访问位置(单调推进)
    for _ in range(L):
        idx = i * W + j
        order.append(idx)
        inverse[idx] = len(order) - 1
        visited[idx] = True
        a_pos = int(pos_in_anchor[idx])

        best_score, best_k = -1.0, -1
        for k in range(8):
            ni, nj = i + nb_di[k], j + nb_dj[k]
            if not (0 <= ni < H and 0 <= nj < W):
                continue
            nidx = ni * W + nj
            if visited[nidx]:
                continue
            # 门控①: 偏离锚窗口(只允许在锚路径前方 tau_win 步内重排)
            if not (0 < pos_in_anchor[nidx] - a_pos <= tau_win):
                continue
            # 门控②: 候选相干性过低(背景) → 不把路径引过去
            if coh_flat[nidx] < tau_coh:
                continue
            align = abs(np.cos(theta_flat[idx] - nb_angle[k]))
            cont = abs(np.cos(theta_flat[nidx] - nb_angle[k]))
            score = (align * (1.0 + coh_weight * coh_flat[nidx])
                     * (1.0 + cont_weight * cont))
            if score > best_score:
                best_score, best_k = score, k

        if best_k >= 0:
            dirs.append(nb_code[best_k])
            i, j = i + nb_di[best_k], j + nb_dj[best_k]
            continue

        # 刚性回退: 锚路径单调推进(全覆盖保证)
        while anchor_ptr < L and visited[anchor_order[anchor_ptr]]:
            anchor_ptr += 1
        if anchor_ptr >= L:
            break
        nidx = anchor_order[anchor_ptr]
        ni, nj = nidx // W, nidx % W
        k = delta2k.get((ni - i, nj - j))
        dirs.append(nb_code[k] if k is not None else 0)   # 不相邻 → 跳步码 0
        i, j = ni, nj

    while len(dirs) < L:
        dirs.append(dirs[-1] if dirs else 1)
    dirs = dirs[:L]
    dirs = [0] + dirs[:-1]              # shift, 与 SASS 约定一致
    return tuple(order[:L]), tuple(inverse.tolist()), tuple(dirs)


class TPAS(nn.Module):
    """Topology-Preserving Anisotropic Scanning (TPAS)

    Generates adaptive scan paths that follow local crack orientation.
    Computationally lightweight: structure tensor analysis on a 32x32 grid.
    """
    def __init__(self, n_directions=4, sigma=1.0, coh_weight=2.0, cont_weight=1.0,
                 mode='sass_dir', coh_gate=False, coh_threshold=0.3,
                 tau_coh=0.35, tau_win=12):
        super().__init__()
        self.n_directions = n_directions
        self.coh_weight = coh_weight    # 相干性加权(greedy/自适应路径打分用; 0=关闭)
        self.cont_weight = cont_weight  # 方向延续加权(greedy/自适应路径打分用)
        self.mode = 'sass_dir'          # ★ 公开版降级：仅保留基线 SASS 扫描（自适应/朝向码实现未随此版本发布）
                                        # 'theta_dir' =SASS固定路径 + 逐样本无向朝向码(稳定+数据驱动, ★推荐)
                                        # 'theta_soft'=SASS固定路径 + 连续方向条件化(cos2θ,sin2θ,coh→投影)
                                        # 'theta_path'=SASS 4条(保覆盖) + 输入图驱动的自适应贴缝4条(稳定自适应路径)
                                        # 'adaptive4' =4条SASS锚全部改造为自适应路径(逐样本, 阈值门控, ★真·自适应)
                                        # 'greedy'    =纯贪心贴缝路径(4条, 来自学习特征→会漂移)
                                        # 'hybrid'    =SASS固定路径 + 贪心TPAS路径(共8条, 会漂移)
        self.coh_gate = coh_gate        # (保留参数, 当前未使用)
        self.coh_threshold = coh_threshold  # theta_dir 的背景门控阈值
        self.tau_coh = tau_coh          # adaptive4 门控①: 候选最小相干性
        self.tau_win = tau_win          # adaptive4 门控②: 偏离锚路径的最大窗口

        # Gaussian kernel for structure tensor smoothing
        ksize = int(4 * sigma + 1) | 1  # ensure odd
        x = torch.arange(ksize, dtype=torch.float32) - ksize // 2
        g = torch.exp(-0.5 * (x / sigma) ** 2)
        g = g / g.sum()
        g2d = g[:, None] * g[None, :]
        self.register_buffer('gauss_kernel', g2d.view(1, 1, ksize, ksize))

    def compute_direction_field(self, x_2d):
        """
        Compute principal direction field from structure tensor analysis.

        Input:  x_2d (B, C, H, W) -- LL subband features
        Output: theta (B, H, W) -- crack direction at each pixel, in radians
                coh   (B, H, W) -- coherence (λ1-λ2)/(λ1+λ2) ∈ [0,1], 供贪心打分

        The direction theta points along the crack at each pixel.
        For background pixels (low gradient), theta is unreliable but gets
        overridden by the Gaussian smoothing.
        """
        B, C, H, W = x_2d.shape

        # Step 1: gradient using central finite differences
        # Pad to handle boundaries
        x_pad = F.pad(x_2d.mean(dim=1, keepdim=True), (1, 1, 1, 1), mode='replicate')
        Gx = (x_pad[:, :, 1:H+1, 2:W+2] - x_pad[:, :, 1:H+1, 0:W]) / 2.0   # (B, 1, H, W)
        Gy = (x_pad[:, :, 2:H+2, 1:W+1] - x_pad[:, :, 0:H, 1:W+1]) / 2.0

        # Step 2: structure tensor components
        T_xx = Gx * Gx                                                    # (B, 1, H, W)
        T_xy = Gx * Gy
        T_yy = Gy * Gy

        # Step 3: Gaussian smooth each component (suppresses noise)
        T_xx = F.conv2d(T_xx, self.gauss_kernel, padding='same')
        T_xy = F.conv2d(T_xy, self.gauss_kernel, padding='same')
        T_yy = F.conv2d(T_yy, self.gauss_kernel, padding='same')

        # Step 4: 稳健的 2×2 特征分解（★ 修复: 原公式在轴对齐边缘处退化,
        #         垂直裂缝方向被错误算成水平; 现加退化保护）
        # For matrix [[a, b], [b, c]]:
        #   λ1 = (trace + disc) / 2, disc = sqrt((a-c)^2 + 4b^2)
        #   次特征向量 v2 = [a-λ1, b] 即裂缝方向(沿边缘、垂直梯度)
        a = T_xx.squeeze(1)   # (B, H, W)
        b = T_xy.squeeze(1)
        c = T_yy.squeeze(1)

        trace = a + c
        discriminant = torch.sqrt(torch.clamp((a - c) ** 2 + 4 * b * b, min=1e-12))
        lambda1 = (trace + discriminant) / 2.0

        # b≈0 且 a>c 时 v2=[a-λ1, b] 退化为零向量 → 轴对齐处理(垂直边缘→竖直裂缝)
        axis = torch.abs(b) < 1e-8
        theta = torch.where(
            axis,
            torch.where(a > c, torch.full_like(a, torch.pi / 2), torch.zeros_like(a)),
            torch.atan2(b, a - lambda1))

        # 相干性 (λ1-λ2)/(λ1+λ2) = disc/trace ∈ [0,1]
        coh = torch.where(trace > 1e-8,
                          discriminant / torch.clamp(trace, min=1e-8),
                          torch.zeros_like(trace))

        return theta, coh

    def generate_greedy_path(self, theta, coh, start_pos, H, W):
        """
        Generate one scan path by greedily following the direction field.
        Always moves to the unvisited neighbor most aligned with local theta.

        ★ BUG FIX + 优化:
          1. 8 邻域偏移表与角度表对齐(原代码错位, 打分实际变成 |sin(θ-φ)|,
             路径会垂直裂缝移动)
          2. 打分加相干性加权: score = |cos(θ-φ)| × (1 + coh_weight·coh[邻居]),
             优先贴着裂缝(高相干)走
          3. ★ 死胡同改"就近跳": 在"高相干未访问"候选里选空间最近的一个,
             而旧版跳全局最相干点(常常跳到图像另一端) → 跳步距离大幅变小,
             序列连续性/拓扑保持性更好; 跳步码记为 0(无移动方向), 不再伪造方向

        Input:
          theta:    (H, W) direction field
          coh:      (H, W) coherence field
          start_pos: (i, j) starting pixel
        Output:
          order:    list of L pixel indices
          inverse:  list mapping original position -> position in order
          dirs:     list of direction codes (1=right, 2=left, 3=up, 4=down, 0=跳步)
        """
        L = H * W
        visited = np.zeros(L, dtype=bool)
        order = []
        inverse = np.full(L, -1, dtype=int)
        dirs = []
        # 死胡同候选阈值: 只跳到"相干性位于前 15%"的未访问像素(保持贴缝)
        _coh_thr = np.quantile(coh.ravel(), 0.85)

        # Direction angle for each of 8 neighbors
        # idx: 0=right(0°),1=down-right(45°),2=down(90°),3=down-left(135°),
        #      4=left(180°),5=up-left(225°),6=up(270°),7=up-right(315°)
        nb_di = [0, 1, 1, 1, 0, -1, -1, -1]   # row offsets(★ 已与角度表对齐)
        nb_dj = [1, 1, 0, -1, -1, -1, 0, 1]   # col offsets(★ 已与角度表对齐)
        nb_angle = np.array([0, np.pi/4, np.pi/2, 3*np.pi/4,
                             np.pi, -3*np.pi/4, -np.pi/2, -np.pi/4])
        # Direction codes for SASS compatibility
        nb_code = [1, 1, 4, 4, 2, 2, 3, 3]  # simplified mapping

        coh_flat = coh.ravel()
        theta_flat = theta.ravel()

        i, j = start_pos
        for _ in range(L):
            idx = i * W + j
            order.append(idx)
            inverse[idx] = len(order) - 1
            visited[idx] = True

            # Find best unvisited neighbor
            best_score = -999.0
            best_k = -1
            for k in range(8):
                ni, nj = i + nb_di[k], j + nb_dj[k]
                if 0 <= ni < H and 0 <= nj < W:
                    nidx = ni * W + nj
                    if not visited[nidx]:
                        # Alignment score: |cos(theta_i - phi_k)|
                        d_angle = theta[i, j] - nb_angle[k]
                        score = abs(np.cos(d_angle)) * (1.0 + self.coh_weight * coh_flat[nidx])
                        # 方向延续: 邻居处方向场是否延续当前移动方向
                        cont = abs(np.cos(theta_flat[nidx] - nb_angle[k]))
                        score *= (1.0 + self.cont_weight * cont)
                        if score > best_score:
                            best_score = score
                            best_k = k

            if best_k >= 0:
                dirs.append(nb_code[best_k])
                i, j = i + nb_di[best_k], j + nb_dj[best_k]
            else:
                # 死胡同: 在"高相干未访问"候选里选空间最近者(旧版跳全局最相干点
                # 会横穿整幅图, 破坏连续性; 就近跳的跳步距离小得多)
                unvisited = np.where(~visited)[0]
                if len(unvisited) > 0:
                    cand = unvisited[coh_flat[unvisited] >= _coh_thr]
                    if len(cand) == 0:
                        cand = unvisited
                    ci, cj = cand // W, cand % W
                    d2 = (ci - i) ** 2 + (cj - j) ** 2
                    nidx = int(cand[np.argmin(d2)])
                    i, j = nidx // W, nidx % W
                    dirs.append(0)          # 跳步 → 默认码(不再伪造方向)
                else:
                    break

        # Ensure dirs has exactly L elements (pad with last direction if needed)
        while len(dirs) < L:
            dirs.append(dirs[-1] if dirs else 1)
        dirs = dirs[:L]
        dirs = [0] + dirs[:-1]  # shift to match SASS convention
        return tuple(order[:L]), tuple(inverse.tolist()), tuple(dirs)

    def _resolve_field(self, x_2d, orient, H, W):
        """
        解析方向场来源, 返回逐样本 (theta, coh), 各 (B, H, W)。

        优先用外部 orient(来自原始输入图像, 训练中恒定 → 不漂移);
        orient 为 None 时退化为从当前特征 x_2d 计算(逐样本, 不再 batch 取均值)。
        """
        if orient is not None:
            o = F.interpolate(orient, size=(H, W), mode='bilinear', align_corners=False)
            cos2, sin2, coh = o[:, 0], o[:, 1], o[:, 2].clamp(0, 1)
            # 插值会缩短 (cos2θ, sin2θ) 向量长度, 重新归一化后再取角
            norm = torch.sqrt(cos2 * cos2 + sin2 * sin2).clamp(min=1e-6)
            theta = torch.atan2(sin2 / norm, cos2 / norm) / 2.0
            return theta, coh
        return self.compute_direction_field(x_2d)   # (B,H,W) 逐样本

    def forward(self, x_2d, hw_shape, orient=None):
        """
        sass_dir (默认, 修复版方案 B): SASS 固定路径 + 逐像素"扫描方向"码。

          扫描顺序 = SASS 的 4 条固定蛇形路径(方向全覆盖, 稳健);
          方向码   = 每个位置的"真实移动方向"(右/左/上/下, 跳步归 0),
                    由固定路径直接算出 → 语义正确(扫描方向)且稳定(不会随训练漂移)。
          比 SASS 恒定码更细: 告诉 SSM 每步往哪个方向扫, 而不是"整条路径一个码"。

        theta_dir (★ 用于超越 sass_dir): SASS 固定路径 + 逐样本"裂缝朝向"码。
          码来自原始输入图像的结构张量方向场(训练中恒定, 不漂移),
          逐样本逐位置、无向量化(2θ 分 4 类), 背景经 coh 门控归 0。
          与 sass_dir 的唯一差别 = 码从"图像无关静态模式"变成"数据驱动裂缝信息"。

        theta_soft: SASS 固定路径 + 连续方向条件化。
          directions 返回 None, 由 WAVSS 层用 orient (cos2θ,sin2θ,coh) 线性投影成
          逐样本 B 偏置, 比 5 槽离散码更细, 且完全绕开量化误差。

        theta_path (★ 真·自适应路径版): SASS 4 条 + 自适应贴缝 4 条(共 8 条)。
          自适应路径的方向场来自原始输入图像(恒定) → 路径每轮一致、不漂移;
          死胡同就近跳, 跳步距离小; SASS 4 条兜底保证全图覆盖。
          = "自适应路径"与"稳定/全覆盖"兼得。

        adaptive4 (★ 改造 SASS 本身): 4 条锚路径全部变成自适应("会拐弯的 SASS")。
          逐样本(每张图走自己的路), 阈值门控(coh < tau_coh 不跟 /
          偏离锚 > tau_win 不跟), 刚性回退保证全覆盖, 方向场来自输入图(恒定)。
          orders/inverses/directions 均为 LongTensor (B, L)。

        mode='greedy' 时只跑 4 条贪心贴缝路径(移动方向码);
        mode='hybrid' 时 = SASS 4 条 + 贪心 4 条一起求和(共 8 条)。

        Input:
          x_2d:     (B, C, H, W) LL subband features
          hw_shape: (H, W) spatial dimensions
          orient:   (B, 3, H0, W0) 输入图像方向场 [cos2θ,sin2θ,coh], 可为 None
        Output:
          orders:         n 个扫描索引序列
          inverse_orders: n 个逆映射
          directions:     n 个方向码序列 (n=8 for hybrid, 4 otherwise);
                          theta_dir 时为 LongTensor (B, L) 逐样本码,
                          theta_soft 时为 None
        """
        H = x_2d.shape[2]
        W = x_2d.shape[3]

        # ===== adaptive4: 4条SASS锚全部改造为逐样本自适应路径 =====
        if self.mode == 'adaptive4':
            theta, coh = self._resolve_field(x_2d, orient, H, W)      # (B,H,W) 逐样本
            th = theta.detach().cpu().numpy().astype(np.float32)
            co = coh.detach().cpu().numpy().astype(np.float32)
            B_ = th.shape[0]
            device = theta.device

            s_orders, _, _ = sass_scan((H, W))            # 4 条锚路径
            per_sample = []
            for b in range(B_):
                key = (hashlib.blake2b(
                    np.ascontiguousarray(np.stack([th[b], co[b]])).tobytes(),
                    digest_size=8).hexdigest(),
                    H, W, self.tau_coh, self.tau_win)
                hit = _ADAPTIVE_CACHE.get(key)
                if hit is None:
                    hit = tuple(
                        generate_anchored_adaptive_path(
                            th[b], co[b], anchor, H, W,
                            self.tau_coh, self.tau_win,
                            self.coh_weight, self.cont_weight)
                        for anchor in s_orders)
                    if len(_ADAPTIVE_CACHE) >= _ADAPTIVE_CACHE_MAX:
                        _ADAPTIVE_CACHE.pop(next(iter(_ADAPTIVE_CACHE)))
                    _ADAPTIVE_CACHE[key] = hit
                per_sample.append(hit)

            # 组装成逐样本张量: 4 条路径 × (B, L)
            orders = tuple(torch.as_tensor(
                np.stack([per_sample[b][p][0] for b in range(B_)]),
                dtype=torch.long, device=device) for p in range(4))
            inverses = tuple(torch.as_tensor(
                np.stack([per_sample[b][p][1] for b in range(B_)]),
                dtype=torch.long, device=device) for p in range(4))
            directions = tuple(torch.as_tensor(
                np.stack([per_sample[b][p][2] for b in range(B_)]),
                dtype=torch.long, device=device) for p in range(4))
            return orders, inverses, directions

        # ===== theta_path: 真·自适应路径, 但信号源恒定(不漂移) =====
        if self.mode == 'theta_path':
            # ① SASS 4 条固定路径: 保证全图/全方向覆盖(自适应路径的结构性短板)
            s_orders, s_inverses, s_dirs = sass_scan((H, W))
            # ② 4 条自适应贴缝路径: 方向场来自原始输入图像 → 训练中恒定,
            #    路径每轮完全一致(修复 greedy 的 θ 漂移根因), 死胡同改就近跳
            theta, coh = self._resolve_field(x_2d, orient, H, W)
            theta_np = theta.mean(0).detach().cpu().numpy()   # 与旧 greedy 一致: batch 共享
            coh_np = coh.mean(0).detach().cpu().numpy()
            g_orders, g_inverses, g_dirs = self._greedy_paths(theta_np, coh_np, H, W)
            return (s_orders + g_orders, s_inverses + g_inverses, s_dirs + g_dirs)

        # ===== theta_dir / theta_soft: 固定路径 + 数据驱动方向条件化 =====
        if self.mode == 'theta_dir':
            orders, inverse_orders, _ = sass_scan((H, W))
            theta, coh = self._resolve_field(x_2d, orient, H, W)
            directions = quantize_orientation_codes(
                theta, coh, orders, W, self.coh_threshold)
            return orders, inverse_orders, directions

        if self.mode == 'theta_soft':
            orders, inverse_orders, _ = sass_scan((H, W))
            # 连续条件化在 WAVSS._ssm_scan 里做(需要可学习投影), 这里只给路径
            return orders, inverse_orders, [None] * len(orders)

        # ===== 以下模式需要从特征算场 (batch 取均值, 比只用第 0 个样本更稳) =====
        theta, coh = self.compute_direction_field(x_2d)  # (B, H, W)

        theta_np = theta.mean(0).detach().cpu().numpy()
        coh_np = coh.mean(0).detach().cpu().numpy()

        if self.mode == 'hybrid':
            s_orders, s_inverses, s_dirs = sass_scan((H, W))        # 4 条 SASS
            g_orders, g_inverses, g_dirs = self._greedy_paths(theta_np, coh_np, H, W)  # 4 条贪心
            return (s_orders + g_orders,
                    s_inverses + g_inverses,
                    s_dirs + g_dirs)

        if self.mode == 'sass_dir':
            # 修复后的方案 B: SASS 固定路径 + 逐像素"扫描方向"码(稳定、语义正确)
            orders, inverse_orders, _ = sass_scan((H, W))
            directions = scan_direction_codes(orders, H, W)
            return orders, inverse_orders, directions

        # mode == 'greedy': 纯贪心贴缝
        return self._greedy_paths(theta_np, coh_np, H, W)

    def _greedy_paths(self, theta_np, coh_np, H, W):
        """生成 4 条相干引导的贪心贴缝路径(起点=高相干最远点采样)。"""
        L = H * W
        coh_flat = coh_np.ravel()
        pool = np.argsort(coh_flat)[::-1][:min(60, L)]
        seeds = [int(pool[0])]
        while len(seeds) < self.n_directions and len(seeds) < len(pool):
            best, best_d = -1, -1.0
            for c in pool:
                ci = int(c)
                d = min((ci // W - s // W) ** 2 + (ci % W - s % W) ** 2
                        for s in seeds)
                if d > best_d:
                    best_d, best = d, ci
            seeds.append(best)
        starts = [(s // W, s % W) for s in seeds]

        orders, inverses, dirs_list = [], [], []
        for start in starts[:self.n_directions]:
            o, inv, d = self.generate_greedy_path(theta_np, coh_np, start, H, W)
            orders.append(o)
            inverses.append(inv)
            dirs_list.append(d)

        return tuple(orders), tuple(inverses), tuple(dirs_list)


def quantize_direction_codes(theta, orders, H, W, coh=None, coh_threshold=0.3):
    """
    方案 B: 方向场 θ → 逐像素方向码(语义与贪心 nb_code 一致)。

      θ ∈ [-π, π] 量化到 4 个方向码:
        code 1 = 水平向右 (|θ| ≤ π/4)
        code 4 = 竖直向下 (π/4 < θ ≤ 3π/4)
        code 2 = 水平向左 (|θ| > 3π/4)
        code 3 = 竖直向上 (-3π/4 ≤ θ < -π/4)

      coh 为 None: 所有位置量化 θ。
      coh 给定:  低相干(背景)位置用码 0(方向偏置表的"默认"行), 裂缝位置用量化码
                 → 方向条件化只作用在裂缝上, 背景不参与。

    输出: 与 SASS/TPAS 的 directions 格式一致(4 个 list, 每个 L 个码)。
    """
    dirs_all = []
    for o in orders:
        dirs = []
        for p in o:
            i, j = int(p) // W, int(p) % W
            if coh is not None and coh[i, j] < coh_threshold:
                dirs.append(0)          # 背景 → 默认方向
                continue
            t = theta[i, j]
            if abs(t) <= np.pi / 4:
                dirs.append(1)          # 水平向右
            elif t > 3 * np.pi / 4 or t < -3 * np.pi / 4:
                dirs.append(2)          # 水平向左
            elif t > 0:
                dirs.append(4)          # 竖直向下
            else:
                dirs.append(3)          # 竖直向上
        dirs_all.append(dirs)
    return tuple(dirs_all)


def scan_direction_codes(orders, H, W):
    """
    修复方案 B: 逐像素"扫描移动方向"码(几何信号, 固定稳定)。

    位置 t 的码 = order[t-1] → order[t] 的移动方向, 语义与贪心 nb_code 一致:
      1=向右(含右下/右上)  2=向左  4=向下  3=向上; 跳步(不相邻)归 0(默认方向)。

    与 SASS 恒定码的区别: 每个位置给出真实空间移动方向, 比"整条路径一个码"更细;
    与方案 B(θ 量化)的区别: 码=扫描方向(正确语义)且由固定路径直接算出(稳定),
    不再把会漂移的裂缝朝向塞进 direction_Bs 槽位。
    """
    result = []
    for o in orders:
        dirs = [0]                      # 起点: 默认方向
        for k in range(1, len(o)):
            di = o[k] // W - o[k - 1] // W
            dj = o[k] % W - o[k - 1] % W
            if abs(di) > 1 or abs(dj) > 1:
                dirs.append(0)          # 跳步(不相邻)
            elif dj > 0:
                dirs.append(1)          # 向右(含对角)
            elif dj < 0:
                dirs.append(2)          # 向左(含对角)
            elif di > 0:
                dirs.append(4)          # 向下
            elif di < 0:
                dirs.append(3)          # 向上
            else:
                dirs.append(0)
        result.append(dirs)
    return tuple(result)


def sass_scan(hw_shape):
    """
    Fixed 4-direction serpentine scan (SASS) —— 消融实验的基线扫描策略。

    与 TPAS 相反，SASS 生成 4 条固定的蛇形扫描路径（水平 / 垂直 / 主对角 / 反对角），
    每条路径使用恒定的方向编码，不依赖数据。返回格式与 TPAS.forward 完全一致。

    输入: hw_shape: (H, W)
    输出: (orders, inverse_orders, directions)
    """
    H, W = hw_shape
    L = H * W
    idx = np.arange(L).reshape(H, W)

    def snake(grid):
        g = grid.copy()
        g[1::2] = g[1::2, ::-1]          # 奇数行反向（蛇形）
        return g.reshape(-1).tolist()

    def diag(grid):
        out = []
        for s in range(H + W - 1):
            for i in range(max(0, s - W + 1), min(H, s + 1)):
                out.append(int(grid[i, s - i]))
        return out

    orders = [
        snake(idx),          # 方向1: 水平蛇形
        snake(idx.T),        # 方向2: 垂直蛇形
        diag(idx),           # 方向3: 主对角
        diag(idx[:, ::-1]),  # 方向4: 反对角
    ]

    result_orders, result_inverses, result_dirs = [], [], []
    for k, o in enumerate(orders):
        inv = np.argsort(o).tolist()          # 逆映射：原索引 -> 扫描位置
        result_orders.append(tuple(o))
        result_inverses.append(tuple(inv))
        result_dirs.append([k + 1] * L)       # 每条路径恒定方向编码(1~4)

    return tuple(result_orders), tuple(result_inverses), tuple(result_dirs)


def scan_orders(strategy, H, W):
    """
    固定扫描策略消融 —— 生成 4 条扫描路径（与 TPAS 无关，仅当 use_tpas=False 时使用）。

    strategy 可选:
      - 'sass'    : 水平蛇形 + 垂直蛇形 + 主对角 + 反对角（默认，等于原 sass_scan）
      - 'parallel': 4 个平行方向（左→右 / 右→左 / 上→下 / 下→上）
      - 'diag'    : 4 个对角方向（主对角 + 反对角 + 各自反向）
      - 'parasna' : 平行蛇形（水平蛇 + 垂直蛇 + 各自反向）
      - 'diagsna' : 对角蛇形（沿对角线蛇形 + 反向）

    返回格式与 sass_scan / TPAS.forward 完全一致: (orders, inverse_orders, directions)
    """
    idx = np.arange(H * W).reshape(H, W)

    def snake(g):
        g = g.copy()
        g[1::2] = g[1::2, ::-1]          # 奇数行反向（蛇形）
        return g.reshape(-1).tolist()

    def diag(g):
        out = []
        for s in range(H + W - 1):
            for i in range(max(0, s - W + 1), min(H, s + 1)):
                out.append(int(g[i, s - i]))
        return out

    def diag_snake(g):
        out = []
        for s in range(H + W - 1):
            d = [int(g[i, s - i]) for i in range(max(0, s - W + 1), min(H, s + 1))]
            if s % 2 == 1:
                d = d[::-1]              # 奇数对角线反向（蛇形）
            out.extend(d)
        return out

    if strategy == 'parallel':
        orders = [idx.reshape(-1).tolist(), idx[:, ::-1].reshape(-1).tolist(),
                  idx.T.reshape(-1).tolist(), idx.T[::-1].reshape(-1).tolist()]
    elif strategy == 'diag':
        orders = [diag(idx), diag(idx[:, ::-1]), diag(idx)[::-1], diag(idx[:, ::-1])[::-1]]
    elif strategy == 'parasna':
        orders = [snake(idx), snake(idx)[::-1], snake(idx.T), snake(idx.T)[::-1]]
    elif strategy == 'diagsna':
        orders = [diag_snake(idx), diag_snake(idx)[::-1],
                  diag_snake(idx[:, ::-1]), diag_snake(idx[:, ::-1])[::-1]]
    elif strategy == 'sass':
        orders = [snake(idx), snake(idx.T), diag(idx), diag(idx[:, ::-1])]
    else:
        raise ValueError(f'unknown scan strategy: {strategy}')

    L = H * W
    result_orders, result_inverses, result_dirs = [], [], []
    for k, o in enumerate(orders):
        result_orders.append(tuple(o))
        result_inverses.append(tuple(np.argsort(o).tolist()))
        result_dirs.append([k + 1] * L)

    return tuple(result_orders), tuple(result_inverses), tuple(result_dirs)
