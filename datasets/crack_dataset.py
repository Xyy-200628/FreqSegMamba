"""
这个文件负责：从硬盘读取裂缝图片和标签，处理成Tensor格式，喂给模型训练。

用到的文件结构（你必须按这个格式准备数据）：
  DeepCrack/
    ├── train_img/    ← 裂缝照片（.jpg或.png都行）
    ├── train_lab/    ← 人工标注的黑白标签图（必须是.png）
    ├── test_img/     ← 测试照片
    └── test_lab/     ← 测试标签

规则：
  - train_img/001.jpg 对应的标签必须是 train_lab/001.png
  - 文件名必须一样，标签扩展名固定为 .png
"""

import os.path
import cv2
from PIL import Image
from .base_dataset import BaseDataset
import torchvision.transforms as transforms
from .image_folder import make_dataset
from .utils import MaskToTensor


class CrackDataset(BaseDataset):

    def __init__(self, args):
        """创建数据集对象时执行一次，做三件事：收集文件路径、准备图像预处理、准备标签预处理"""

        BaseDataset.__init__(self, args)

        # ===== 第1件事：把 train_img 文件夹里所有图片的路径记下来 =====
        # 比如 phase='train' → 去 {DeepCrack}/train_img/ 里找所有 .jpg .png
        # 返回一个列表：['.../train_img/001.jpg', '.../train_img/002.jpg', ...]
        self.img_paths = make_dataset(
            os.path.join(args.dataset_path, '{}_img'.format(args.phase))
        )

        # ===== 第2件事：记下标签文件夹的路径 =====
        # 比如 phase='train' → {DeepCrack}/train_lab/
        self.lab_dir = os.path.join(args.dataset_path, '{}_lab'.format(args.phase))

        # ===== 第3件事：定义图片的"预处理流水线" =====
        # 两张图（原图和标签）都要做 transform，但处理方式不同:
        #
        # 【图像 transform】
        #  ToTensor(): 把图片从 numpy(H,W,C) 变成 Tensor(C,H,W)，
        #              同时把像素值从 0~255 缩放到 0~1
        #  Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)):
        #              把 [0,1] 变成 [-1,1]
        #              公式：(x - 0.5) / 0.5
        #              比如原来 0 → 变成 -1，原来 1 → 变成 1
        #              这样三个通道都围绕 0 对称，训练更稳定
        self.img_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        # 【标签 transform】
        #  标签是黑白图，里面的值只有 0（背景）和 1（裂缝）
        #  MaskToTensor 做的事：把 numpy 数组 → Long 类型 Tensor
        #  不需要 Normalize，因为标签就是 0 和 1，归一化反而出错
        self.lab_transform = MaskToTensor()

        self.phase = args.phase

    def __getitem__(self, index):
        """训练时每次取一张图，index 是 PyTorch 随机生成的一个数字"""

        # ===== 第1步：拼出图片路径和标签路径 =====
        # 比如 index=3 → img_paths[3] = '.../train_img/abc.jpg'
        # 标签路径 = lab_dir/abc.png （去掉扩展名，换上.png）
        img_path = self.img_paths[index]
        lab_path = os.path.join(
            self.lab_dir,
            os.path.basename(img_path).split('.')[0] + '.png'
        )

        # ===== 第2步：用 OpenCV 读取文件 =====
        # cv2.imread 读出来是 numpy 数组（就是一个数字矩阵）
        #   - 图片：三维数组 (高, 宽, 3)，每个格子存 BGR 三个色
        #   - 标签：二维数组 (高, 宽)，每个格子存一个数
        # OpenCV 读出来默认是 BGR 顺序，我们用 cvtColor 转成 RGB
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        lab = cv2.imread(lab_path, cv2.IMREAD_UNCHANGED)

        # 如果标签不小心存成了彩色图(3通道)，转成灰度图(1通道)
        if len(lab.shape) == 3:
            lab = cv2.cvtColor(lab, cv2.COLOR_BGR2GRAY)

        # ===== 第3步：把所有的图都缩放成一样大（512x512）=====
        # 原因：神经网络要求所有输入尺寸一致
        # cv2.resize 会自动放大或缩小
        # INTER_CUBIC 是一种平滑的缩放算法，比普通的邻居插值好
        w, h = self.args.load_width, self.args.load_height
        if w > 0 or h > 0:
            img = cv2.resize(img, (w, h), interpolation=cv2.INTER_CUBIC)
            lab = cv2.resize(lab, (w, h), interpolation=cv2.INTER_CUBIC)

        # ===== 第4步：把标签变成纯 0 和 1 =====
        # cv2.threshold(图, 阈值, 最大值, THRESH_BINARY):
        #   像素 ≥ 127 → 变成 255
        #   像素 < 127 → 变成 0
        # 做两次：
        #   第一次：0~255 的灰度 → 0 和 255（纯黑白）
        #   第二次：0 和 255     → 0 和 1（二值）
        _, lab = cv2.threshold(lab, 127, 255, cv2.THRESH_BINARY)
        _, lab = cv2.threshold(lab, 127, 1, cv2.THRESH_BINARY)

        # ===== 第5步：把 numpy 变成 PyTorch 的 Tensor =====
        # 图片：numpy → PIL Image → ToTensor → Normalize
        #   最终是 Tensor(3, 512, 512)，值在 [-1, 1] 之间
        img = self.img_transforms(Image.fromarray(img.copy()))

        # 标签：numpy → Tensor(512, 512) → unsqueeze(0) → Tensor(1, 512, 512)
        #   最终是 Tensor(1, 512, 512)，值只有 0 或 1
        lab = self.lab_transform(lab.copy()).unsqueeze(0)

        # ===== 返回 =====
        # 返回一个字典，包含4样东西：
        #   'image'   : 处理好的图片Tensor
        #   'label'   : 处理好的标签Tensor
        #   'A_paths' : 图片的硬盘路径
        #   'B_paths' : 标签的硬盘路径
        return {
            'image': img,
            'label': lab,
            'A_paths': img_path,
            'B_paths': lab_path
        }

    def __len__(self):
        """返回这个数据集总共有多少张图"""
        return len(self.img_paths)
