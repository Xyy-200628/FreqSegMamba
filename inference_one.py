import os
import sys
import argparse
import cv2
import torch
import numpy as np
import torchvision.transforms as transforms

# 将项目根目录加入 Python 搜索路径，确保能导入 mmcls, models 等模块
sys.path.append(os.getcwd())

from main import get_args_parser, build_model


def inference_single_image(args, img_path, checkpoint_path):
    # 1. 构建模型（tpas_mode / no_adp 等由命令行参数决定，保证和训练时一致）
    print(f"正在加载模型权重: {checkpoint_path}")
    model, _ = build_model(args)
    state_dict = torch.load(checkpoint_path, map_location=args.device)
    model.load_state_dict(state_dict["model"])
    model.to(args.device)
    model.eval()

    # 2. 读取图片（兼容带 alpha 通道的截图）
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"找不到图片: {img_path}")
    if img.ndim == 3 and img.shape[2] == 4:      # RGBA → RGB（去掉 alpha）
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (args.load_width, args.load_height), interpolation=cv2.INTER_CUBIC)

    # 3. 预处理（与官方测试一致：归一化到 [-1, 1]）
    img_tensor = transforms.ToTensor()(img)
    img_tensor = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(img_tensor)
    input_tensor = img_tensor.unsqueeze(0).to(args.device)

    # 4. 推理
    print("开始推理...")
    with torch.no_grad():
        output = model(input_tensor)

    # 5. 二值化输出（裂缝=白 255，背景=黑 0）
    out_mask = output[0, 0, ...].cpu().numpy()
    out_mask = 255 * (out_mask / np.max(out_mask))
    out_mask[out_mask >= 127.5] = 255
    out_mask[out_mask < 127.5] = 0

    out_dir = "./results/single_test"
    os.makedirs(out_dir, exist_ok=True)
    file_name = os.path.basename(img_path).split('.')[0]
    save_path = os.path.join(out_dir, f"{file_name}_pre.png")
    cv2.imwrite(save_path, out_mask.astype(np.uint8))

    print(f"✅ 结果已保存至: {save_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(parents=[get_args_parser()])
    parser.add_argument('--img', type=str, required=True, help='要测试的图片路径')
    parser.add_argument('--checkpoint', type=str, required=True, help='checkpoint_best.pth 路径')
    args = parser.parse_args()
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.load_width = 512
    args.load_height = 512

    inference_single_image(args, args.img, args.checkpoint)
