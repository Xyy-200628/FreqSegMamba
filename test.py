'''
Author: Hui Liu
Github: https://github.com/Karl1109
Email: liuhui@ieee.org
'''

import numpy as np
import torch
import argparse
import os
import cv2
from datasets import create_dataset
from models import build_model
from main import get_args_parser

parser = argparse.ArgumentParser('FreqSegMamba FOR CRACK', parents=[get_args_parser()])
parser.add_argument('--checkpoint', type=str, default=None,
                    help='要评估的模型权重路径；不填则自动找 checkpoint_best.pth')
args = parser.parse_args()
args.phase = 'test'

if __name__ == '__main__':
    args.batch_size = 1
    t_all = []
    device = torch.device(args.device)
    test_dl = create_dataset(args)
    # 优先用 --checkpoint 指定的权重；否则自动找 checkpoint_best.pth
    if args.checkpoint:
        load_model_file = args.checkpoint
    else:
        import glob
        cands = sorted(glob.glob('./checkpoints/weights/*/checkpoint_best.pth'))
        load_model_file = cands[-1] if cands else None
        if load_model_file is None:
            raise FileNotFoundError('未找到 checkpoint_best.pth，请用 --checkpoint 指定权重')
    print(f"Load Model: {load_model_file}")
    data_size = len(test_dl)
    model, criterion = build_model(args)
    state_dict = torch.load(load_model_file)
    model.load_state_dict(state_dict["model"])
    model.to(device)
    print("Load Model Successful!")
    suffix = load_model_file.split('/')[-2]
    save_root = "./results/results_test/" + suffix
    if not os.path.isdir(save_root):
        os.makedirs(save_root)
    with torch.no_grad():
        model.eval()
        for batch_idx, (data) in enumerate(test_dl):
            x = data["image"]
            target = data["label"]
            if device != 'cpu':
                x, target = x.cuda(), target.to(dtype=torch.int64).cuda()
            out = model(x)

            target = target[0, 0, ...].cpu().numpy()
            out = out[0, 0, ...].cpu().numpy()
            root_name = data["A_paths"][0].split("/")[-1][0:-4]
            target = 255 * (target / np.max(target))
            out = 255 * (out / np.max(out))

            cv2.imwrite(os.path.join(save_root, "{}_lab.png".format(root_name)), target)
            cv2.imwrite(os.path.join(save_root, "{}_pre.png".format(root_name)), out)

    print("Finished!")
