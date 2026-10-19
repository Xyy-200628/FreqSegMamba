# -*- coding: utf-8 -*-
"""
final_test.py —— 加载 best checkpoint 跑测试集并一次性算指标(mIoU/ODS/OIS/F1/P/R)

用法:
  python final_test.py \
    --checkpoint 'checkpoints/weights/<目录名>/checkpoint_best.pth' \
    --dataset_path ../data/CrackMap --device cuda:0

预测图存到 ./results/results_test/<目录名>/，并直接打印最终成绩。
"""
import os
import numpy as np
import cv2
import torch
import argparse
import logging
from models import build_model
from datasets import create_dataset
from eval.evaluate import eval
from main import get_args_parser

parser = argparse.ArgumentParser('FINAL TEST', parents=[get_args_parser()])
parser.add_argument('--checkpoint', type=str, required=True, help='checkpoint_best.pth 路径')
args = parser.parse_args()
args.phase = 'test'
args.batch_size = 1

device = torch.device(args.device)
model, criterion = build_model(args)
ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
model.load_state_dict(ckpt['model'])
model.to(device)
model.eval()
print('加载权重:', args.checkpoint)

save_root = './results/results_test/' + args.checkpoint.split('/')[-2]
os.makedirs(save_root, exist_ok=True)

dl = create_dataset(args)
with torch.no_grad():
    for data in dl:
        x, target = data['image'], data['label']
        if device != 'cpu':
            x, target = x.cuda(), target.to(dtype=torch.int64).cuda()
        out = model(x)
        target = target[0, 0].cpu().numpy()
        out = out[0, 0].cpu().numpy()
        root = data['A_paths'][0].split('/')[-1][:-4]
        cv2.imwrite(os.path.join(save_root, root + '_lab.png'),
                    255 * (target / np.max(target)))
        cv2.imwrite(os.path.join(save_root, root + '_pre.png'),
                    255 * (out / np.max(out)))

logging.basicConfig(level=logging.INFO)
r = eval(logging.getLogger('e'), save_root, 'best')
print('===== 最终测试成绩 =====')
print('mIoU=%.4f  ODS=%.4f  OIS=%.4f  F1=%.4f  P=%.4f  R=%.4f' %
      (r['mIoU'], r['ODS'], r['OIS'], r['F1'], r['Precision'], r['Recall']))
