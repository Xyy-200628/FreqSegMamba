'''
Author: Hui Liu
Github: https://github.com/Karl1109
Email: liuhui@ieee.org

=============================================================================
main.py —— 训练的总开关（你运行 python main.py 就从这个文件开始执行）

这个文件做什么？
  就像一台洗衣机的控制面板：
    ① 设置参数（水温、转速、时间）   ← get_args_parser()
    ② 准备（放衣服、加洗衣液）       ← 创建模型、加载数据
    ③ 循环洗衣服                     ← 50轮训练循环
    ④ 洗完晾干                       ← 保存最佳模型

用到的其他文件：
  models/decoder.py     → 搭积木：把 WAVSS 主干 + MFS 融合 拼在一起
  datasets/crack_dataset.py → 读图片：从硬盘加载裂缝照片和标签
  engine.py             → 练一轮：把300张图都过一遍
  eval/evaluate.py      → 打分：算mIoU、F1这些指标
  util/logger.py        → 记笔记：把训练过程写到txt日志
  util/misc.py          → 存文件：把模型权重保存到磁盘
'''

import os
# 告诉电脑用哪张显卡。'0' = 第0张卡。一张卡就写'0'，多张卡可写'0,1,2,3'
# setdefault：只有环境变量里没指定时才默认用 0，这样可在外层用 CUDA_VISIBLE_DEVICES=N 指定卡
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')

import argparse                         # 处理命令行参数（--epochs 50 这类）
import datetime                         # 显示时间
import random                           # 随机数
import time                             # 计时
from pathlib import Path                # 创建文件夹
import numpy as np
import torch
import util.misc as utils               # 工具箱：存文件等功能
from engine import train_one_epoch      # 练一轮的函数
from models import build_model          # 搭模型的函数
from datasets import create_dataset     # 读数据的函数
import cv2                              # 保存预测结果图片
from eval.evaluate import eval          # 打分的函数（算mIoU）
from util.logger import get_logger      # 写日志的函数
from tqdm import tqdm                   # 进度条
from mmengine.optim.scheduler.lr_scheduler import PolyLR  # 学习率逐渐降低的策略


def get_args_parser():
    '''
    定义所有可以调整的参数（旋钮）。
    每个参数都有默认值，你不改就用默认的。
    '''
    parser = argparse.ArgumentParser('FreqSegMamba FOR CRACK', add_help=False)

    # ================================================================
    # ★★★ 常用参数——训练时改这几个就行 ★★★
    # ================================================================

    # 你的数据集放在哪个文件夹？ 例：/home/gpu/data/DeepCrack
    parser.add_argument('--dataset_path', default="../data/TUT",
                        help='数据集文件夹路径')

    # 训练多少轮？1轮 = 把300张图都看一遍。越多越久，但太少学不好
    parser.add_argument('--epochs', default=50, type=int,
                        help='训练轮数')

    # 学习率：每次更新参数时迈多大步。
    # 太大 → 跳过正确答案、不收敛。太小 → 进步太慢。
    # 5e-4 = 0.0005，是比较稳的值
    parser.add_argument('--lr', default=5e-4, type=float,
                        help='学习率')

    # 一次喂几张图给模型？1 表示一次一张，显存不够就保持1
    parser.add_argument('--batch_size_train', type=int, default=1,
                        help='训练时一次取几张图')
    parser.add_argument('--batch_size_test', type=int, default=1,
                        help='测试时一次取几张图')

    # 学习率怎么降低？PolyLR是慢慢降，比较平滑
    parser.add_argument('--lr_scheduler', type=str, default='PolyLR',
                        help='学习率降低策略 [PolyLR|StepLR|CosLR]')
    parser.add_argument('--T_0', default=10, type=int,
                        help='CosLR重启周期(轮)，越小重启越频繁')

    # 模型文件存哪里？
    parser.add_argument('--output_dir', default='./checkpoints/weights',
                        help='模型权重保存位置')

    # 用什么优化器？默认 AdamW（效果比较好），加 --sgd 就用 SGD
    parser.add_argument('--sgd', action='store_true',
                        help='加这个参数就换SGD优化器')

    # 图片统一缩放到多大？512×512 是常用尺寸
    parser.add_argument('--load_width', type=int, default=512,
                        help='图片宽度')
    parser.add_argument('--load_height', type=int, default=512,
                        help='图片高度')

    # ================================================================
    # 以下参数一般不用改
    # ================================================================

    # BCE损失和Dice损失的比例，加起来等于1就行。0.83+0.17=1
    # 边界损失权重（FP-ADB）：λ=0.05，弱监督边缘引导
    parser.add_argument('--edge_weight', default=0.05, type=float,
                        help='边界损失权重(0=不启用，0.05=默认弱监督)')

    # 消融实验开关：默认全关（不影响正常训练）
    parser.add_argument('--no_wavelet', action='store_true',
                        help='消融: 关闭小波模块，SSM直接在全分辨率扫描')
    parser.add_argument('--no_tpas', action='store_true',
                        help='消融: TPAS退化为固定4方向SASS扫描')
    parser.add_argument('--scan_strategy', type=str, default='sass',
                        help='固定扫描策略 [sass|parallel|diag|parasna|diagsna]（仅 --no_tpas 时生效）')
    parser.add_argument('--tpas_mode', type=str, default='sass_dir',
                        help='TPAS方向码模式 [sass_dir|theta_dir|theta_soft|theta_path|adaptive4|greedy|hybrid]：'
                             'sass_dir=固定路径+静态逐像素扫描方向码(默认基线)；'
                             'theta_dir=固定路径+输入图像逐样本裂缝朝向码(稳定+数据驱动)；'
                             'theta_soft=固定路径+连续方向条件化(cos2θ,sin2θ,coh投影)；'
                             'theta_path=SASS4条+输入图驱动的自适应贴缝4条(稳定自适应路径)；'
                             'adaptive4=4条SASS锚全部改造为逐样本自适应路径(阈值门控, 真·自适应)；'
                             'greedy/hybrid=旧的贪心贴缝路径(源自学习特征, 会漂移)')
    parser.add_argument('--tpas_tau_coh', type=float, default=0.35,
                        help='adaptive4 门控①: 候选像素最小相干性(低于则不跟)')
    parser.add_argument('--tpas_tau_win', type=int, default=12,
                        help='adaptive4 门控②: 允许偏离锚路径的最大窗口(步)')
    parser.add_argument('--freq_mode', type=str, default='default',
                        help='频域-空域角色分配 [default|swap|all_ssm|all_conv]（default=LL走SSM/高频走Conv；swap=对调；all_ssm=全SSM；all_conv=全Conv）')
    parser.add_argument('--no_adp', action='store_true',
                        help='消融: 关闭边缘先验分支(ADP-Stream+CFAF+边界头)')

    # 实验标签：加进目录名，避免多卡并行同时启动时目录撞名
    parser.add_argument('--tag', default='', type=str,
                        help='实验标签(如 baseline/wavss/tpas)，会拼进 checkpoint 目录名')

    parser.add_argument('--BCELoss_ratio', default=0.83, type=float,
                        help='BCE损失占比')
    parser.add_argument('--DiceLoss_ratio', default=0.17, type=float,
                        help='Dice损失占比')

    # GN = 分组归一化（适合小batch）。BN = 批归一化（batch太小不稳定）
    parser.add_argument('--Norm_Type', default='GN', type=str,
                        help='归一化类型 [GN|BN]')

    # 学习率的最低值（PolyLR 降到最后不会低于这个数）
    parser.add_argument('--min_lr', default=1e-6, type=float,
                        help='最低学习率')

    # L2正则化系数：惩罚过大的权重，防止过拟合
    parser.add_argument('--weight_decay', default=0.01, type=float,
                        help='权重衰减系数')

    # 如果训练中断了，可以从第N轮接着训（配合 checkpoint.pth 用）
    parser.add_argument('--start_epoch', default=0, type=int,
                        help='从第几轮开始（断点续训用）')

    # 选 StepLR 时，每多少轮学习率减半
    parser.add_argument('--lr_drop', default=30, type=int,
                        help='StepLR: 每几轮减半一次')

    # cpu 或 cuda。用CPU训练巨慢，一般不改
    parser.add_argument('--device', default='cuda',
                        help='计算设备 [cuda|cpu]')

    # 随机种子：固定后每次训练结果一样（方便做实验对比）
    parser.add_argument('--seed', default=42, type=int,
                        help='随机种子')

    # 数据集类型。crack = 裂缝分割专用格式，不改
    parser.add_argument('--dataset_mode', type=str, default='crack',
                        help='数据集类型')

    # 训练时图片打乱（默认），测试时按顺序
    parser.add_argument('--serial_batches', action='store_true',
                        help='不开=随机打乱，开了=按文件名顺序')

    # 用几个线程读图。1就够，调大可能快一点但占CPU
    parser.add_argument('--num_threads', default=1, type=int,
                        help='读图线程数')

    # 当前模式：train=训练 / val=验证(选模型) / test=测试(出最终成绩)
    parser.add_argument('--phase', type=str, default='train',
                        help='运行模式 [train|val|test]')

    # 泄漏协议开关：开启后在"测试集"上选模型（复现 SCSegamba 原协议，仅用于消融验证）
    parser.add_argument('--leaky', action='store_true',
                        help='泄漏协议：在测试集上选模型（默认关闭，用验证集选模型）')

    return parser


def run_eval(model, criterion, args, device, phase, save_root, log_eval, epoch):
    """
    在指定 phase (val/test) 上跑推理并保存预测图，然后计算指标。

    返回 metrics 字典（含 mIoU / ODS / OIS / F1 / Precision / Recall / epoch）。
    训练循环里用 phase='val' 来选最佳模型；训练结束后用 phase='test' 出最终成绩。
    """
    args.phase = phase
    args.batch_size = args.batch_size_test
    dl = create_dataset(args)
    os.makedirs(save_root, exist_ok=True)

    model.eval()
    progress_bar = tqdm(total=len(dl), desc=f"{phase} 评估中")
    with torch.no_grad():
        for data in dl:
            x = data["image"]
            target = data["label"]
            if device != 'cpu':
                x, target = x.cuda(), target.to(dtype=torch.int64).cuda()

            out = model(x)                                # 模型预测
            loss = criterion(out, target.float())         # 算损失

            target = target[0, 0, ...].cpu().numpy()
            out = out[0, 0, ...].cpu().numpy()
            root_name = data["A_paths"][0].split("/")[-1][0:-4]
            target = 255 * (target / np.max(target))
            out = 255 * (out / np.max(out))
            cv2.imwrite(os.path.join(save_root, f"{root_name}_lab.png"), target)
            cv2.imwrite(os.path.join(save_root, f"{root_name}_pre.png"), out)
            progress_bar.set_description(f"Loss: {loss.item():.4f}")
            progress_bar.update(1)
    progress_bar.close()

    return eval(log_eval, save_root, epoch)               # → eval/evaluate.py


def main(args):
    '''
    训练的主函数。整个流程分6步。
    '''

    # ================================================================
    # 第1步：准备"笔记本"——创建文件夹和日志文件
    # ================================================================
    # 用"当前时间+数据集名"给这次训练取个名字
    cur_time = time.strftime('%Y_%m_%d_%H:%M:%S', time.localtime(time.time()))
    if args.tag:
        cur_time = cur_time + '_' + args.tag
    dataset_name = (args.dataset_path).split('/')[-1]  # 比如 'DeepCrack'
    log_folder = os.path.join('./checkpoints',
                              cur_time + '_Dataset->' + dataset_name)
    os.makedirs(log_folder, exist_ok=True)             # 创建日志文件夹

    # 三本"笔记本"：一本记训练、一本记测试、一本记评估分数
    log_train = get_logger(log_folder, 'train')        # → util/logger.py
    log_test  = get_logger(log_folder, 'test')         # → util/logger.py
    log_eval  = get_logger(log_folder, 'eval')         # → util/logger.py

    # 把你用的参数记在笔记本第一行，方便以后回顾
    log_train.info("参数: " + str(args))

    # ================================================================
    # 第2步：固定随机性——保证每次训练结果一样
    # ================================================================
    device = torch.device(args.device)   # GPU还是CPU？
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)              # PyTorch的随机种子
    np.random.seed(seed)                 # numpy的随机种子
    random.seed(seed)                    # Python的随机种子

    # ================================================================
    # 第3步：搭积木——创建模型、损失函数、数据加载器
    # ================================================================
    # model: 裂缝分割网络 (FreqSegMamba)
    # criterion: 损失函数 (BCE + Dice)，用来判断"预测和标签差多少"
    model, criterion = build_model(args)                   # → models/decoder.py
    model.to(device)                                       # 搬到GPU

    # 数据加载器：自动读图片、resize、归一化、随机打乱
    args.batch_size = args.batch_size_train
    train_dataLoader = create_dataset(args)                # → datasets/crack_dataset.py
    print(f'训练图片数量 = {len(train_dataLoader)}')
    log_train.info(f'训练图片数量 = {len(train_dataLoader)}')

    # ================================================================
    # 第4步：找"教练"——优化器 + 学习率策略
    # ================================================================
    # 收集模型所有参数
    model_params = [{"params": [p for n, p in model.named_parameters()],
                     "lr": args.lr}]

    # 选教练：AdamW（自适应，适合新手）或 SGD（经典，需要调参）
    if args.sgd:
        optimizer = torch.optim.SGD(model_params, lr=args.lr,
                                    momentum=0.9, weight_decay=args.weight_decay)
        print('使用 SGD 优化器')
    else:
        optimizer = torch.optim.AdamW(model_params, lr=args.lr,
                                      weight_decay=args.weight_decay)
        print('使用 AdamW 优化器')

    # 学习率怎么降：一开始大步走，越往后步子越小
    if args.lr_scheduler == 'StepLR':
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)
    elif args.lr_scheduler == 'CosLR':
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=args.T_0, T_mult=2, eta_min=1e-5)
    elif args.lr_scheduler == 'PolyLR':
        lr_scheduler = PolyLR(optimizer, eta_min=args.min_lr,
                              begin=args.start_epoch, end=args.epochs)
    else:
        raise ValueError(f"不支持的学习率策略: {args.lr_scheduler}")

    # 创建存模型权重的文件夹
    weight_dir = args.output_dir + '/' + cur_time + '_Dataset->' + dataset_name
    Path(weight_dir).mkdir(parents=True, exist_ok=True)
    weight_dir = Path(weight_dir)

    # ================================================================
    # 第5步：★ 训练循环（核心）——反复做"训练→保存→测试→打分"
    # ================================================================
    print("开始训练！")
    log_train.info("开始训练！")
    start_time = time.time()

    best_mIoU = 0                                        # 记下最高分
    best_result = {'epoch': 0, 'mIoU': 0, 'F1': 0}      # 记下最佳那轮的完整成绩

    for epoch in range(args.start_epoch, args.epochs):    # epoch从0到49，重复50次
        print("─" * 80)
        print(f"训练轮次 {epoch} 开始")

        # ============ 5a. 训练一轮 ============
        # engine.py 里的函数：把300张图都过一遍，算loss，更新参数
        train_one_epoch(model, criterion, train_dataLoader,
                        optimizer, epoch, args, log_train)
        lr_scheduler.step()  # 微微降低学习率

        # ============ 5b. 保存模型文件 ============
        if args.output_dir:
            # checkpoint.pth：永远存最新的（下次可以接着训）
            utils.save_on_master({
                'model': model.state_dict(),           # 模型参数
                'optimizer': optimizer.state_dict(),   # 优化器状态（断点续训需要）
                'lr_scheduler': lr_scheduler.state_dict(),
                'epoch': epoch,
                'args': args,
            }, weight_dir / 'checkpoint.pth')

            # checkpoint{epoch}.pth：这轮单独存一份（方便回溯）
            utils.save_on_master({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'epoch': epoch,
                'args': args,
            }, weight_dir / f'checkpoint{epoch}.pth')

        print(f"训练轮次 {epoch} 结束")

        # ============ 5c. 评估——选最佳模型 ============
        # 默认诚实协议：在验证集(val)上选模型；
        # --leaky 泄漏协议：在测试集(test)上选模型（复现 SCSegamba 原协议，仅用于消融验证）
        if args.leaky:
            eval_phase = 'test'
            eval_label = '测试'
        else:
            eval_phase = 'val'
            eval_label = '验证'

        print(f"{eval_label}轮次 {epoch} 开始")

        save_root_eval = f'./results/{cur_time}_Dataset->{dataset_name}/results_{eval_phase}_{epoch}'
        metrics = run_eval(model, criterion, args, device, eval_phase,
                           save_root_eval, log_eval, epoch)

        for key, value in metrics.items():
            print(f"  {key} = {value:.4f}")

        # 如果这轮 mIoU 比之前最好的还高 → 更新最佳模型
        if metrics['mIoU'] > best_mIoU:
            best_mIoU = metrics['mIoU']
            best_result = metrics
            utils.save_on_master({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'epoch': epoch,
                'args': args,
            }, weight_dir / 'checkpoint_best.pth')
            print(f"  ★ 新最佳模型！保存于epoch {epoch}，{eval_phase} mIoU = {best_mIoU:.4f}")

        print(f"当前最高分: {eval_phase} mIoU = {best_mIoU:.4f} (第{best_result['epoch']}轮)")
        print(f"{eval_label}轮次 {epoch} 结束")

    # ================================================================
    # 第6步：训练结束 → 加载最佳模型 → 在测试集上跑最终成绩
    #   验证集只用来选模型，测试集才是论文表 1 里报告的数字。
    # ================================================================
    print("训练结束！加载最佳模型，在测试集上评估最终成绩...")
    best_ckpt = weight_dir / 'checkpoint_best.pth'
    if best_ckpt.exists():
        ckpt = torch.load(best_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model'])
        log_test.info(f"加载最佳模型 epoch {ckpt['epoch']}，开始最终测试")
        print(f"已加载最佳模型 epoch {ckpt['epoch']}，其验证集 mIoU = {best_mIoU:.4f}")
    else:
        log_test.info("未找到 checkpoint_best.pth，用当前(最后一轮)模型跑测试")
        print("未找到 checkpoint_best.pth，用当前(最后一轮)模型跑测试")

    save_root_test = f'./results/{cur_time}_Dataset->{dataset_name}/results_test'
    final_metrics = run_eval(model, criterion, args, device, 'test',
                             save_root_test, log_eval, epoch='best')

    log_test.info("===== 最终测试集成绩（报告用） =====")
    for key, value in final_metrics.items():
        if key == 'epoch':
            continue
        print(f"  [最终测试] {key} = {value:.4f}")
        log_test.info(f"  [最终测试] {key} = {value:.4f}")

    # ================================================================
    # 第7步：全部训练完毕，总结
    # ================================================================
    for key, value in best_result.items():
        log_eval.info(f"{key} = {value}")

    total_seconds = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_seconds)))
    print(f'训练完成！总耗时: {total_time_str}')
    print(f'最佳模型: epoch {best_result["epoch"]}, mIoU = {best_mIoU:.4f}')
    log_train.info(f'训练完成！总耗时: {total_time_str}')


# ================================================================
# 程序入口：当你敲 python main.py 时从这里开始执行
# ================================================================
if __name__ == '__main__':
    # ① 读取命令行参数（如 --epochs 100 --lr 1e-4）
    parser = argparse.ArgumentParser('FreqSegMamba FOR CRACK',
                                     parents=[get_args_parser()])
    args = parser.parse_args()

    # ② 调用上面的 main 函数开始训练
    main(args)
