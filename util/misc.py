
"""
=============================================================================
util/misc.py  —— 工具函数集合
=============================================================================

本文件提供训练过程中需要的辅助工具，主要分为三类：

┌─────────────────────────────────────────────────────────────────┐
│ ① SmoothedValue     — 平滑记录器：跟踪loss等指标的滑动平均值        │
│    MetricLogger      — 指标日志器：管理多个 SmoothedValue          │
│                                                                  │
│ ② save_on_master    — 只在主进程保存文件（避免多卡保存多次）        │
│    get_rank / get_world_size — 获取当前进程在分布式中的身份         │
│    init_distributed_mode — 初始化多GPU分布式训练                   │
│    all_gather / reduce_dict — 多卡数据同步                        │
│                                                                  │
│ ③ accuracy          — 分类准确率计算                               │
│    get_total_grad_norm — 梯度范数（监控梯度是否爆炸/消失）           │
│    inverse_sigmoid   — 逆Sigmoid函数                              │
└─────────────────────────────────────────────────────────────────┘

新手提示：
  - 单卡训练时，is_main_process() 始终返回 True
  - get_rank() 单卡时返回 0，多卡时返回当前卡编号(0,1,2...)
  - main.py 里调用的 save_on_master() 本质就是 torch.save()
  - MetricLogger 用于训练中打印进度条和控制台日志
"""

import os
import subprocess
import time
from collections import defaultdict, deque
import datetime
import pickle
from typing import Optional, List

import torch
import torch.nn as nn
import torch.distributed as dist     # 多GPU分布式通信
from torch import Tensor

import torchvision


# ============================================================================
# ① SmoothedValue —— 平滑值记录器
# ============================================================================
class SmoothedValue(object):
    """
    用一个固定大小的窗口记录历史值，提供：
      - median: 窗口内中位数（抗异常值）
      - avg:    窗口内均值（最近趋势）
      - global_avg: 全局均值（整体趋势）
      - max / value: 窗口内最大值 / 最新值

    通俗理解：
      就像"过去20场比赛的平均得分"——不是只看最近一场，
      也不是把所有场次都算上，而是滑动窗口。

    使用场景：
      main.py 里用 MetricLogger（内部就是 SmoothedValue）
      来平滑打印训练时的 loss 值。
    """

    def __init__(self, window_size=20, fmt=None):
        """
        参数:
          window_size: 窗口大小，默认20。记录最近20次的数值。
          fmt:         打印格式，默认 "{median:.4f} ({global_avg:.4f})"
        """
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)  # 定长队列，满了自动丢掉最旧的
        self.total = 0.0                         # 总和（用于算全局均值）
        self.count = 0                           # 总次数（用于算全局均值）
        self.fmt = fmt

    def update(self, value, n=1):
        """
        记录一个新值
        参数:
          value: 本次记录的数值（如 loss=0.35）
          n:     权重（通常为1，表示一个样本/一个batch）
        """
        self.deque.append(value)     # 加入窗口
        self.count += n              # 累计次数
        self.total += value * n      # 累计总和（value * n 支持加权）

    def synchronize_between_processes(self):
        """多GPU训练时，同步所有卡上的 total 和 count（保证全局均值一致）"""
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)           # 所有卡的值求和
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    # ---- 以下 @property 把方法变成属性，用 meter.median 而不用 meter.median() ----
    @property
    def median(self):
        """窗口内中位数（不受极端值影响）"""
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        """窗口内均值（反映最近趋势）"""
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        """全局均值 = 总和 / 总次数（反映整体水平）"""
        return self.total / self.count

    @property
    def max(self):
        """窗口内最大值"""
        return max(self.deque)

    @property
    def value(self):
        """最新一次记录的值"""
        return self.deque[-1]

    def __str__(self):
        """打印时输出格式化字符串，如 "0.3500 (0.4200)" 表示 中位数(全局均值)"""
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


# ============================================================================
# all_gather —— 多卡数据收集
# ============================================================================
def all_gather(data):
    """
    分布式训练时，从所有GPU收集任意Python对象（不限于Tensor）。

    通俗理解：
      每张卡上有一段数据（如当前batch的loss），all_gather 把它们收集
      到一张列表里，让每张卡都看到所有卡的数据。

    参数:
      data: 任意可被 pickle 序列化的 Python 对象
    返回:
      list[data]: 每张卡上的数据组成的列表

    单卡时直接返回 [data]。
    """
    # 获取总GPU数
    world_size = get_world_size()
    if world_size == 1:
        return [data]                    # 单卡，直接返回

    # 将数据序列化为字节流 → 转成 Tensor
    buffer = pickle.dumps(data)                      # Python对象 → 字节
    storage = torch.ByteStorage.from_buffer(buffer)   # 字节 → torch存储
    tensor = torch.ByteTensor(storage).to("cuda")     # 存储 → GPU Tensor

    # 获取每张卡上数据的大小（不同卡可能有不同大小的数据）
    local_size = torch.tensor([tensor.numel()], device="cuda")
    size_list = [torch.tensor([0], device="cuda") for _ in range(world_size)]
    dist.all_gather(size_list, local_size)           # 收集所有卡的大小
    size_list = [int(size.item()) for size in size_list]
    max_size = max(size_list)

    # 把所有卡的数据收集起来（不足max_size的补0）
    tensor_list = []
    for _ in size_list:
        tensor_list.append(torch.empty((max_size,), dtype=torch.uint8, device="cuda"))
    if local_size != max_size:
        padding = torch.empty(size=(max_size - local_size,), dtype=torch.uint8, device="cuda")
        tensor = torch.cat((tensor, padding), dim=0)  # 补0对齐
    dist.all_gather(tensor_list, tensor)

    # 将字节流还原为Python对象
    data_list = []
    for size, tensor in zip(size_list, tensor_list):
        buffer = tensor.cpu().numpy().tobytes()[:size]  # 去掉padding
        data_list.append(pickle.loads(buffer))           # 字节 → Python对象

    return data_list


# ============================================================================
# reduce_dict —— 多卡字典数据归约
# ============================================================================
def reduce_dict(input_dict, average=True):
    """
    多GPU训练时，对所有卡上字典中同名key的值求平均（或求和）。

    使用场景：
      每个GPU算出了各自的loss，需要求平均得到本step的最终loss。

    参数:
      input_dict: 如 {'loss': 0.35, 'lr': 0.001}
      average:    True=求平均, False=求和
    返回:
      归约后的字典
    """
    world_size = get_world_size()
    if world_size < 2:
        return input_dict                # 单卡，不处理

    with torch.no_grad():
        names = []
        values = []
        # 按key字母排序，确保所有卡的顺序一致
        for k in sorted(input_dict.keys()):
            names.append(k)
            values.append(input_dict[k])
        values = torch.stack(values, dim=0)    # 多个值堆叠成Tensor
        dist.all_reduce(values)                # 所有卡求和
        if average:
            values /= world_size               # 除以卡数 = 平均
        reduced_dict = {k: v for k, v in zip(names, values)}
    return reduced_dict


# ============================================================================
# ① MetricLogger —— 训练指标日志器
# ============================================================================
class MetricLogger(object):
    """
    管理多个 SmoothedValue，统一打印训练进度。

    主要方法：
      .update(loss=0.35, lr=0.001)  — 记录一次指标
      .meters['loss']               — 获取某个指标的 SmoothedValue
      .log_every(iterator, 10)      — 每10个batch打印一次进度

    使用示例（见 engine.py）：
      metric_logger = MetricLogger(delimiter="  ")
      metric_logger.update(loss=loss_value, lr=current_lr)
      metric_logger.meters['loss'].global_avg  # 获取平均loss
    """

    def __init__(self, delimiter="\t"):
        """
        参数:
          delimiter: 打印时各指标间的分隔符，默认Tab
        """
        self.meters = defaultdict(SmoothedValue)  # 自动创建 SmoothedValue
        self.delimiter = delimiter

    def update(self, **kwargs):
        """
        批量更新多个指标。

        示例：
          metric_logger.update(loss=0.35, lr=0.0005, grad_norm=2.1)
          → 同时记录 loss、lr、grad_norm 三个指标
        """
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()                     # Tensor转Python数字
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        """
        支持 metric_logger.loss 这样的写法。
        内部会去 self.meters 里找对应的 SmoothedValue。
        """
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        """打印所有指标的值，如 "loss: 0.3500 (0.4200)  lr: 0.000500 (0.000500)" """
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append("{}: {}".format(name, str(meter)))
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        """多卡同步所有 SmoothedValue"""
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        """手动添加一个 SmoothedValue"""
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        """
        遍历 iterable，每 print_freq 次打印一次进度。

        这是训练循环中最常用的方法，用法：
          for batch in metric_logger.log_every(data_loader, print_freq=10, header='Train'):
              ...

        参数:
          iterable:   可迭代对象（通常是 DataLoader）
          print_freq: 打印频率（每N个batch打印一次）
          header:     打印前缀（如 'Epoch: [0]'）
        """
        i = 0
        if not header:
            header = ''

        start_time = time.time()
        end = time.time()

        # 两个计时器：iter_time=每步迭代时间，data_time=数据加载等待时间
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')

        # 格式化：让进度数字对齐
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'

        # 构建打印模板
        if torch.cuda.is_available():
            log_msg = self.delimiter.join([
                header,
                '[{0' + space_fmt + '}/{1}]',    # [  5/100]
                'eta: {eta}',                     # 预计剩余时间
                '{meters}',                       # 各项指标值
                'time: {time}',                   # 每步耗时
                'data: {data}',                   # 数据加载耗时
                'max mem: {memory:.0f}'           # GPU显存峰值(MB)
            ])
        else:
            log_msg = self.delimiter.join([
                header,
                '[{0' + space_fmt + '}/{1}]',
                'eta: {eta}',
                '{meters}',
                'time: {time}',
                'data: {data}'
            ])

        MB = 1024.0 * 1024.0  # 1MB = 1024×1024 bytes

        for obj in iterable:
            data_time.update(time.time() - end)    # 记录数据等待时间
            yield obj                               # ★ 把当前batch返回给调用方
            iter_time.update(time.time() - end)    # 记录迭代计算时间

            # 达到打印频率 → 输出日志
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()

        # 全部遍历完，打印总耗时和平均每步时间
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {} ({:.4f} s / it)'.format(
            header, total_time_str, total_time / len(iterable)))


# ============================================================================
# get_sha —— 获取Git提交信息（用于记录实验版本）
# ============================================================================
def get_sha():
    """返回当前代码的Git SHA、分支名和是否有未提交修改"""
    cwd = os.path.dirname(os.path.abspath(__file__))

    def _run(command):
        return subprocess.check_output(command, cwd=cwd).decode('ascii').strip()

    sha = 'N/A'
    diff = "clean"
    branch = 'N/A'
    try:
        sha = _run(['git', 'rev-parse', 'HEAD'])                 # 当前commit哈希
        subprocess.check_output(['git', 'diff'], cwd=cwd)
        diff = _run(['git', 'diff-index', 'HEAD'])               # 是否有未提交修改
        diff = "has uncommited changes" if diff else "clean"
        branch = _run(['git', 'rev-parse', '--abbrev-ref', 'HEAD'])  # 分支名
    except Exception:
        pass
    message = f"sha: {sha}, status: {diff}, branch: {branch}"
    return message


# ============================================================================
# 辅助函数：_max_by_axis
# ============================================================================
def _max_by_axis(the_list):
    """
    对整数列表的列表，逐位置取最大值。

    例: _max_by_axis([[2,5], [3,1], [1,4]]) → [3, 5]
    """
    maxes = the_list[0]
    for sublist in the_list[1:]:
        for index, item in enumerate(sublist):
            maxes[index] = max(maxes[index], item)
    return maxes


# ============================================================================
# ② 分布式训练相关
# ============================================================================
def setup_for_distributed(is_master):
    """
    分布式训练时，让非主进程不打印（避免N张卡打印N次重复日志）。

    原理：替换Python内置的 print 函数。
    is_master=True 的进程正常打印，其他进程的 print 被"静音"。
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:               # 主进程 / 强制打印 → 输出
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def is_dist_avail_and_initialized():
    """分布式训练是否已就绪"""
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    """获取总GPU数（单卡返回1）"""
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    """获取当前进程的GPU编号（0~world_size-1），单卡返回0"""
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def get_local_size():
    """获取当前节点的GPU数"""
    if not is_dist_avail_and_initialized():
        return 1
    return int(os.environ['LOCAL_SIZE'])


def get_local_rank():
    """获取当前进程在节点内的编号"""
    if not is_dist_avail_and_initialized():
        return 0
    return int(os.environ['LOCAL_RANK'])


def is_main_process():
    """当前进程是否为主进程（rank=0），单卡时始终返回 True"""
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    """
    只在主进程保存文件（避免多卡时重复保存）。

    用法和 torch.save 完全相同:
      save_on_master({'model': model.state_dict()}, 'checkpoint.pth')

    等价于:
      if get_rank() == 0:
          torch.save(...)

    单卡训练时，这个函数和 torch.save 完全一样。
    """
    if is_main_process():
        torch.save(*args, **kwargs)


def init_distributed_mode(args):
    """
    初始化多GPU分布式训练模式。

    支持两种启动方式：
      1. torchrun / torch.distributed.launch 启动（自动设 RANK, WORLD_SIZE）
      2. SLURM 集群调度器启动

    单卡运行时，此函数直接返回，不做任何事。

    所做的事：
      - 设置 args.rank, args.world_size, args.gpu
      - 绑定当前进程到指定GPU（torch.cuda.set_device）
      - 初始化进程间通信（init_process_group）
      - 让非主进程静音打印
    """
    # ---- 方式1: 通过 torchrun 启动 ----
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
        args.dist_url = 'env://'
        os.environ['LOCAL_SIZE'] = str(torch.cuda.device_count())

    # ---- 方式2: 通过 SLURM 集群启动 ----
    elif 'SLURM_PROCID' in os.environ:
        proc_id = int(os.environ['SLURM_PROCID'])
        ntasks = int(os.environ['SLURM_NTASKS'])
        node_list = os.environ['SLURM_NODELIST']
        num_gpus = torch.cuda.device_count()
        addr = subprocess.getoutput(
            'scontrol show hostname {} | head -n1'.format(node_list))
        os.environ['MASTER_PORT'] = os.environ.get('MASTER_PORT', '29500')
        os.environ['MASTER_ADDR'] = addr
        os.environ['WORLD_SIZE'] = str(ntasks)
        os.environ['RANK'] = str(proc_id)
        os.environ['LOCAL_RANK'] = str(proc_id % num_gpus)
        os.environ['LOCAL_SIZE'] = str(num_gpus)
        args.dist_url = 'env://'
        args.world_size = ntasks
        args.rank = proc_id
        args.gpu = proc_id % num_gpus

    # ---- 单卡：直接返回 ----
    else:
        args.distributed = False
        return

    # 多卡：初始化通信
    args.distributed = True
    torch.cuda.set_device(args.gpu)       # 绑定GPU
    args.dist_backend = 'nccl'            # NCCL是NVIDIA的多GPU通信库
    print('| distributed init (rank {}): {}'.format(args.rank, args.dist_url), flush=True)
    torch.distributed.init_process_group(
        backend=args.dist_backend,
        init_method=args.dist_url,
        world_size=args.world_size,
        rank=args.rank)
    torch.distributed.barrier()           # 等待所有进程都初始化完毕
    setup_for_distributed(args.rank == 0) # 非主进程静音


# ============================================================================
# ③ 其他工具函数
# ============================================================================
@torch.no_grad()
def accuracy(output, target, topk=(1,)):
    """
    计算分类任务 top-K 准确率。

    参数:
      output: 模型输出 logits, shape (batch_size, num_classes)
      target: 真实标签, shape (batch_size,)
      topk:   要计算哪些Top-K，如 (1, 5) 表示 Top-1 和 Top-5 准确率

    返回:
      [top1_acc, top5_acc] 列表，单位%

    注意：本项目是做裂缝分割（像素级二分类），一般不用这个函数。
    这是从 torchvision 分类任务复用的遗留代码。
    """
    if target.numel() == 0:
        return [torch.zeros([], device=output.device)]

    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)     # 取概率最高的maxk个类别
    pred = pred.t()                                  # (maxk, batch_size)
    correct = pred.eq(target.view(1, -1).expand_as(pred))  # 和标签比较

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))  # 转百分比
    return res


def get_total_grad_norm(parameters, norm_type=2):
    """
    计算模型所有参数的梯度总范数。

    用途：
      监控训练是否稳定：
      - 梯度范数太大 → 梯度爆炸，需要降低学习率
      - 梯度范数太小 → 梯度消失，模型没在学
      - 正常范围：0.1 ~ 10 左右

    参数:
      parameters: model.parameters()
      norm_type:  范数类型（2 = L2范数，即欧氏距离）
    """
    # 只考虑有梯度的参数（有些参数可能 requires_grad=False）
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    norm_type = float(norm_type)
    device = parameters[0].grad.device

    # 每个参数的梯度算一次范数 → 堆叠 → 再算一次范数 = 整体梯度范数
    total_norm = torch.norm(
        torch.stack([torch.norm(p.grad.detach(), norm_type).to(device)
                     for p in parameters]),
        norm_type
    )
    return total_norm


def inverse_sigmoid(x, eps=1e-5):
    """
    逆Sigmoid函数: log(x / (1-x))

    用途：
      Sigmoid:   (-∞, +∞) → (0, 1)
      逆Sigmoid: (0, 1)   → (-∞, +∞)

    常用于需要把概率值还原到 logits 空间的场景。
    eps 防止取 log(0) 导致数值溢出。
    """
    x = x.clamp(min=0, max=1)            # 确保在 [0,1] 范围
    x1 = x.clamp(min=eps)                # 防止 log(0)
    x2 = (1 - x).clamp(min=eps)          # 防止 log(0)
    return torch.log(x1 / x2)
