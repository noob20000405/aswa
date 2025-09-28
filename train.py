import argparse
import os
import sys
import time
import json
import copy
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
import models
import utils
import tabulate
from torch.utils.data import random_split
import pandas as pd

parser = argparse.ArgumentParser(description='SWA-ASWA training')
parser.add_argument('--dir', type=str, default='.', required=False, help='training directory (default: None)')

parser.add_argument('--dataset', type=str, default='CIFAR10', help='dataset name (default: CIFAR10)')
parser.add_argument('--batch_size', type=int, default=128, metavar='N', help='input batch size (default: 128)')
parser.add_argument('--num_workers', type=int, default=16, metavar='N', help='number of workers (default: 16)')
parser.add_argument('--model', type=str, default='VGG16', required=False, metavar='MODEL',
                    help='model name (default: None)')
parser.add_argument('--optim', type=str, default='SGD', help='dataset name (default: CIFAR10)')
parser.add_argument('--epochs', type=int, default=200, metavar='N', help='number of epochs to train (default: 200)')
parser.add_argument('--save_freq', type=int, default=25, metavar='N', help='save frequency (default: 25)')
parser.add_argument('--eval_freq', type=int, default=1, metavar='N', help='evaluation frequency (default: 5)')
parser.add_argument('--lr_init', type=float, default=0.1, metavar='LR', help='initial learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='SGD momentum (default: 0.9)')
parser.add_argument('--wd', type=float, default=1e-4, help='weight decay (default: 1e-4)')

parser.add_argument('--swa', action='store_true', help='swa usage flag (default: off)')
parser.add_argument('--aswa', action='store_true', help='aswa usage flag (default: off)')

parser.add_argument('--swa_start', type=int, default=161, metavar='N',
                    help='SWA start epoch number (default: 161)')
parser.add_argument('--swa_lr', type=float, default=0.05, metavar='LR', help='SWA LR (default: 0.05)')
parser.add_argument('--swa_c_epochs', type=int, default=1, metavar='N',
                    help='SWA model collection frequency/cycle length in epochs (default: 1)')

parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
parser.add_argument('--val_ratio', type=float, default='0.1')

# ---------- 仅新增（默认关闭，不影响原行为）：函数侧集成与 α 学习 ----------
parser.add_argument('--do_func_ens', action='store_true',
                    help='enable function-side ensemble evaluation on SWA snapshots (default: off)')
parser.add_argument('--func_type', type=str, default='both', choices=['linear', 'logpool', 'both'],
                    help='function-side ensemble type when enabled')
parser.add_argument('--func_weights', type=str, default=None,
                    help='optional path to weights (json/npy) for function ensemble; default uniform')
parser.add_argument('--learn_func_w', action='store_true',
                    help='learn function-ensemble weights α on validation (default: off)')
parser.add_argument('--func_w_type', type=str, default='linear', choices=['linear','logpool'],
                    help='objective for α learning (linear / logpool)')
parser.add_argument('--func_w_steps', type=int, default=150, help='steps to optimize α (default 150)')
parser.add_argument('--func_w_lr', type=float, default=0.3, help='lr to optimize α (default 0.3)')
parser.add_argument('--crit_fraction', type=float, default=0.5,
                    help='fraction of lowest-margin val samples used for α learning (default 0.5)')
parser.add_argument('--snapshot_stride', type=int, default=1,
                    help='keep every n-th SWA snapshot during SWA stage (>=1); only used if --do_func_ens')

parser.add_argument('--func_w_l2', type=float, default=0.0,
                    help='L2 regularization coeff λ for α-learning (towards uniform). Default 0 keeps old behavior.')
parser.add_argument('--func_w_eta', type=float, default=0.0,
                    help='Correlation (quadratic) coeff η for α-learning. Default 0 keeps old behavior.')


args = parser.parse_args()

print('Preparing directory %s' % args.dir)
os.makedirs(args.dir, exist_ok=True)

torch.backends.cudnn.benchmark = True
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

print('Using model %s' % args.model)
model_cfg = getattr(models, args.model)

# Dataset
print('Loading dataset:', args.dataset)
assert 1.0 > float(args.val_ratio) > 0.0
if args.dataset == "CIFAR10":
    train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True,
                                             transform=model_cfg.transform_train)
    test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True,
                                            transform=model_cfg.transform_test)

    train_size = int(len(train_set) * (1 - float(args.val_ratio)))
    val_size = len(train_set) - train_size
elif args.dataset == "CIFAR100":
    train_set = torchvision.datasets.CIFAR100(root='./data', train=True, download=True,
                                              transform=model_cfg.transform_train)
    test_set = torchvision.datasets.CIFAR100(root='./data', train=False, download=True,
                                             transform=model_cfg.transform_test)

    train_size = int(len(train_set) * (1 - float(args.val_ratio)))
    val_size = len(train_set) - train_size
else:
    print("Incorred dataset", args.dataset)
    sys.exit(1)

num_classes = max(train_set.targets) + 1
train_set, val_set = random_split(train_set, [train_size, val_size], generator=torch.Generator().manual_seed(args.seed))

# ---- Build a no-augmentation view of *the same* val split ----
# 取出 random_split 产生的 val 下标
assert hasattr(val_set, 'indices'), "val_set must be a torch.utils.data.Subset"
val_indices = val_set.indices

# 用 eval transform 构建一个“无增强的完整训练集”视图，再用相同 indices 做子集
if args.dataset == "CIFAR10":
    full_train_noaug = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=False, transform=model_cfg.transform_test
    )
else:
    full_train_noaug = torchvision.datasets.CIFAR100(
        root='./data', train=True, download=False, transform=model_cfg.transform_test
    )

val_noaug = torch.utils.data.Subset(full_train_noaug, val_indices)
val_noaug_loader = torch.utils.data.DataLoader(
    val_noaug,
    batch_size=args.batch_size,
    shuffle=False,
    num_workers=args.num_workers,
    pin_memory=True
)

# 在构建 val_noaug_loader 之后，紧跟着加：
train_noaug_loader = torch.utils.data.DataLoader(
    full_train_noaug,
    batch_size=args.batch_size,
    shuffle=False,
    num_workers=args.num_workers,
    pin_memory=True
)



# Loaders
print(f"|Train|:{len(train_set)} |Val|:{len(val_set)} |Test|:{len(test_set)}")
loaders = {
    'train': torch.utils.data.DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    ),
    'val': torch.utils.data.DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    ),
    'test': torch.utils.data.DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

}

print('Preparing model')
def build_model():
    return model_cfg.base(*model_cfg.args, num_classes=num_classes, **model_cfg.kwargs)

model = build_model()
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

print('SWA and ASWA training')
swa_model = build_model()
swa_n = 0

aswa_model = build_model()
aswa_model.load_state_dict(swa_model.state_dict())
aswa_ensemble_weights = [0]  # 保持与源码一致

model.to(device)
swa_model.to(device)
aswa_model.to(device)

# 仅在需要做函数侧集成时，才收集快照，避免影响默认行为与内存
swa_snapshots = []  # list of CPU state_dict

def schedule(epoch):
    t = (epoch) / (args.swa_start if args.swa else args.epochs)
    lr_ratio = args.swa_lr / args.lr_init if args.swa else 0.01
    if t <= 0.5:
        factor = 1.0
    elif t <= 0.9:
        factor = 1.0 - (1.0 - lr_ratio) * (t - 0.5) / 0.4
    else:
        factor = lr_ratio
    return args.lr_init * factor

criterion = F.cross_entropy
if args.optim == "SGD":
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr_init, momentum=args.momentum, weight_decay=args.wd)
elif args.optim == "Adam":
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr_init)
else:
    print("NNN")
    sys.exit(1)

start_epoch = 0
df = []
for epoch in range(start_epoch, args.epochs):
    time_ep = time.time()
    # (Adjust LR)
    if args.optim == "SGD" and epoch > 0:
        lr = schedule(epoch)
        utils.adjust_learning_rate(optimizer, lr)
    else:
        lr = args.lr_init

    # (.) Running model over the training data
    train_res = utils.train_epoch(epoch=epoch, loader=loaders['train'], model=model, criterion=criterion, optimizer=optimizer, device=device)
    # (.) Compute BN update before checking val performance
    utils.bn_update(loaders['train'], model)
    val_res = utils.eval(loaders['val'], model, criterion, device)
    test_res = utils.eval(loaders['test'], model, criterion, device)

    epoch_res = dict()
    epoch_res["Running"] = {"train": train_res, "val": val_res, "test": test_res}

    epoch_res["SWA"] = {"train": {"loss": "-", "accuracy": "-"}, "val": {"loss": "-", "accuracy": "-"},
                        "test": {"loss": "-", "accuracy": "-"}}
    epoch_res["ASWA"] = {"train": {"loss": "-", "accuracy": "-"}, "val": {"loss": "-", "accuracy": "-"},
                         "test": {"loss": "-", "accuracy": "-"}}

    if args.swa and (epoch + 1) >= args.swa_start and (epoch + 1 - args.swa_start) % args.swa_c_epochs == 0:
        # (1) SWA: Maintaing running average of model parameters
        utils.moving_average(swa_model, model, 1.0 / (swa_n + 1.0))
        swa_n += 1.0

        # (1.a) 若启用函数侧集成，按 stride 收集快照（CPU），不影响默认行为
        if args.do_func_ens:
            idx_since_start = (epoch + 1 - args.swa_start)
            if (idx_since_start % max(1, args.snapshot_stride)) == 0:
                swa_snapshots.append({k: v.detach().cpu().clone() for k, v in model.state_dict().items()})

        # (2) ASWA:
        if args.aswa:
            # (2.1) Lookahead
            utils.bn_update(loaders['train'], aswa_model)
            current_val = utils.eval(loaders['val'], aswa_model, criterion, device)

            # (2.2) Remember params
            current_aswa_state_dict = aswa_model.state_dict()
            aswa_state_dict = aswa_model.state_dict()

            # (2.3) Perform provisional param update on (2.2)
            for k, params in model.state_dict().items():
                aswa_state_dict[k] = (aswa_state_dict[k] * sum(aswa_ensemble_weights) + params) / (
                        1 + sum(aswa_ensemble_weights))

            # (2.4) Compute performance of updated ASWA ensemble
            aswa_model.load_state_dict(aswa_state_dict)
            utils.bn_update(loaders['train'], aswa_model)
            prov_val = utils.eval(loaders['val'], aswa_model, criterion, device)

            # (2.5) Decision
            if prov_val["accuracy"] >= current_val["accuracy"]:
                aswa_ensemble_weights.append(1.0)
            else:
                aswa_model.load_state_dict(current_aswa_state_dict)

    # Compute validation performances to report
    utils.bn_update(loaders['train'], swa_model)
    utils.bn_update(loaders['train'], aswa_model)

    epoch_res["SWA"] = {
        "train": utils.eval(loaders['train'], swa_model, criterion, device),
        "val": utils.eval(loaders['val'], swa_model, criterion, device),
        "test": utils.eval(loaders['test'], swa_model, criterion, device)}

    epoch_res["ASWA"] = {
        "train": utils.eval(loaders['train'], aswa_model, criterion, device),
        "val": utils.eval(loaders['val'], aswa_model, criterion, device),
        "test": utils.eval(loaders['test'], aswa_model, criterion, device)}

    time_ep = time.time() - time_ep

    columns = ["ep", "time", "lr", "train_loss", "val_acc","train_acc",
               "test_acc", "swa_train_acc", "swa_val_acc", "swa_test_acc",
               "aswa_train_acc", "aswa_val_acc", "aswa_test_acc"]
    values = [epoch + 1, time_ep, lr,
              epoch_res["Running"]["train"]["loss"],
              epoch_res["Running"]["train"]["accuracy"],
              epoch_res["Running"]["val"]["accuracy"],
              epoch_res["Running"]["test"]["accuracy"],
              epoch_res["SWA"]["train"]["accuracy"],
              epoch_res["SWA"]["val"]["accuracy"],
              epoch_res["SWA"]["test"]["accuracy"],
              epoch_res["ASWA"]["train"]["accuracy"],
              epoch_res["ASWA"]["val"]["accuracy"],
              epoch_res["ASWA"]["test"]["accuracy"]]

    df.append(values)
    table = tabulate.tabulate([values], columns, tablefmt='simple', floatfmt='4.4f')

    if epoch % 40 == 0:
        table = table.split('\n')
        table = '\n'.join([table[1]] + table)
    else:
        table = table.split('\n')[2]
    print(table)

if args.epochs % args.save_freq != 0:
    utils.save_checkpoint(
        args.dir,
        args.epochs,
        state_dict=model.state_dict(),
        swa_state_dict=swa_model.state_dict() if args.swa else None,
        swa_n=swa_n if args.swa else None,
        optimizer=optimizer.state_dict()
    )

df = pd.DataFrame(df, columns=columns)
df.to_csv(f"{args.dir}/results.csv")

# ---------------- Extra metrics (Acc/NLL/ECE) — 新增，仅打印，不影响训练 ----------------
@torch.no_grad()
def _eval_full_metrics(model_, loader_):
    model_.eval()
    probs_all, labels_all = [], []
    for x, y in loader_:
        x = x.to(device, non_blocking=True)
        logits = model_(x)
        probs = F.softmax(logits, dim=1).float().detach().cpu().numpy()
        probs_all.append(probs); labels_all.append(y.numpy())
    probs = np.concatenate(probs_all, axis=0)
    labels = np.concatenate(labels_all, axis=0)
    acc = float((probs.argmax(1) == labels).mean())
    nll = float(-np.log(np.clip(probs[np.arange(labels.size), labels], 1e-12, 1.0)).mean())
    # ECE (15 bins)
    bins = np.linspace(0,1,16)
    conf = probs.max(1)
    correct = (probs.argmax(1) == labels).astype(np.float32)
    ece = 0.0
    for i in range(15):
        lo, hi = bins[i], bins[i+1]
        m = (conf > lo) & (conf <= hi) if i>0 else (conf >= lo) & (conf <= hi)
        if m.any():
            ece += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return {'acc':acc, 'nll':nll, 'ece':ece}

utils.bn_update(loaders['train'], model)
print("Running model Train: ", utils.eval(loaders['train'], model, criterion, device))
print("Runing model Val:", utils.eval(loaders['val'], model, criterion, device))
print("Running model Test:", utils.eval(loaders['test'], model, criterion, device))
run_train = _eval_full_metrics(model, loaders['train'])
run_val   = _eval_full_metrics(model, loaders['val'])
run_test  = _eval_full_metrics(model, loaders['test'])
print(f"Running (Acc/NLL/ECE) Train: {run_train}")
print(f"Running (Acc/NLL/ECE) Val  : {run_val}")
print(f"Running (Acc/NLL/ECE) Test : {run_test}")

if args.swa:
    # utils.bn_update(loaders['train'], swa_model)
    utils.bn_update(train_noaug_loader, swa_model)
    print("SWA Train: ", utils.eval(loaders['train'], swa_model, criterion, device))
    print("SWA Val:", utils.eval(loaders['val'], swa_model, criterion, device))
    print("SWA Test:", utils.eval(loaders['test'], swa_model, criterion, device))
    swa_train = _eval_full_metrics(swa_model, loaders['train'])
    swa_val   = _eval_full_metrics(swa_model, loaders['val'])
    swa_test  = _eval_full_metrics(swa_model, loaders['test'])
    print(f"SWA (Acc/NLL/ECE) Train: {swa_train}")
    print(f"SWA (Acc/NLL/ECE) Val  : {swa_val}")
    print(f"SWA (Acc/NLL/ECE) Test : {swa_test}")

if args.aswa:
    # utils.bn_update(loaders['train'], aswa_model)
    utils.bn_update(train_noaug_loader, aswa_model)
    print("ASWA Train: ", utils.eval(loaders['train'], aswa_model, criterion, device))
    print("ASWA Val:", utils.eval(loaders['val'], aswa_model, criterion, device))
    print("ASWA Test:", utils.eval(loaders['test'], aswa_model, criterion, device))
    aswa_train = _eval_full_metrics(aswa_model, loaders['train'])
    aswa_val   = _eval_full_metrics(aswa_model, loaders['val'])
    aswa_test  = _eval_full_metrics(aswa_model, loaders['test'])
    print(f"ASWA (Acc/NLL/ECE) Train: {aswa_train}")
    print(f"ASWA (Acc/NLL/ECE) Val  : {aswa_val}")
    print(f"ASWA (Acc/NLL/ECE) Test : {aswa_test}")

# ---------------- Function-side ensembles (optional; 默认关闭，不影响原行为) ----------------
def _build_crit_subset_indices(ref_model, val_loader, frac):
    """基于 ref_model 的验证集 margin 选底部 frac 样本；frac∈(0,1]；返回索引或 None"""
    # utils.bn_update(loaders['train'], ref_model)
    utils.bn_update(train_noaug_loader, ref_model)  # 用 no-aug 训练集

  
    # 计算 margin
    ref_model.eval()
    logits_all, labels_all = [], []
    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(device, non_blocking=True)
            logits_all.append(ref_model(x).detach().cpu())
            labels_all.append(y.detach().cpu())
    logits = torch.cat(logits_all, 0)
    labels = torch.cat(labels_all, 0)
    true = logits[torch.arange(labels.size(0)), labels]
    logits_sc = logits.clone()
    logits_sc[torch.arange(labels.size(0)), labels] = -1e9
    other_max = logits_sc.max(dim=1).values
    margins = (true - other_max).numpy()
    frac = float(np.clip(frac, 0.0, 1.0))
    if frac <= 0 or frac >= 1: return None
    thr = np.quantile(margins, frac)
    return np.nonzero(margins <= thr)[0].astype(np.int64)

@torch.no_grad()
def _cache_val_probs_or_logprobs(snapshots, loader, build_model_fn, want_log=False):
    """按快照顺序逐个推理，返回 list[K]，每个 [N,C] 的 float32 numpy 数组。"""
    outs = []
    for sd in snapshots:
        # m = build_model_fn()
        # m.load_state_dict(sd, strict=True)
        # m.to(device).eval()
        m = build_model_fn()
        m.load_state_dict(sd, strict=True)
        m.to(device)
        utils.bn_update(train_noaug_loader, m)  # ★ 每个快照单独 BN 重估（no-aug）
        m.eval()
      
        chunks = []
        for x, _ in loader:
            x = x.to(device, non_blocking=True)
            out = m(x)
            arr = (F.log_softmax(out, dim=1) if want_log else F.softmax(out, dim=1)).float()
            chunks.append(arr.detach().cpu().numpy())
        outs.append(np.concatenate(chunks, axis=0))  # [N,C] float32
        del m
        torch.cuda.empty_cache()
    return outs

def _build_M_from_true_probs(P_list, y):
    """
    P_list: list of length K, each [N,C] prob on val for snapshot k
    y: numpy [N] true labels
    Return: M [K,K] (float64, PSD) built from centered true-class prob matrix F (K×N)
    """
    K = len(P_list)
    # F[k, n] = P_k(n, y_n)
    F = np.stack([Pi[np.arange(y.size), y] for Pi in P_list], axis=0).astype(np.float64)  # [K,N]
    F = F - F.mean(axis=1, keepdims=True)  # center each row
    N = max(1, F.shape[1])
    M = (F @ F.T) / float(N)               # covariance-like, PSD
    # 数值稳健：trace 归一到 K（可不做；保持量级稳定）
    tr = np.trace(M)
    if np.isfinite(tr) and tr > 1e-12:
        M = M * (K / tr)
    return M


def _learn_alpha_on_val_linear(snapshots, loader_val, build_model_fn, steps=150, lr=0.3, S_idx=None):
    """
    在 val(noaug) 上学习 α（线性概率混合目标）。
    - 不依赖外部缓存助手，逐快照评估，显存友好。
    - 全程 float64，避免 dtype 冲突。
    """
    device_ = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # 收集标签（与 loader_val 顺序严格对齐）
    labels = []
    for _, y in loader_val:
        labels.append(y.numpy())
    y = np.concatenate(labels, axis=0).astype(np.int64)
    N = y.shape[0]

    K = len(snapshots)
    assert K > 0, "No SWA snapshots to learn α from."

    # 逐快照缓存: P_list[k] 是 [N,C] 的概率矩阵（float64）
    P_list = []
    for k, sd in enumerate(snapshots):
        m = build_model_fn()
        m.load_state_dict(sd, strict=True)
        m.to(device_).eval()

        probs_chunks = []
        with torch.no_grad():
            for x, _ in loader_val:
                x = x.to(device_, non_blocking=True)
                p = torch.softmax(m(x), dim=1).double().cpu().numpy()  # -> float64
                probs_chunks.append(p)
        P_k = np.concatenate(probs_chunks, axis=0).astype(np.float64)
        assert P_k.shape[0] == N, "Val length mismatch while caching probs."
        P_list.append(P_k)

        # 及时释放
        del m
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # 组装为 [K,N,C]
    C = P_list[0].shape[1]
    P = np.stack(P_list, axis=0).astype(np.float64)  # [K,N,C]

    # 相关性矩阵 M（K×K）与均匀向量 u（与 WRN 版一致）
    # 假定你已实现 _build_M_from_true_probs(P_list, y) -> [K,K]
    M_np = _build_M_from_true_probs(P_list, y)             # [K,K], float64
    u_np = np.ones(K, dtype=np.float64) / float(max(1, K))

    # 转 torch（float64, CPU）
    phi = torch.zeros(K, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.Adam([phi], lr=lr)

    y_t  = torch.from_numpy(y).long()
    P_t  = torch.from_numpy(P).to(torch.float64)           # [K,N,C]
    M_t  = torch.from_numpy(M_np).to(torch.float64)        # [K,K]
    u_t  = torch.from_numpy(u_np).to(torch.float64)        # [K]

    if S_idx is not None:
        if isinstance(S_idx, np.ndarray):
            idx_t = torch.from_numpy(S_idx).long()
        elif isinstance(S_idx, torch.Tensor):
            idx_t = S_idx.long().cpu()
        else:
            idx_t = torch.tensor(S_idx, dtype=torch.long)
    else:
        idx_t = None

    lam = float(getattr(args, 'func_w_l2', 1e-3))
    eta = float(getattr(args, 'func_w_eta', 1e-2))

    for _ in range(1, steps + 1):
        opt.zero_grad()
        w = torch.softmax(phi, dim=0)                          # [K], float64
        mix = torch.einsum('k,knc->nc', w, P_t)                # [N,C], float64

        if idx_t is not None:
            mix_sel = mix.index_select(0, idx_t)
            y_sel   = y_t.index_select(0, idx_t)
            ce = F.nll_loss(torch.log(mix_sel.clamp_min(1e-12)).to(torch.float64), y_sel)
        else:
            ce = F.nll_loss(torch.log(mix.clamp_min(1e-12)).to(torch.float64), y_t)

        reg_l2 = lam * torch.sum((w - u_t) * (w - u_t))
        quad   = eta * torch.sum(w * (M_t @ w))
        loss = ce + reg_l2 + quad
        loss.backward()
        opt.step()

    with torch.no_grad():
        w = torch.softmax(phi, dim=0).cpu().numpy().astype(np.float64)
    return w


def _learn_alpha_on_val_logpool(snapshots, loader_val, build_model_fn, steps=150, lr=0.3, S_idx=None):
    """
    在 val(noaug) 上学习 α（LogPool 目标：对数概率的加权和后再 softmax）。
    - 同样逐快照缓存，显存友好。
    - 全程 float64。
    """
    device_ = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # 收集标签
    labels = []
    for _, y in loader_val:
        labels.append(y.numpy())
    y = np.concatenate(labels, axis=0).astype(np.int64)
    N = y.shape[0]

    K = len(snapshots)
    assert K > 0, "No SWA snapshots to learn α from."

    # 逐快照缓存：log-probs 与 probs（logpool 用 logp，相关性矩阵更稳用概率）
    L_list, P_list = [], []
    for k, sd in enumerate(snapshots):
        m = build_model_fn()
        m.load_state_dict(sd, strict=True)
        m.to(device_).eval()

        logp_chunks, probs_chunks = [], []
        with torch.no_grad():
            for x, _ in loader_val:
                x = x.to(device_, non_blocking=True)
                out = m(x)
                logp = torch.log_softmax(out, dim=1).double().cpu().numpy()
                p    = torch.softmax(out, dim=1).double().cpu().numpy()
                logp_chunks.append(logp)
                probs_chunks.append(p)
        L_k = np.concatenate(logp_chunks, axis=0).astype(np.float64)  # [N,C]
        P_k = np.concatenate(probs_chunks, axis=0).astype(np.float64)  # [N,C]
        assert L_k.shape[0] == N and P_k.shape[0] == N
        L_list.append(L_k)
        P_list.append(P_k)

        del m
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # 组装为 [K,N,C]
    C = L_list[0].shape[1]
    L = np.stack(L_list, axis=0).astype(np.float64)  # [K,N,C]
    # 相关性矩阵（基于概率）
    M_np = _build_M_from_true_probs(P_list, y)       # [K,K]
    u_np = np.ones(K, dtype=np.float64) / float(max(1, K))

    # 转 torch（float64, CPU）
    phi = torch.zeros(K, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.Adam([phi], lr=lr)

    y_t  = torch.from_numpy(y).long()
    L_t  = torch.from_numpy(L).to(torch.float64)     # [K,N,C]
    M_t  = torch.from_numpy(M_np).to(torch.float64)
    u_t  = torch.from_numpy(u_np).to(torch.float64)

    if S_idx is not None:
        if isinstance(S_idx, np.ndarray):
            idx_t = torch.from_numpy(S_idx).long()
        elif isinstance(S_idx, torch.Tensor):
            idx_t = S_idx.long().cpu()
        else:
            idx_t = torch.tensor(S_idx, dtype=torch.long)
    else:
        idx_t = None

    lam = float(getattr(args, 'func_w_l2', 1e-3))
    eta = float(getattr(args, 'func_w_eta', 1e-2))

    for _ in range(1, steps + 1):
        opt.zero_grad()
        w = torch.softmax(phi, dim=0)                               # [K]
        log_mix = torch.einsum('k,knc->nc', w, L_t)                 # [N,C]

        if idx_t is not None:
            log_sel = log_mix.index_select(0, idx_t)
            y_sel   = y_t.index_select(0, idx_t)
            ce = F.nll_loss(torch.log_softmax(log_sel, dim=1).to(torch.float64), y_sel)
        else:
            ce = F.nll_loss(torch.log_softmax(log_mix, dim=1).to(torch.float64), y_t)

        reg_l2 = lam * torch.sum((w - u_t) * (w - u_t))
        quad   = eta * torch.sum(w * (M_t @ w))
        loss = ce + reg_l2 + quad
        loss.backward()
        opt.step()

    with torch.no_grad():
        w = torch.softmax(phi, dim=0).cpu().numpy().astype(np.float64)
    return w



def _load_or_uniform_weights(K):
    w = None
    if args.func_weights is not None and os.path.isfile(args.func_weights):
        if args.func_weights.endswith('.npy'):
            w = np.load(args.func_weights)
        elif args.func_weights.endswith('.json'):
            with open(args.func_weights,'r') as f:
                w = np.array(json.load(f), dtype=np.float32)
    if w is None:
        w = np.ones(K, dtype=np.float32) / float(max(1, K))
    w = np.asarray(w, dtype=np.float32).reshape(-1)
    assert w.size == K, f"weights length {w.size} != #snapshots {K}"
    s = w.sum()
    if not np.isfinite(s) or s <= 0:
        w = np.ones(K, dtype=np.float32) / float(max(1, K))
    else:
        w = w / s
    return w

@torch.no_grad()
def _eval_function_ensemble(snapshots, weights, loader, build_model_fn, mode='linear'):
    """在测试集上评估函数侧集成，返回 dict(acc,nll,ece)。"""
    K = len(snapshots)
    weights = np.asarray(weights, dtype=np.float32)
    probs_all, labels_all = [], []
    # 逐快照推理并累计
    mix_probs = None
    for j in range(K):
        # m = build_model_fn()
        # m.load_state_dict(snapshots[j], strict=True)
        # m.to(device).eval()
        m = build_model_fn()
        m.load_state_dict(snapshots[j], strict=True)
        m.to(device)
        utils.bn_update(train_noaug_loader, m)  # ★
        m.eval()

      
        chunk_probs = []
        for x, _ in loader:
            x = x.to(device, non_blocking=True)
            if mode == 'logpool':
                logp = F.log_softmax(m(x), dim=1).float()
                chunk_probs.append(logp.detach().cpu().numpy())
            else:
                p = F.softmax(m(x), dim=1).float()
                chunk_probs.append(p.detach().cpu().numpy())
        chunk = np.concatenate(chunk_probs, axis=0)  # [N,C]
        if mode == 'logpool':
            contrib = weights[j] * chunk  # log 概率的加权求和
            mix_probs = contrib if mix_probs is None else (mix_probs + contrib)
        else:
            contrib = weights[j] * chunk
            mix_probs = contrib if mix_probs is None else (mix_probs + contrib)
        del m
        torch.cuda.empty_cache()

    if mode == 'logpool':
        probs = torch.softmax(torch.from_numpy(mix_probs), dim=1).numpy().astype(np.float32)
    else:
        probs = mix_probs.astype(np.float32)

    # 收集标签一次即可
    for _, y in loader: labels_all.append(y.numpy())
    labels = np.concatenate(labels_all, axis=0)

    acc = float((probs.argmax(1) == labels).mean())
    nll = float(-np.log(np.clip(probs[np.arange(labels.size), labels], 1e-12, 1.0)).mean())
    bins = np.linspace(0,1,16); conf = probs.max(1); correct = (probs.argmax(1)==labels).astype(np.float32)
    ece=0.0
    for i in range(15):
        lo,hi=bins[i],bins[i+1]
        m = (conf>lo)&(conf<=hi) if i>0 else (conf>=lo)&(conf<=hi)
        if m.any(): ece += m.mean()*abs(correct[m].mean()-conf[m].mean())
    return {'acc':acc,'nll':nll,'ece':ece}

if args.do_func_ens and len(swa_snapshots) > 0:
    # 基于 SWA（若启用，否则用 running）在 val 上选 critical 子集
    ref_for_margin = swa_model if args.swa else model
    S_idx = _build_crit_subset_indices(ref_for_margin, val_noaug_loader, args.crit_fraction)

    if S_idx is None:
        print(f"[α-learn] Using FULL val set (crit_fraction={args.crit_fraction:.2f} ignored).")
    else:
        print(f"[α-learn] Using critical subset |S|={len(S_idx)} / N={len(val_set)} (frac={args.crit_fraction:.2f}).")

    # 选择/学习 α
    def _bm():  # build model
        return build_model()

    if args.learn_func_w:
        if args.func_w_type == 'linear':
            alpha = _learn_alpha_on_val_linear(swa_snapshots, val_noaug_loader, _bm,
                                               steps=args.func_w_steps, lr=args.func_w_lr, S_idx=S_idx)
        else:
            alpha = _learn_alpha_on_val_logpool(swa_snapshots, val_noaug_loader, _bm,
                                                steps=args.func_w_steps, lr=args.func_w_lr, S_idx=S_idx)
        print("[Func-Ensemble] learned α (sum=1):", np.round(alpha, 4).tolist())
    else:
        alpha = _load_or_uniform_weights(len(swa_snapshots))
        print("[Func-Ensemble] use weights:", np.round(alpha, 4).tolist())

    # 测试集评测
    if args.func_type in ('linear','both'):
        res_lin = _eval_function_ensemble(swa_snapshots, alpha, loaders['test'], _bm, mode='linear')
        print(f"[Func-Ensemble Linear] Test: Acc={res_lin['acc']:.4f} | NLL={res_lin['nll']:.4f} | ECE={res_lin['ece']:.4f}")
    if args.func_type in ('logpool','both'):
        res_log = _eval_function_ensemble(swa_snapshots, alpha, loaders['test'], _bm, mode='logpool')
        print(f"[Func-Ensemble LogPool]  Test: Acc={res_log['acc']:.4f} | NLL={res_log['nll']:.4f} | ECE={res_log['ece']:.4f}")
elif args.do_func_ens:
    print("[Func-Ensemble] No SWA snapshots were collected; nothing to evaluate.")
