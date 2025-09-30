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

# ---------- 函数侧集成与 α 学习（已替换为“第一版”逻辑） ----------
parser.add_argument('--do_func_ens', action='store_true',
                    help='enable function-side ensemble evaluation on SWA snapshots (default: off)')
parser.add_argument('--func_type', type=str, default='both', choices=['linear', 'logpool', 'both'],
                    help='function-side ensemble type when enabled')
parser.add_argument('--func_weights', type=str, default=None,
                    help='optional path to weights (json/npy) for function ensemble; default uniform')
parser.add_argument('--learn_func_w', action='store_true',
                    help='learn function-ensemble weights α on validation (default: off)')
# func_w_type 参数保留但在“第一版”路径中不再区分，统一在 logits 空间学习 A，再转为 ᾱ
parser.add_argument('--func_w_type', type=str, default='linear', choices=['linear','logpool'],
                    help='(kept for compatibility; ignored by logits-space learning)')
parser.add_argument('--func_w_steps', type=int, default=150, help='steps to optimize α/A (default 150)')
parser.add_argument('--func_w_lr', type=float, default=0.3, help='lr to optimize α/A (default 0.3)')
parser.add_argument('--crit_fraction', type=float, default=0.5,
                    help='fraction of lowest-margin val samples used for α/A learning (default 0.5)')
parser.add_argument('--snapshot_stride', type=int, default=1,
                    help='keep every n-th SWA snapshot during SWA stage (>=1); only used if --do_func_ens')

parser.add_argument('--func_w_l2', type=float, default=1e-3,
                    help='L2 regularization coeff λ towards uniform (default 1e-3)')
parser.add_argument('--func_w_eta', type=float, default=1e-2,
                    help='HAC correlation coeff η (default 1e-2)')

# ★ 类条件权重（K×C）；学习时总是用类条件矩阵 A，再取列均值为 ᾱ
parser.add_argument('--func_class_cond', action='store_true',
                    help='kept for compatibility; final fusion uses alpha_bar regardless')

parser.add_argument('--func_topk', type=int, default=8,
    help='per-class top-k projection for A[K,C]; 0 disables (default: 8)')


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
assert hasattr(val_set, 'indices'), "val_set must be a torch.utils.data.Subset"
val_indices = val_set.indices

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
    utils.bn_update(loaders['train'], swa_model)
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
    utils.bn_update(loaders['train'], aswa_model)
    print("ASWA Train: ", utils.eval(loaders['train'], aswa_model, criterion, device))
    print("ASWA Val:", utils.eval(loaders['val'], aswa_model, criterion, device))
    print("ASWA Test:", utils.eval(loaders['test'], aswa_model, criterion, device))
    aswa_train = _eval_full_metrics(aswa_model, loaders['train'])
    aswa_val   = _eval_full_metrics(aswa_model, loaders['val'])
    aswa_test  = _eval_full_metrics(aswa_model, loaders['test'])
    print(f"ASWA (Acc/NLL/ECE) Train: {aswa_train}")
    print(f"ASWA (Acc/NLL/ECE) Val  : {aswa_val}")
    print(f"ASWA (Acc/NLL/ECE) Test : {aswa_test}")

# ===================== 第一版：logits 空间学习 A -> 预测空间用 ᾱ 融合 =====================

@torch.no_grad()
def _cache_val_logits_no_bn(snapshots, loader, build_model_fn, device_):
    """缓存每个快照在 val_noaug 上的 logits（不做 BN 更新），返回 list[K]，每个 [N,C] float64。"""
    outs = []
    for sd in snapshots:
        m = build_model_fn()
        m.load_state_dict(sd, strict=True)
        m.to(device_).eval()
        chunks = []
        for x, _ in loader:
            x = x.to(device_, non_blocking=True)
            logits = m(x).double().cpu().numpy()
            chunks.append(logits)
        outs.append(np.concatenate(chunks, axis=0).astype(np.float64))
        del m
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return outs  # list of [N,C]

def _build_M_from_true_probs_no_bn(snapshots, loader_val, build_model_fn, device_, y_vec):
    """用真类概率（不 BN 更新）构造 HAC 相关性矩阵 M[K,K]（PSD）。"""
    K = len(snapshots)
    N = y_vec.size
    F = np.zeros((K, N), dtype=np.float64)  # [K,N]
    with torch.no_grad():
        for k, sd in enumerate(snapshots):
            m = build_model_fn(); m.load_state_dict(sd, strict=True); m.to(device_).eval()
            idx0 = 0
            for x, y in loader_val:
                x = x.to(device_, non_blocking=True)
                p = F_softmax(m(x), dim=1) if False else None  # 占位，下一行直接算
                logits = m(x)
                probs = F.softmax(logits, dim=1).double().cpu().numpy()
                y_np = y.numpy()
                F[k, idx0:idx0+len(y_np)] = probs[np.arange(len(y_np)), y_np]
                idx0 += len(y_np)
            del m
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    # 中心化并构建协方差型矩阵
    Fc = F - F.mean(axis=1, keepdims=True)
    M = (Fc @ Fc.T) / max(1, Fc.shape[1])
    # 归一/PSD 投影（简单稳健版）
    # PSD 投影
    w, V = np.linalg.eigh(M)
    w = np.maximum(w, 1e-9)
    M_psd = (V * w) @ V.T
    # 迹归一（可选）
    tr = np.trace(M_psd)
    if np.isfinite(tr) and tr > 1e-12:
        M_psd = M_psd * (K / tr)
    return M_psd  # [K,K] float64

def _project_topk_per_class(A: np.ndarray, k_top: int) -> np.ndarray:
    if not isinstance(A, np.ndarray):
        A = np.asarray(A)
    assert A.ndim == 2, "A must be KxC"
    K, C = A.shape
    if k_top is None or k_top <= 0 or k_top >= K:
        return A
    out = np.zeros_like(A, dtype=np.float64)
    for c in range(C):
        col = A[:, c]
        keep = np.argsort(col)[::-1][:k_top]
        s = float(col[keep].sum())
        if not np.isfinite(s) or s <= 1e-12:
            out[:, c] = 1.0 / K
        else:
            out[keep, c] = col[keep] / s
    return out

def _build_crit_subset_indices(ref_model, val_loader, frac):
    """与原版一致：用 ref_model 的 margin 选困难子集；这里沿用现有实现。"""
    utils.bn_update(loaders['train'], ref_model)
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

def _learn_A_from_logits(L_list, y_vec, S_idx, M_np, steps=150, lr=0.3, lam=1e-3, eta=1e-2, topk=8):
    """
    第一版：在 logits 空间学习类条件矩阵 A[K,C]（每列 softmax 归一），
    目标：CE( sum_k A[k,c]*L[k,n,c], y ) + lam*||A-U||^2 + eta*tr(A^T M A)
    然后做 per-class top-k 投影并返回。
    """
    K = len(L_list)
    N, C = L_list[0].shape
    assert all(li.shape == (N, C) for li in L_list)
    L = np.stack(L_list, axis=0).astype(np.float64)  # [K,N,C]

    y_t  = torch.from_numpy(y_vec).long()
    L_t  = torch.from_numpy(L).to(torch.float64)     # [K,N,C]
    M_t  = torch.from_numpy(M_np).to(torch.float64)  # [K,K]
    U    = torch.full((K, C), 1.0/float(K), dtype=torch.float64)

    if S_idx is not None:
        idx_t = torch.from_numpy(np.asarray(S_idx, dtype=np.int64))
    else:
        idx_t = None

    phi = torch.zeros(K, C, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.Adam([phi], lr=lr)

    for _ in range(1, steps+1):
        opt.zero_grad()
        A = torch.softmax(phi, dim=0)                        # [K,C]（每列对 K softmax）
        # logits 融合：log_mix[n,c] = sum_k A[k,c]*L[k,n,c]
        log_mix = torch.einsum('kc,knc->nc', A, L_t)         # [N,C]
        if idx_t is not None:
            ce = F.cross_entropy(log_mix.index_select(0, idx_t), y_t.index_select(0, idx_t))
        else:
            ce = F.cross_entropy(log_mix, y_t)
        reg_l2 = lam * torch.sum((A - U) * (A - U))
        quad   = eta * torch.sum(A * (M_t @ A))              # tr(A^T M A)
        loss = ce + reg_l2 + quad
        loss.backward()
        opt.step()

    with torch.no_grad():
        A = torch.softmax(phi, dim=0).cpu().numpy().astype(np.float64)

    if topk is not None and topk > 0:
        A = _project_topk_per_class(A, int(topk))
    return A  # [K,C]

@torch.no_grad()
def _eval_function_ensemble(snapshots, weights, loader, build_model_fn, mode='linear'):
    """
    第一版评估：不做 BN 更新；weights 支持 α[K] 或 A[K,C]（若提供 A 会逐类乘）。
    """
    K = len(snapshots)
    weights = np.asarray(weights)
    class_cond = (weights.ndim == 2)
    mix_accum = None
    labels_all = []

    for j in range(K):
        m = build_model_fn()
        m.load_state_dict(snapshots[j], strict=True)
        m.to(device).eval()  # 不做 bn_update（与第一版一致）

        chunk_list = []
        for x, _ in loader:
            x = x.to(device, non_blocking=True)
            if mode == 'logpool':
                arr = F.log_softmax(m(x), dim=1).double().cpu().numpy()
            else:
                arr = F.softmax(m(x), dim=1).double().cpu().numpy()
            chunk_list.append(arr)
        pred = np.concatenate(chunk_list, axis=0)  # [N,C] float64

        if mix_accum is None:
            if class_cond:
                mix_accum = pred * weights[j].reshape(1, -1)
            else:
                mix_accum = pred * float(weights[j])
        else:
            if class_cond:
                mix_accum += pred * weights[j].reshape(1, -1)
            else:
                mix_accum += pred * float(weights[j])

        del m
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    for _, y in loader:
        labels_all.append(y.numpy())
    labels = np.concatenate(labels_all, axis=0)

    if mode == 'logpool':
        probs = torch.softmax(torch.from_numpy(mix_accum), dim=1).numpy().astype(np.float32)
    else:
        probs = mix_accum.astype(np.float32)

    preds = probs.argmax(1)
    acc = float((preds == labels).mean())
    p_true = np.clip(probs[np.arange(labels.size), labels], 1e-12, 1.0)
    nll = float(-np.log(p_true).mean())
    bins = np.linspace(0,1,16); conf = probs.max(1); correct = (preds==labels).astype(np.float32)
    ece=0.0
    for i in range(15):
        lo,hi=bins[i],bins[i+1]
        m = (conf>lo)&(conf<=hi) if i>0 else (conf>=lo)&(conf<=hi)
        if m.any(): ece += m.mean()*abs(correct[m].mean()-conf[m].mean())
    return {'acc':acc,'nll':nll,'ece':ece}

# ------------------------------ 主：函数侧集成（第一版） ------------------------------
if args.do_func_ens and len(swa_snapshots) > 0:
    # 1) 困难子集（沿用原 margin 选择）
    ref_for_margin = swa_model if args.swa else model
    S_idx = _build_crit_subset_indices(ref_for_margin, val_noaug_loader, args.crit_fraction)
    if S_idx is None:
        print(f"[α/A-learn] Using FULL val set (crit_fraction={args.crit_fraction:.2f} ignored).")
    else:
        print(f"[α/A-learn] Using critical subset |S|={len(S_idx)} / N={len(val_set)} (frac={args.crit_fraction:.2f}).")

    # 2) 准备标签（val_noaug）
    y_list = []
    for _, y in val_noaug_loader:
        y_list.append(y.numpy())
    y_val = np.concatenate(y_list, axis=0).astype(np.int64)

    # 3) 构造 HAC 相关性矩阵 M（用真类概率；不 BN 更新）
    def _bm(): return build_model()
    M_np = _build_M_from_true_probs_no_bn(swa_snapshots, val_noaug_loader, _bm, device, y_val)

    # 4) 缓存每个快照的 logits（不 BN 更新）
    L_list = _cache_val_logits_no_bn(swa_snapshots, val_noaug_loader, _bm, device)  # list[K] each [N,C]

    # 5) 学 A[K,C]（logits 空间）→ top-k → ᾱ
    if args.learn_func_w:
        A = _learn_A_from_logits(
            L_list, y_val, S_idx, M_np,
            steps=args.func_w_steps, lr=args.func_w_lr,
            lam=args.func_w_l2, eta=args.func_w_eta,
            topk=args.func_topk if args.func_topk > 0 else 0
        )
        alpha_bar = A.mean(axis=1).astype(np.float64)  # [K]
        s = alpha_bar.sum()
        alpha_bar = (np.ones_like(alpha_bar)/len(alpha_bar)) if (not np.isfinite(s) or s<=1e-12) else (alpha_bar/s)
        print("[Func-Ensemble] learned A[K,C] via logits; alpha_bar preview:",
              np.round(alpha_bar[:min(10, alpha_bar.size)], 4).tolist())
    else:
        alpha_bar = np.ones(len(swa_snapshots), dtype=np.float64)/float(len(swa_snapshots))
        print("[Func-Ensemble] use uniform alpha_bar (no learning).")

    # 6) 测试集评测（linear / logpool）
    if args.func_type in ('linear','both'):
        res_lin = _eval_function_ensemble(swa_snapshots, alpha_bar, loaders['test'], _bm, mode='linear')
        print(f"[Func-Ensemble Linear] Test: Acc={res_lin['acc']:.4f} | NLL={res_lin['nll']:.4f} | ECE={res_lin['ece']:.4f}")
    if args.func_type in ('logpool','both'):
        res_log = _eval_function_ensemble(swa_snapshots, alpha_bar, loaders['test'], _bm, mode='logpool')
        print(f"[Func-Ensemble LogPool]  Test: Acc={res_log['acc']:.4f} | NLL={res_log['nll']:.4f} | ECE={res_log['ece']:.4f}")
elif args.do_func_ens:
    print("[Func-Ensemble] No SWA snapshots were collected; nothing to evaluate.")
