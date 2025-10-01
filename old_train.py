# train.py
import argparse
import os
import sys
import time
import json
import copy
import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import models
import utils
import tabulate
from torch.utils.data import random_split
import pandas as pd

# ------------------------- Argparse -------------------------
parser = argparse.ArgumentParser(description='SWA-ASWA training')

parser.add_argument('--dir', type=str, default='.', required=False, help='training directory')
parser.add_argument('--dataset', type=str, default='CIFAR10', help='CIFAR10 or CIFAR100')
parser.add_argument('--batch_size', type=int, default=128, help='batch size')
parser.add_argument('--num_workers', type=int, default=16, help='num workers')
parser.add_argument('--model', type=str, default='VGG16', help='model name (must exist in models)')
parser.add_argument('--optim', type=str, default='SGD', help='SGD or Adam')
parser.add_argument('--epochs', type=int, default=200, help='epochs')
parser.add_argument('--save_freq', type=int, default=25, help='save frequency')
parser.add_argument('--eval_freq', type=int, default=1, help='eval frequency')
parser.add_argument('--lr_init', type=float, default=0.1, help='initial LR')
parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum')
parser.add_argument('--wd', type=float, default=1e-4, help='weight decay')

parser.add_argument('--swa', action='store_true', help='enable SWA')
parser.add_argument('--aswa', action='store_true', help='enable ASWA')
parser.add_argument('--swa_start', type=int, default=161, help='SWA start epoch')
parser.add_argument('--swa_lr', type=float, default=0.05, help='SWA LR')
parser.add_argument('--swa_c_epochs', type=int, default=1, help='SWA cycle length in epochs')

parser.add_argument('--seed', type=int, default=1, help='random seed')
parser.add_argument('--val_ratio', type=float, default=0.1, help='validation ratio in train set')

# ---------- FS-Alpha：函数侧/权重学习 ----------
parser.add_argument('--do_func_ens', action='store_true',
                    help='enable function-side ensemble evaluation on SWA snapshots')
parser.add_argument('--func_type', type=str, default='both', choices=['linear', 'logpool', 'both'],
                    help='function-side ensemble type')
parser.add_argument('--func_weights', type=str, default=None,
                    help='optional path to weights (json/npy) for function ensemble; default uniform')
parser.add_argument('--learn_func_w', action='store_true',
                    help='learn function-ensemble weights α on validation (HeadOnly/Wb space)')
parser.add_argument('--func_w_steps', type=int, default=150, help='steps to optimize α/A')
parser.add_argument('--func_w_lr', type=float, default=0.3, help='lr to optimize α/A')
parser.add_argument('--crit_fraction', type=float, default=0.5,
                    help='fraction of lowest-margin val samples for α/A learning')
parser.add_argument('--snapshot_stride', type=int, default=1,
                    help='keep every n-th SWA snapshot during SWA stage (>=1)')

parser.add_argument('--func_w_l2', type=float, default=1e-3,
                    help='L2 regularization coeff λ towards uniform')
parser.add_argument('--func_w_eta', type=float, default=1e-2,
                    help='HAC correlation coeff η')
parser.add_argument('--func_topk', type=int, default=8,
                    help='per-class top-k projection for A[K,C]; 0 disables')

args = parser.parse_args()

# ------------------------- Setup -------------------------
print('Preparing directory %s' % args.dir)
os.makedirs(args.dir, exist_ok=True)

torch.backends.cudnn.benchmark = True
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)

print('Using model %s' % args.model)
model_cfg = getattr(models, args.model)

# ------------------------- Dataset -------------------------
print('Loading dataset:', args.dataset)
assert 1.0 > float(args.val_ratio) > 0.0
if args.dataset.upper() == "CIFAR10":
    train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True,
                                             transform=model_cfg.transform_train)
    test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True,
                                            transform=model_cfg.transform_test)
elif args.dataset.upper() == "CIFAR100":
    train_set = torchvision.datasets.CIFAR100(root='./data', train=True, download=True,
                                              transform=model_cfg.transform_train)
    test_set = torchvision.datasets.CIFAR100(root='./data', train=False, download=True,
                                             transform=model_cfg.transform_test)
else:
    print("Incorrect dataset", args.dataset); sys.exit(1)

# Split train/val
train_size = int(len(train_set) * (1 - float(args.val_ratio)))
val_size = len(train_set) - train_size
num_classes = (max(train_set.targets) + 1) if hasattr(train_set, 'targets') else len(train_set.classes)
train_set, val_set = random_split(train_set, [train_size, val_size],
                                  generator=torch.Generator().manual_seed(args.seed))

print(f"|Train|:{len(train_set)} |Val|:{len(val_set)} |Test|:{len(test_set)}")
loaders = {
    'train': torch.utils.data.DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True
    ),
    'val': torch.utils.data.DataLoader(
        val_set, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True
    ),
    'test': torch.utils.data.DataLoader(
        test_set, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True
    )
}

# ------------------------- Models -------------------------
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

model.to(device); swa_model.to(device); aswa_model.to(device)

# 仅在需要做函数侧集成时，才收集快照，避免影响默认行为与内存
swa_snapshots = []  # list of CPU state_dict

# ------------------------- LR schedule -------------------------
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
if args.optim.upper() == "SGD":
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr_init, momentum=args.momentum, weight_decay=args.wd)
elif args.optim.upper() == "ADAM":
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr_init, weight_decay=args.wd)
else:
    print("Unsupported optim:", args.optim); sys.exit(1)

# ------------------------- Train Loop -------------------------
start_epoch = 0
df = []
for epoch in range(start_epoch, args.epochs):
    time_ep = time.time()
    # Adjust LR
    if args.optim.upper() == "SGD" and epoch > 0:
        lr = schedule(epoch)
        utils.adjust_learning_rate(optimizer, lr)
    else:
        lr = args.lr_init

    # Train one epoch
    train_res = utils.train_epoch(epoch=epoch, loader=loaders['train'], model=model,
                                  criterion=criterion, optimizer=optimizer, device=device)

    # BN update before eval (保持源码风格：用 train loader 上的增广分布)
    utils.bn_update(loaders['train'], model)
    val_res = utils.eval(loaders['val'], model, criterion, device)
    test_res = utils.eval(loaders['test'], model, criterion, device)

    epoch_res = dict()
    epoch_res["Running"] = {"train": train_res, "val": val_res, "test": test_res}
    epoch_res["SWA"] = {"train": {"loss": "-", "accuracy": "-"},
                        "val": {"loss": "-", "accuracy": "-"},
                        "test": {"loss": "-", "accuracy": "-"}}
    epoch_res["ASWA"] = {"train": {"loss": "-", "accuracy": "-"},
                         "val": {"loss": "-", "accuracy": "-"},
                         "test": {"loss": "-", "accuracy": "-"}}

    # SWA/ASWA steps
    if args.swa and (epoch + 1) >= args.swa_start and (epoch + 1 - args.swa_start) % args.swa_c_epochs == 0:
        # SWA running average
        utils.moving_average(swa_model, model, 1.0 / (swa_n + 1.0))
        swa_n += 1.0

        # 收集快照（仅当启用函数侧集成）
        if args.do_func_ens:
            idx_since_start = (epoch + 1 - args.swa_start)
            if (idx_since_start % max(1, args.snapshot_stride)) == 0:
                swa_snapshots.append({k: v.detach().cpu().clone() for k, v in model.state_dict().items()})

        # ASWA
        if args.aswa:
            utils.bn_update(loaders['train'], aswa_model)
            current_val = utils.eval(loaders['val'], aswa_model, criterion, device)

            current_aswa_state_dict = aswa_model.state_dict()
            aswa_state_dict = aswa_model.state_dict()
            for k, params in model.state_dict().items():
                aswa_state_dict[k] = (aswa_state_dict[k] * sum(aswa_ensemble_weights) + params) / (
                        1 + sum(aswa_ensemble_weights))
            aswa_model.load_state_dict(aswa_state_dict)
            utils.bn_update(loaders['train'], aswa_model)
            prov_val = utils.eval(loaders['val'], aswa_model, criterion, device)
            if prov_val["accuracy"] >= current_val["accuracy"]:
                aswa_ensemble_weights.append(1.0)
            else:
                aswa_model.load_state_dict(current_aswa_state_dict)

    # eval SWA/ASWA
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

    # 注意：columns 顺序与 values 对齐
    columns = ["ep", "time", "lr", "train_loss", "train_acc", "val_acc",
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
        table = table.split('\n'); table = '\n'.join([table[1]] + table)
    else:
        table = table.split('\n')[2]
    print(table)

# Save last
if args.epochs % args.save_freq != 0:
    utils.save_checkpoint(
        args.dir, args.epochs,
        state_dict=model.state_dict(),
        swa_state_dict=swa_model.state_dict() if args.swa else None,
        swa_n=swa_n if args.swa else None,
        optimizer=optimizer.state_dict()
    )

df = pd.DataFrame(df, columns=columns)
df.to_csv(f"{args.dir}/results.csv", index=False)

# ---------------- Extra metrics (Acc/NLL/ECE) — 仅打印 ----------------
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
print("Running (Acc/NLL/ECE) Train:", _eval_full_metrics(model, loaders['train']))
print("Running (Acc/NLL/ECE) Val  :", _eval_full_metrics(model, loaders['val']))
print("Running (Acc/NLL/ECE) Test :", _eval_full_metrics(model, loaders['test']))

if args.swa:
    utils.bn_update(loaders['train'], swa_model)
    print("SWA Train: ", utils.eval(loaders['train'], swa_model, criterion, device))
    print("SWA Val:", utils.eval(loaders['val'], swa_model, criterion, device))
    print("SWA Test:", utils.eval(loaders['test'], swa_model, criterion, device))
    print("SWA (Acc/NLL/ECE) Train:", _eval_full_metrics(swa_model, loaders['train']))
    print("SWA (Acc/NLL/ECE) Val  :", _eval_full_metrics(swa_model, loaders['val']))
    print("SWA (Acc/NLL/ECE) Test :", _eval_full_metrics(swa_model, loaders['test']))

if args.aswa:
    utils.bn_update(loaders['train'], aswa_model)
    print("ASWA Train: ", utils.eval(loaders['train'], aswa_model, criterion, device))
    print("ASWA Val:", utils.eval(loaders['val'], aswa_model, criterion, device))
    print("ASWA Test:", utils.eval(loaders['test'], aswa_model, criterion, device))
    print("ASWA (Acc/NLL/ECE) Train:", _eval_full_metrics(aswa_model, loaders['train']))
    print("ASWA (Acc/NLL/ECE) Val  :", _eval_full_metrics(aswa_model, loaders['val']))
    print("ASWA (Acc/NLL/ECE) Test :", _eval_full_metrics(aswa_model, loaders['test']))

# ===================== FS-Alpha (W,b 空间 + aug-val + HAC(QS)) =====================

EVAL_AUG_SEED = 12345  # 保证不同快照看到相同增广

def _fix_aug_rng():
    random.seed(EVAL_AUG_SEED); np.random.seed(EVAL_AUG_SEED); torch.manual_seed(EVAL_AUG_SEED)

def _find_last_linear(model: nn.Module):
    last_name, last_mod = None, None
    for name, m in model.named_modules():
        if isinstance(m, nn.Linear):
            last_name, last_mod = name, m
    if last_mod is None:
        raise RuntimeError("No nn.Linear layer found as classifier head.")
    return last_name, last_mod

def _get_module_by_name(model: nn.Module, dotted: str):
    m = model
    for p in dotted.split('.'):
        m = getattr(m, p)
    return m

@torch.no_grad()
def _cache_features_aug_val(body_model, val_loader, device):
    _fix_aug_rng()
    name, last = _find_last_linear(body_model)
    feats, last_in = [], None
    def _hook(m, inp, out):
        nonlocal last_in
        last_in = inp[0].detach()
    h = last.register_forward_hook(_hook)
    body_model.eval()
    for x, _ in val_loader:
        x = x.to(device, non_blocking=True)
        _ = body_model(x)
        feats.append(last_in.flatten(1).cpu())
    h.remove()
    Z = torch.cat(feats, dim=0).double().numpy()
    return Z, name

def _extract_last_linear_from_sd(sd: dict, last_name: str):
    W = sd[f"{last_name}.weight"].cpu().numpy().astype(np.float64)
    b = sd[f"{last_name}.bias"].cpu().numpy().astype(np.float64)
    return W, b

def _stack_last_linear_from_snaps(snapshots, last_name: str):
    W_list, b_list = [], []
    for sd in snapshots:
        Wi, bi = _extract_last_linear_from_sd(sd, last_name)
        W_list.append(Wi[None, ...]); b_list.append(bi[None, ...])
    W = np.concatenate(W_list, axis=0).astype(np.float64)
    B = np.concatenate(b_list, axis=0).astype(np.float64)
    return W, B  # [K,C,D], [K,C]

@torch.no_grad()
def _build_M_HAC_QS_trueprob_aug_val(snapshots, val_loader, build_model_fn, device, y_vec):
    K = len(snapshots); N = y_vec.size
    P_true = np.zeros((K, N), dtype=np.float64)
    for k, sd in enumerate(snapshots):
        _fix_aug_rng()
        m = build_model_fn(); m.load_state_dict(sd, strict=True); m.to(device).eval()
        idx0 = 0
        for x, y in val_loader:
            x = x.to(device, non_blocking=True)
            logits = m(x)
            probs = torch.softmax(logits, dim=1).to(torch.float64).cpu().numpy()
            y_np = y.numpy()
            P_true[k, idx0: idx0+y_np.shape[0]] = probs[np.arange(y_np.shape[0]), y_np]
            idx0 += y_np.shape[0]
        del m
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    def _center(X): return X - X.mean(axis=1, keepdims=True)
    def _qs_kernel(x):
        if x == 0.0: return 1.0
        a = 6.0*math.pi*x/5.0
        return (25.0/(12.0*math.pi**2*x**2))*(math.sin(a)/a - math.cos(a))
    def _hac_qs_rho(Fc, H=None):
        raw = (Fc @ Fc.T) / max(1, Fc.shape[1])
        gamma0 = float(np.mean(np.diag(raw)))
        if H is None: H = max(1, int(round(Fc.shape[0]**(1/3))))
        rho=[1.0]
        for h in range(1, H+1):
            val = float(np.mean(np.diag(raw, k=h)) / (gamma0 + 1e-12))
            rho.append(max(min(val, 0.999999), -0.999999))
        return np.array(rho), gamma0
    def _build_M_from_rho(K, rho):
        H = len(rho)-1
        M = np.eye(K, dtype=np.float64)
        for h in range(1, H+1):
            k = _qs_kernel(h/float(H)) * rho[h]
            if abs(k) < 1e-12: continue
            i = np.arange(0, K-h)
            M[i, i+h] += k; M[i+h, i] += k
        return M
    def _psd_project(M, eps=1e-9):
        w, V = np.linalg.eigh(M); w = np.maximum(w, eps)
        return (V * w) @ V.T

    F = P_true.astype(np.float64)
    Fc = _center(F)
    rho, _ = _hac_qs_rho(Fc, H=None)
    M = _psd_project(_build_M_from_rho(K, rho))
    return M  # [K,K]

@torch.no_grad()
def _build_crit_subset_indices_aug(ref_model, val_loader, frac, device):
    utils.bn_update(loaders['train'], ref_model)
    ref_model.eval()
    _fix_aug_rng()
    logits_all, labels_all = [], []
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

def _project_topk_per_class(A: np.ndarray, k_top: int) -> np.ndarray:
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

def _optimize_alpha_class_conditional_Wb(Z_val: np.ndarray, y_val: np.ndarray, S_idx: np.ndarray,
                                         W_stack: np.ndarray, b_stack: np.ndarray,
                                         M: np.ndarray,
                                         steps=150, lr=0.3, lam=1e-3, eta=1e-2, topk=8, log_every=25):
    K, C, D = W_stack.shape
    Zt = torch.from_numpy(Z_val).to(torch.float64)
    yt = torch.from_numpy(y_val).to(torch.long)
    Wt = torch.from_numpy(W_stack).to(torch.float64)
    bt = torch.from_numpy(b_stack).to(torch.float64)
    Mt = torch.from_numpy(M).to(torch.float64)
    U  = torch.full((K, C), 1.0/float(K), dtype=torch.float64)

    St = torch.from_numpy(np.asarray(S_idx, dtype=np.int64)) if S_idx is not None else None

    phi = torch.zeros(K, C, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.Adam([phi], lr=lr)

    best_obj, best_A = float('inf'), None
    for t in range(1, steps+1):
        A = torch.softmax(phi, dim=0)                     # [K,C]
        Wc = torch.einsum('kc,kcd->cd', A, Wt)            # [C,D]
        bc = torch.einsum('kc,kc->c',   A, bt)            # [C]
        logits = Zt @ Wc.t() + bc.unsqueeze(0)            # [N,C]
        ce = F.cross_entropy(logits.index_select(0, St), yt.index_select(0, St)) if St is not None else F.cross_entropy(logits, yt)
        reg  = lam * torch.sum((A - U) * (A - U))
        quad = eta * torch.sum(A * (Mt @ A))
        obj = ce + reg + quad
        obj.backward(); opt.step(); opt.zero_grad()

        val = float(obj.detach().cpu().item())
        if val < best_obj:
            best_obj = val
            best_A = A.detach().cpu().numpy().astype(np.float64)
        if (t % log_every == 0) or (t == 1):
            with torch.no_grad():
                r2 = float(torch.sum(A*A).cpu().item())
                ess = 1.0 / max(r2, 1e-12)
            print(f"[A {t:03d}] obj={val:.6e} | CE={float(ce):.6e} | η·αᵀMα={float((eta*quad).detach().cpu().item()):.4e} | ESS≈{ess:.2f}")

    A_np = best_A if best_A is not None else A.detach().cpu().numpy().astype(np.float64)
    A_np = _project_topk_per_class(A_np, topk)
    return A_np

@torch.no_grad()
def _replace_last_linear(model, last_name, W_mix, b_mix):
    last = _get_module_by_name(model, last_name)
    last.weight.copy_(torch.from_numpy(W_mix).to(last.weight.dtype).to(last.weight.device))
    last.bias.copy_(  torch.from_numpy(b_mix).to(last.bias.dtype).to(last.bias.device))

# ====== 仅最小改动：为函数侧集成补齐“每快照 BN 重估”的可选开关 ======
@torch.no_grad()
def _eval_function_ensemble_from_alpha_bar(
    snapshots, alpha_bar, loader, build_model_fn, device, mode='linear',
    bn_loader=None, bn_recal=False  # <-- 新增：可选 BN 重估
):
    K = len(snapshots)
    alpha_bar = np.asarray(alpha_bar, dtype=np.float64)
    mix_accum = None
    labels_all = []
    for j in range(K):
        try: _fix_aug_rng()  # 验证集时固定增广；测试集通常无增广
        except Exception: pass
        m = build_model_fn(); m.load_state_dict(snapshots[j], strict=True); m.to(device).eval()

        # ✨ 可选：为每个快照重估 BN（与 HeadOnly 使用同一口径的 loader，例如 loaders['train']）
        if bn_recal and (bn_loader is not None):
            utils.bn_update(bn_loader, m)

        chunk_list = []
        for x, _ in loader:
            x = x.to(device, non_blocking=True)
            if mode == 'logpool':
                arr = F.log_softmax(m(x), dim=1).double().cpu().numpy()
            else:
                arr = F.softmax(m(x), dim=1).double().cpu().numpy()
            chunk_list.append(arr)
        pred = np.concatenate(chunk_list, axis=0)  # [N,C]
        mix_accum = pred * float(alpha_bar[j]) if mix_accum is None else (mix_accum + pred * float(alpha_bar[j]))
        del m
        if torch.cuda.is_available(): torch.cuda.empty_cache()

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

# ------------------------------ 主：FS-Alpha（W,b + aug-val） ------------------------------
if args.do_func_ens and len(swa_snapshots) > 0:
    # 1) 困难子集（aug-val）
    ref_for_margin = swa_model if args.swa else model
    S_idx = _build_crit_subset_indices_aug(ref_for_margin, loaders['val'], args.crit_fraction, device)
    if S_idx is None:
        print(f"[α/A-learn] Using FULL aug-val (crit_fraction={args.crit_fraction:.2f} ignored).")
    else:
        print(f"[α/A-learn] Using critical subset |S|={len(S_idx)} / N={len(val_set)} (aug-val, frac={args.crit_fraction:.2f}).")

    # 2) y on aug-val
    y_list = []
    _fix_aug_rng()
    for _, y in loaders['val']:
        y_list.append(y.numpy())
    y_val = np.concatenate(y_list, axis=0).astype(np.int64)

    # 3) 用 SWA 身体缓存 aug-val 特征
    utils.bn_update(loaders['train'], swa_model if args.swa else model)
    body_for_feats = swa_model if args.swa else model
    Z_val, last_name = _cache_features_aug_val(body_for_feats, loaders['val'], device)

    # 4) 快照的最后线性层堆叠
    W_stack, b_stack = _stack_last_linear_from_snaps(swa_snapshots, last_name)
    K, C, D = W_stack.shape
    assert C == num_classes, f"Head classes mismatch: {C} vs {num_classes}"

    # 5) HAC(QS) on aug-val true prob
    def _bm(): return build_model()
    M_np = _build_M_HAC_QS_trueprob_aug_val(swa_snapshots, loaders['val'], _bm, device, y_val)

    # 6) 在 W,b 空间优化类条件 A
    if args.learn_func_w:
        A = _optimize_alpha_class_conditional_Wb(
            Z_val, y_val, S_idx, W_stack, b_stack, M_np,
            steps=args.func_w_steps, lr=args.func_w_lr,
            lam=args.func_w_l2, eta=args.func_w_eta,
            topk=args.func_topk if args.func_topk > 0 else 0,
            log_every=25
        )
        alpha_bar = A.mean(axis=1).astype(np.float64)
        s = alpha_bar.sum()
        alpha_bar = (np.ones_like(alpha_bar)/len(alpha_bar)) if (not np.isfinite(s) or s<=1e-12) else (alpha_bar/s)
        print("[FS-Alpha] learned A[K,C] on aug-val; alpha_bar preview:",
              np.round(alpha_bar[:min(10, alpha_bar.size)], 4).tolist())
    else:
        A = None
        alpha_bar = np.ones(len(swa_snapshots), dtype=np.float64)/float(len(swa_snapshots))
        print("[FS-Alpha] use uniform alpha_bar (no A learning).")

    # 7) HeadOnly（把 W_mix, b_mix 写回一个拷贝，测试）
    if A is not None:
        W_mix = np.einsum('kc,kcd->cd', A, W_stack)
        b_mix = np.einsum('kc,kc->c',   A, b_stack)
        headonly_model = build_model()
        headonly_model.load_state_dict((swa_model if args.swa else model).state_dict(), strict=True)
        headonly_model.to(device)
        _replace_last_linear(headonly_model, last_name, W_mix, b_mix)
        utils.bn_update(loaders['train'], headonly_model)  # 与源码一致（aug BN）
        head_res = utils.eval(loaders['test'], headonly_model, criterion, device)
        print(f"[FS-Alpha HeadOnly (W,b on aug-val)] Test Acc: {head_res['accuracy']:.4f}")
        # 若需要三指标：
        print("[FS-Alpha HeadOnly] (Acc/NLL/ECE) Test:", _eval_full_metrics(headonly_model, loaders['test']))

    # 8) 函数侧集成（ᾱ）：Linear / LogPool
    # —— 仅最小改动：为 func-ensemble 传入与 HeadOnly 相同的 BN 口径（loaders['train']，aug BN）
    if args.func_type in ('linear','both'):
        res_lin = _eval_function_ensemble_from_alpha_bar(
            swa_snapshots, alpha_bar, loaders['test'], _bm, device,
            mode='linear', bn_loader=loaders['train'], bn_recal=True
        )
        print(f"[FS-Alpha Func-Ensemble Linear] Test: Acc={res_lin['acc']:.4f} | NLL={res_lin['nll']:.4f} | ECE={res_lin['ece']:.4f}")
    if args.func_type in ('logpool','both'):
        res_log = _eval_function_ensemble_from_alpha_bar(
            swa_snapshots, alpha_bar, loaders['test'], _bm, device,
            mode='logpool', bn_loader=loaders['train'], bn_recal=True
        )
        print(f"[FS-Alpha Func-Ensemble LogPool]  Test: Acc={res_log['acc']:.4f} | NLL={res_log['nll']:.4f} | ECE={res_log['ece']:.4f}")

elif args.do_func_ens:
    print("[FS-Alpha] No SWA snapshots were collected; nothing to evaluate.")
