import argparse
import os
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

# --------------------------- Args ---------------------------
parser = argparse.ArgumentParser(description='SWA-ASWA training (+ function-side ensembles)')
parser.add_argument('--dir', type=str, default='.', required=False, help='training directory (default: .)')

parser.add_argument('--dataset', type=str, default='CIFAR10', choices=['CIFAR10','CIFAR100'])
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--num_workers', type=int, default=16)
parser.add_argument('--model', type=str, default='VGG16', metavar='MODEL', help='model name in models/*')
parser.add_argument('--optim', type=str, default='SGD', choices=['SGD','Adam'])
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--save_freq', type=int, default=25)
parser.add_argument('--eval_freq', type=int, default=1)
parser.add_argument('--lr_init', type=float, default=0.1)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--wd', type=float, default=1e-4)

parser.add_argument('--swa', action='store_true', help='enable SWA averaging')
parser.add_argument('--aswa', action='store_true', help='enable ASWA tri-state update (uses val)')
parser.add_argument('--no_aswa', action='store_true', help='when --swa is on, disable ASWA (default: off)')

parser.add_argument('--swa_start', type=int, default=161, help='epoch to start SWA averaging (1-based)')
parser.add_argument('--swa_lr', type=float, default=0.05, help='constant LR during SWA stage')
parser.add_argument('--swa_c_epochs', type=int, default=1, help='SWA snapshot/collection frequency in epochs')
parser.add_argument('--snapshot_stride', type=int, default=1, help='keep every n-th SWA snapshot (>=1)')

parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--val_ratio', type=float, default=0.1)

# Function-side ensemble (from SWA snapshots)
parser.add_argument('--do_func_ens', action='store_true', help='evaluate function-side ensembles on SWA snapshots')
parser.add_argument('--func_type', type=str, default='both', choices=['linear','logpool','both'],
                    help='which function-side ensemble(s) to evaluate')
parser.add_argument('--func_weights', type=str, default=None,
                    help='optional path to weights (json/npy) of length K; default=uniform')

# Learn α on validation (critical subset)
parser.add_argument('--learn_func_w', action='store_true', help='learn ensemble weights α on val set')
parser.add_argument('--func_w_type', type=str, default='linear', choices=['linear','logpool'],
                    help='objective to learn α: linear or logpool')
parser.add_argument('--func_w_steps', type=int, default=150, help='steps to optimize α')
parser.add_argument('--func_w_lr', type=float, default=0.3, help='learning rate for α')
parser.add_argument('--crit_fraction', type=float, default=0.5,
                    help='fraction (0~1) of lowest-margin val samples used to learn α (default 0.5)')

args = parser.parse_args()

# --- 自动行为：开了 SWA 就默认同时跑 ASWA（除非 --no_aswa） ---
if args.swa and not args.no_aswa:
    args.aswa = True
# （容错）若只写了 --aswa，也自动打开 --swa
if args.aswa:
    args.swa = True

# --------------------------- Setup ---------------------------
print('Preparing directory %s' % args.dir)
os.makedirs(args.dir, exist_ok=True)

torch.backends.cudnn.benchmark = True
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

print('Using model %s' % args.model)
model_cfg = getattr(models, args.model)

# Dataset
print('Loading dataset:', args.dataset)
assert 1.0 > args.val_ratio > 0.0
if args.dataset == "CIFAR10":
    train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True,
                                             transform=model_cfg.transform_train)
    test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True,
                                            transform=model_cfg.transform_test)
elif args.dataset == "CIFAR100":
    train_set = torchvision.datasets.CIFAR100(root='./data', train=True, download=True,
                                              transform=model_cfg.transform_train)
    test_set = torchvision.datasets.CIFAR100(root='./data', train=False, download=True,
                                             transform=model_cfg.transform_test)
else:
    raise ValueError(f"Incorrect dataset {args.dataset}")

train_size = int(len(train_set) * (1 - args.val_ratio))
val_size = len(train_set) - train_size
num_classes = max(train_set.targets) + 1
train_set, val_set = random_split(train_set, [train_size, val_size],
                                  generator=torch.Generator().manual_seed(args.seed))

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

# --------------------------- Models ---------------------------
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
aswa_ensemble_weights = [1.0]  # 初始包含“起始 aswa_model”的权重

model.to(device)
swa_model.to(device)
aswa_model.to(device)

# 用于函数侧集成：只使用“按 SWA 频率收集”的快照
swa_snapshots = []  # list of CPU state_dict (each is a full model snapshot)

# --------------------------- Helpers ---------------------------
def schedule(epoch):
    # epoch 从 0 开始
    t = (epoch) / (args.swa_start if args.swa else args.epochs)
    lr_ratio = args.swa_lr / args.lr_init if args.swa else 0.01
    if t <= 0.5:
        factor = 1.0
    elif t <= 0.9:
        factor = 1.0 - (1.0 - lr_ratio) * (t - 0.5) / 0.4
    else:
        factor = lr_ratio
    return args.lr_init * factor

@torch.no_grad()
def eval_full_metrics(model_, loader, device_):
    """返回 {'acc':float, 'nll':float, 'ece':float}，结构无关（VGG/ResNet/WRN通用）"""
    model_.eval()
    probs_all, labels_all = [], []
    for x, y in loader:
        x = x.to(device_, non_blocking=True)
        logits = model_(x)
        probs = F.softmax(logits, dim=1).float().detach().cpu().numpy()
        probs_all.append(probs)
        labels_all.append(y.numpy())
    probs = np.concatenate(probs_all, axis=0)
    labels = np.concatenate(labels_all, axis=0)
    acc = float((probs.argmax(1) == labels).mean())
    nll = float(-np.log(np.clip(probs[np.arange(labels.size), labels], 1e-12, 1.0)).mean())
    # ECE（15 bins）
    bins = np.linspace(0, 1, 16)
    conf = probs.max(1)
    preds = probs.argmax(1)
    correct = (preds == labels).astype(np.float32)
    ece = 0.0
    for i in range(15):
        lo, hi = bins[i], bins[i+1]
        m = (conf > lo) & (conf <= hi) if i > 0 else (conf >= lo) & (conf <= hi)
        if m.any():
            ece += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return {'acc': acc, 'nll': nll, 'ece': ece}

@torch.no_grad()
def _compute_val_margins(model_obj, loader, device_):
    """返回 margins [N], labels [N]；margin = true_logit - max_other_logit"""
    model_obj.eval()
    logits_all, labels_all = [], []
    for x, y in loader:
        x = x.to(device_, non_blocking=True)
        logits_all.append(model_obj(x).detach().cpu())
        labels_all.append(y.detach().cpu())
    logits = torch.cat(logits_all, 0)   # [N,C]
    labels = torch.cat(labels_all, 0)   # [N]
    true = logits[torch.arange(labels.size(0)), labels]
    logits_sc = logits.clone()
    logits_sc[torch.arange(labels.size(0)), labels] = -1e9
    other_max = logits_sc.max(dim=1).values
    margins = (true - other_max).numpy()
    return margins, labels.numpy()

def _build_crit_subset_indices(ref_model, val_loader, device_, frac):
    """基于 ref_model 的验证集 margin，选取最小的 frac 部分索引；frac ∈ (0,1]"""
    utils.bn_update(loaders['train'], ref_model)  # BN 对齐
    margins, _ = _compute_val_margins(ref_model, val_loader, device_)
    frac = float(np.clip(frac, 0.0, 1.0))
    if frac <= 0 or frac >= 1:
        return None  # 用全量
    thr = np.quantile(margins, frac)
    S_idx = np.nonzero(margins <= thr)[0]
    return S_idx.astype(np.int64)

@torch.no_grad()
def _cache_val_probs_or_logprobs(snapshots, loader, build_model_fn, device_, want_log=False):
    """返回 list 长度为 K；每个元素是 [N, C] 的 numpy 数组（概率或对数概率）"""
    K = len(snapshots)
    if K == 0: return []
    # 构建模型列表
    models_list = [build_model_fn() for _ in range(K)]
    for m, sd in zip(models_list, snapshots):
        m.load_state_dict(sd, strict=True)
        m.to(device_).eval()

    mats = []
    for x, _ in loader:
        x = x.to(device_, non_blocking=True)
        cur = []
        for m in models_list:
            out = m(x)
            arr = (F.log_softmax(out, dim=1) if want_log else F.softmax(out, dim=1)).cpu().numpy()
            cur.append(arr)  # [B,C]
        mats.append(np.stack(cur, axis=0))  # [K,B,C]
    mats = np.concatenate(mats, axis=1)  # [K,N,C]
    return [mats[k] for k in range(K)]   # list of K arrays [N,C]

def _learn_alpha_on_val_linear(snapshots, loader_val, build_model_fn, device_, steps=150, lr=0.3, S_idx=None):
    # 收集标签
    labels=[]
    for _, y in loader_val:
        labels.append(y.numpy())
    y = np.concatenate(labels, axis=0).astype(np.int64)

    # 预缓存各模型在 val 上的概率
    P_list = _cache_val_probs_or_logprobs(snapshots, loader_val, build_model_fn, device_, want_log=False)  # list of [N,C]
    K, N, C = len(P_list), P_list[0].shape[0], P_list[0].shape[1]
    P = np.stack(P_list, axis=0)  # [K,N,C]

    phi = torch.zeros(K, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.Adam([phi], lr=lr)
    y_t = torch.from_numpy(y).long()
    P_t = torch.from_numpy(P)  # [K,N,C]
    idx_t = torch.from_numpy(S_idx).long() if S_idx is not None else None

    for _ in range(1, steps+1):
        opt.zero_grad()
        w = torch.softmax(phi, dim=0)                  # [K]
        mix = torch.einsum('k,knc->nc', w, P_t)        # [N,C]
        if idx_t is not None:
            mix_sel = mix.index_select(0, idx_t)
            y_sel   = y_t.index_select(0, idx_t)
            loss = F.nll_loss(torch.log(mix_sel.clamp_min_(1e-12)), y_sel)
        else:
            loss = F.nll_loss(torch.log(mix.clamp_min_(1e-12)), y_t)
        loss.backward(); opt.step()

    with torch.no_grad():
        w = torch.softmax(phi, dim=0).cpu().numpy()
    return w

def _learn_alpha_on_val_logpool(snapshots, loader_val, build_model_fn, device_, steps=150, lr=0.3, S_idx=None):
    labels=[]
    for _, y in loader_val:
        labels.append(y.numpy())
    y = np.concatenate(labels, axis=0).astype(np.int64)

    L_list = _cache_val_probs_or_logprobs(snapshots, loader_val, build_model_fn, device_, want_log=True)  # list of [N,C] (logp)
    K, N, C = len(L_list), L_list[0].shape[0], L_list[0].shape[1]
    L = np.stack(L_list, axis=0)  # [K,N,C]

    phi = torch.zeros(K, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.Adam([phi], lr=lr)
    y_t = torch.from_numpy(y).long()
    L_t = torch.from_numpy(L)     # [K,N,C]
    idx_t = torch.from_numpy(S_idx).long() if S_idx is not None else None

    for _ in range(1, steps+1):
        opt.zero_grad()
        w = torch.softmax(phi, dim=0)                 # [K]
        log_mix = torch.einsum('k,knc->nc', w, L_t)   # [N,C]
        if idx_t is not None:
            log_sel = log_mix.index_select(0, idx_t)
            y_sel   = y_t.index_select(0, idx_t)
            loss = F.nll_loss(torch.log_softmax(log_sel, dim=1), y_sel)
        else:
            loss = F.nll_loss(torch.log_softmax(log_mix, dim=1), y_t)
        loss.backward(); opt.step()

    with torch.no_grad():
        w = torch.softmax(phi, dim=0).cpu().numpy()
    return w

def _load_or_uniform_weights(K):
    # 优先外部文件，否则均匀
    if args.func_weights is not None and os.path.isfile(args.func_weights):
        if args.func_weights.endswith('.npy'):
            w = np.load(args.func_weights)
        elif args.func_weights.endswith('.json'):
            with open(args.func_weights, 'r') as f:
                w = np.array(json.load(f), dtype=np.float64)
        else:
            print(f"[WARN] Unrecognized weight file format: {args.func_weights}; fallback to uniform.")
            w = None
    else:
        w = None
    if w is None:
        w = np.ones(K, dtype=np.float64) / float(max(1,K))
    w = np.asarray(w, dtype=np.float64).reshape(-1)
    assert w.size == K, f"weights length {w.size} != #snapshots {K}"
    s = w.sum()
    if not np.isfinite(s) or s <= 0:
        w = np.ones(K, dtype=np.float64) / float(max(1,K))
    else:
        w = w / s
    return w

# --------------------------- Optim ---------------------------
criterion = F.cross_entropy
if args.optim == "SGD":
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr_init,
                                momentum=args.momentum, weight_decay=args.wd)
elif args.optim == "Adam":
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr_init)
else:
    raise ValueError("Unknown optimizer")

# --------------------------- Training loop ---------------------------
start_epoch = 0
rows = []

for epoch in range(start_epoch, args.epochs):
    t0 = time.time()

    # (Adjust LR)
    if args.optim == "SGD" and epoch > 0:
        lr = schedule(epoch)
        utils.adjust_learning_rate(optimizer, lr)
    else:
        lr = args.lr_init

    # (.) Train one epoch
    train_res = utils.train_epoch(epoch=epoch, loader=loaders['train'],
                                  model=model, criterion=criterion,
                                  optimizer=optimizer, device=device)

    # (.) For fair eval, BN update then evaluate current running model
    utils.bn_update(loaders['train'], model)
    val_res  = utils.eval(loaders['val'],  model, criterion, device)
    test_res = utils.eval(loaders['test'], model, criterion, device)

    # Init result containers
    epoch_res = {
        "Running": {"train": train_res, "val": val_res, "test": test_res},
        "SWA": {"train": {"loss": "-", "accuracy": "-"},
                "val":   {"loss": "-", "accuracy": "-"},
                "test":  {"loss": "-", "accuracy": "-"}},
        "ASWA": {"train": {"loss": "-", "accuracy": "-"},
                 "val":   {"loss": "-", "accuracy": "-"},
                 "test":  {"loss": "-", "accuracy": "-"}},
    }

    # -------------- SWA stage --------------
    if args.swa and (epoch + 1) >= args.swa_start and (epoch + 1 - args.swa_start) % args.swa_c_epochs == 0:
        # (1) moving average for SWA model
        utils.moving_average(swa_model, model, 1.0 / (swa_n + 1.0))
        swa_n += 1.0

        # (1.1) 收集快照（只使用 SWA 的采样节奏；按 stride 下采样）
        idx_since_start = (epoch + 1 - args.swa_start)
        if (idx_since_start % max(1, args.snapshot_stride)) == 0:
            swa_snapshots.append({k: v.detach().cpu().clone() for k, v in model.state_dict().items()})

        # -------------- ASWA tri-state --------------
        if args.aswa:
            # (2.1) 评估“更新前”的 ASWA
            utils.bn_update(loaders['train'], aswa_model)
            current_val = utils.eval(loaders['val'], aswa_model, criterion, device)

            # (2.2) 缓存 ASWA 当前参数
            current_aswa_state_dict = copy.deepcopy(aswa_model.state_dict())
            aswa_state_dict = copy.deepcopy(aswa_model.state_dict())

            # (2.3) 试图把“当前 model”并入 ASWA（等权新增）
            total_w = sum(aswa_ensemble_weights)
            for k, params in model.state_dict().items():
                aswa_state_dict[k] = (aswa_state_dict[k] * total_w + params) / (total_w + 1.0)

            # (2.4) 验证“临时并入后的”表现
            aswa_model.load_state_dict(aswa_state_dict)
            utils.bn_update(loaders['train'], aswa_model)
            prov_val = utils.eval(loaders['val'], aswa_model, criterion, device)

            # (2.5) 决策：若变好则接受（append 1.0），否则回滚
            if prov_val["accuracy"] >= current_val["accuracy"]:
                aswa_ensemble_weights.append(1.0)
            else:
                aswa_model.load_state_dict(current_aswa_state_dict)

    # -------------- 报告 SWA/ASWA 的当前性能（带 BN-recal） --------------
    utils.bn_update(loaders['train'], swa_model)
    utils.bn_update(loaders['train'], aswa_model)

    epoch_res["SWA"] = {
        "train": utils.eval(loaders['train'], swa_model, criterion, device),
        "val":   utils.eval(loaders['val'],   swa_model, criterion, device),
        "test":  utils.eval(loaders['test'],  swa_model, criterion, device),
    }

    epoch_res["ASWA"] = {
        "train": utils.eval(loaders['train'], aswa_model, criterion, device),
        "val":   utils.eval(loaders['val'],   aswa_model, criterion, device),
        "test":  utils.eval(loaders['test'],  aswa_model, criterion, device),
    }

    # -------------- Logging --------------
    time_ep = time.time() - t0
    columns = [
        "ep", "time", "lr",
        "train_loss", "train_acc",
        "val_acc", "test_acc",
        "swa_train_acc", "swa_val_acc", "swa_test_acc",
        "aswa_train_acc", "aswa_val_acc", "aswa_test_acc"
    ]
    values = [
        epoch + 1, time_ep, lr,
        epoch_res["Running"]["train"]["loss"],
        epoch_res["Running"]["train"]["accuracy"],
        epoch_res["Running"]["val"]["accuracy"],
        epoch_res["Running"]["test"]["accuracy"],
        epoch_res["SWA"]["train"]["accuracy"],
        epoch_res["SWA"]["val"]["accuracy"],
        epoch_res["SWA"]["test"]["accuracy"],
        epoch_res["ASWA"]["train"]["accuracy"],
        epoch_res["ASWA"]["val"]["accuracy"],
        epoch_res["ASWA"]["test"]["accuracy"]
    ]

    rows.append(values)
    table = tabulate.tabulate([values], columns, tablefmt='simple', floatfmt='4.4f')
    if epoch % 40 == 0:
        table = table.split('\n')
        table = '\n'.join([table[1]] + table)
    else:
        table = table.split('\n')[2]
    print(table)

# --------------------------- Save last checkpoint ---------------------------
if args.epochs % args.save_freq != 0:
    utils.save_checkpoint(
        args.dir,
        args.epochs,
        state_dict=model.state_dict(),
        swa_state_dict=swa_model.state_dict() if args.swa else None,
        swa_n=swa_n if args.swa else None,
        optimizer=optimizer.state_dict()
    )

# --------------------------- CSV log ---------------------------
df = pd.DataFrame(rows, columns=[
    "ep", "time", "lr",
    "train_loss", "train_acc",
    "val_acc", "test_acc",
    "swa_train_acc", "swa_val_acc", "swa_test_acc",
    "aswa_train_acc", "aswa_val_acc", "aswa_test_acc"
])
df.to_csv(f"{args.dir}/results.csv", index=False)

# --------------------------- Final eval of running / SWA / ASWA (Acc/NLL/ECE) ---------------------------
utils.bn_update(loaders['train'], model)
run_train = eval_full_metrics(model, loaders['train'], device)
run_val   = eval_full_metrics(model, loaders['val'],   device)
run_test  = eval_full_metrics(model, loaders['test'],  device)
print(f"Running Train: Acc={run_train['acc']:.4f} | NLL={run_train['nll']:.4f} | ECE={run_train['ece']:.4f}")
print(f"Running Val  : Acc={run_val['acc']:.4f} | NLL={run_val['nll']:.4f} | ECE={run_val['ece']:.4f}")
print(f"Running Test : Acc={run_test['acc']:.4f} | NLL={run_test['nll']:.4f} | ECE={run_test['ece']:.4f}")

if args.swa:
    utils.bn_update(loaders['train'], swa_model)
    swa_train = eval_full_metrics(swa_model, loaders['train'], device)
    swa_val   = eval_full_metrics(swa_model, loaders['val'],   device)
    swa_test  = eval_full_metrics(swa_model, loaders['test'],  device)
    print(f"SWA Train:   Acc={swa_train['acc']:.4f} | NLL={swa_train['nll']:.4f} | ECE={swa_train['ece']:.4f}")
    print(f"SWA Val:     Acc={swa_val['acc']:.4f} | NLL={swa_val['nll']:.4f} | ECE={swa_val['ece']:.4f}")
    print(f"SWA Test:    Acc={swa_test['acc']:.4f} | NLL={swa_test['nll']:.4f} | ECE={swa_test['ece']:.4f}")

if args.aswa:
    utils.bn_update(loaders['train'], aswa_model)
    aswa_train = eval_full_metrics(aswa_model, loaders['train'], device)
    aswa_val   = eval_full_metrics(aswa_model, loaders['val'],   device)
    aswa_test  = eval_full_metrics(aswa_model, loaders['test'],  device)
    print(f"ASWA Train:  Acc={aswa_train['acc']:.4f} | NLL={aswa_train['nll']:.4f} | ECE={aswa_train['ece']:.4f}")
    print(f"ASWA Val:    Acc={aswa_val['acc']:.4f} | NLL={aswa_val['nll']:.4f} | ECE={aswa_val['ece']:.4f}")
    print(f"ASWA Test:   Acc={aswa_test['acc']:.4f} | NLL={aswa_test['nll']:.4f} | ECE={aswa_test['ece']:.4f}")

# --------------------------- Function-side ensembles (from SWA snapshots) ---------------------------
if args.do_func_ens and len(swa_snapshots) > 0:
    # 先基于 SWA（若开启，否则用 running）在 val 上选 critical 子集
    ref_for_margin = swa_model if args.swa else model
    S_idx = _build_crit_subset_indices(ref_for_margin, loaders['val'], device, args.crit_fraction)
    if S_idx is None:
        print(f"[α-learn] Using FULL val set (crit_fraction={args.crit_fraction:.2f} ignored).")
    else:
        print(f"[α-learn] Using critical subset |S|={len(S_idx)} / N={len(val_set)} (frac={args.crit_fraction:.2f}).")

    # (A) 决定 α：优先学习，否则加载/均匀
    if args.learn_func_w:
        if args.func_w_type == 'linear':
            alpha = _learn_alpha_on_val_linear(
                swa_snapshots, loaders['val'], build_model, device,
                steps=args.func_w_steps, lr=args.func_w_lr, S_idx=S_idx
            )
        else:
            alpha = _learn_alpha_on_val_logpool(
                swa_snapshots, loaders['val'], build_model, device,
                steps=args.func_w_steps, lr=args.func_w_lr, S_idx=S_idx
            )
        print("[Func-Ensemble] learned α (sum=1):", np.round(alpha, 4).tolist())
    else:
        alpha = _load_or_uniform_weights(len(swa_snapshots))
        print("[Func-Ensemble] use weights:", np.round(alpha, 4).tolist())

    # (B) 测试集评测（按选择的集成类型）
    def _build_model_for_eval():
        return build_model()

    if args.func_type in ('linear','both'):
        res_lin = utils.evaluate_function_ensemble_linear(
            snapshots=swa_snapshots,
            loader=loaders['test'],
            device=device,
            build_model_fn=_build_model_for_eval,
            weights=alpha
        )
        print(f"[Func-Ensemble Linear] Test: Acc={res_lin['acc']:.4f} | NLL={res_lin['nll']:.4f} | ECE={res_lin['ece']:.4f}")

    if args.func_type in ('logpool','both'):
        res_log = utils.evaluate_function_ensemble_logpool(
            snapshots=swa_snapshots,
            loader=loaders['test'],
            device=device,
            build_model_fn=_build_model_for_eval,
            weights=alpha
        )
        print(f"[Func-Ensemble LogPool]  Test: Acc={res_log['acc']:.4f} | NLL={res_log['nll']:.4f} | ECE={res_log['ece']:.4f}")

else:
    if args.do_func_ens:
        print("[Func-Ensemble] No SWA snapshots were collected; nothing to evaluate.")
