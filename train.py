import argparse
import os
import sys
import time
import json
import copy
import torch
import torch.nn.functional as F
import torchvision
import models
import utils
import tabulate
from torch.utils.data import random_split
import pandas as pd
import numpy as np

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

args = parser.parse_args()

# --- 自动行为：开了 SWA 就默认同时跑 ASWA（除非 --no_aswa） ---
if args.swa and not args.no_aswa:
    args.aswa = True
# （可选）若只写了 --aswa，也自动打开 --swa，避免误用
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
def eval_nll_ece(model_, loader, device_):
    """返回 {'acc':float, 'nll':float, 'ece':float}，和结构无关（VGG/ResNet/WRN通用）"""
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

# --------------------------- Final eval of running / SWA / ASWA ---------------------------
utils.bn_update(loaders['train'], model)
print("Running model Train: ", utils.eval(loaders['train'], model, criterion, device))
print("Running model Val:  ", utils.eval(loaders['val'],   model, criterion, device))
print("Running model Test: ", utils.eval(loaders['test'],  model, criterion, device))

if args.swa:
    utils.bn_update(loaders['train'], swa_model)
    swa_test_full = eval_nll_ece(swa_model, loaders['test'], device)
    print("SWA Train: ", utils.eval(loaders['train'], swa_model, criterion, device))
    print("SWA Val:   ", utils.eval(loaders['val'],   swa_model, criterion, device))
    print("SWA Test:  ", utils.eval(loaders['test'],  swa_model, criterion, device))
    print(f"SWA Test (Acc/NLL/ECE): Acc={swa_test_full['acc']:.4f} | NLL={swa_test_full['nll']:.4f} | ECE={swa_test_full['ece']:.4f}")

if args.aswa:
    utils.bn_update(loaders['train'], aswa_model)
    aswa_test_full = eval_nll_ece(aswa_model, loaders['test'], device)
    print("ASWA Train: ", utils.eval(loaders['train'], aswa_model, criterion, device))
    print("ASWA Val:   ", utils.eval(loaders['val'],   aswa_model, criterion, device))
    print("ASWA Test:  ", utils.eval(loaders['test'],  aswa_model, criterion, device))
    print(f"ASWA Test (Acc/NLL/ECE): Acc={aswa_test_full['acc']:.4f} | NLL={aswa_test_full['nll']:.4f} | ECE={aswa_test_full['ece']:.4f}")

# --------------------------- Function-side ensembles (from SWA snapshots) ---------------------------
if args.do_func_ens and len(swa_snapshots) > 0:
    # (1) load optional weights
    w = None
    if args.func_weights is not None and os.path.isfile(args.func_weights):
        if args.func_weights.endswith('.npy'):
            w = np.load(args.func_weights)
        elif args.func_weights.endswith('.json'):
            with open(args.func_weights, 'r') as f:
                w = torch.tensor(json.load(f), dtype=torch.float64).numpy()
        else:
            print(f"[WARN] Unrecognized weight file format: {args.func_weights}; fallback to uniform.")
            w = None

    # (2) wrappers calling your utils.* functions
    def _build_model_for_eval():
        m = build_model()
        return m

    if args.func_type in ('linear','both'):
        res_lin = utils.evaluate_function_ensemble_linear(
            snapshots=swa_snapshots,
            loader=loaders['test'],
            device=device,
            build_model_fn=_build_model_for_eval,
            weights=w
        )
        print(f"[Func-Ensemble Linear] Test: Acc={res_lin['acc']:.4f} | NLL={res_lin['nll']:.4f} | ECE={res_lin['ece']:.4f}")

    if args.func_type in ('logpool','both'):
        res_log = utils.evaluate_function_ensemble_logpool(
            snapshots=swa_snapshots,
            loader=loaders['test'],
            device=device,
            build_model_fn=_build_model_for_eval,
            weights=w
        )
        print(f"[Func-Ensemble LogPool] Test: Acc={res_log['acc']:.4f} | NLL={res_log['nll']:.4f} | ECE={res_log['ece']:.4f}")

else:
    if args.do_func_ens:
        print("[Func-Ensemble] No SWA snapshots were collected; nothing to evaluate.")
