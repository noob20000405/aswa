import os
import torch


def adjust_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def save_checkpoint(dir, epoch, **kwargs):
    state = {
        'epoch': epoch,
    }
    state.update(kwargs)
    filepath = os.path.join(dir, 'checkpoint-%d.pt' % epoch)
    torch.save(state, filepath)


def train_epoch(*,epoch=None, loader=None, model=None, criterion=None, optimizer=None, device=None):
    loss_sum = 0.0
    correct = 0.0

    if isinstance(loader.sampler, torch.utils.data.RandomSampler)==False:
        loader.sampler.set_epoch(epoch)

    model.train()

    for i, (input, target) in enumerate(loader):
        input = input.to(device)
        target = target.to(device)

        output = model(input)
        loss = criterion(output, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_sum += loss.item() * input.size(0)
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).sum().item()
        
    return {
        'loss': loss_sum / len(loader.dataset),
        'accuracy': correct / len(loader.dataset) * 100.0,
    }

@torch.no_grad()
def eval(loader, model, criterion,device):
    loss_sum = 0.0
    correct = 0.0

    model.eval()

    for i, (input, target) in enumerate(loader):
        input = input.to(device)
        target = target.to(device)

        output = model(input)
        loss = criterion(output, target)

        loss_sum += loss.item() * input.size(0)
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).sum().item()

    return {
        'loss': loss_sum / len(loader.dataset),
        'accuracy': correct / len(loader.dataset) * 100.0,
    }

def moving_average(net1, net2, alpha=1):
    for param1, param2 in zip(net1.parameters(), net2.parameters()):
        param1.data *= (1.0 - alpha)
        param1.data += param2.data * alpha


def _check_bn(module, flag):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        flag[0] = True


def check_bn(model):
    flag = [False]
    model.apply(lambda module: _check_bn(module, flag))
    return flag[0]


def reset_bn(module):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        module.running_mean = torch.zeros_like(module.running_mean)
        module.running_var = torch.ones_like(module.running_var)


def _get_momenta(module, momenta):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        momenta[module] = module.momentum


def _set_momenta(module, momenta):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        module.momentum = momenta[module]


def bn_update(loader, model):
    """
        BatchNorm buffers update (if any).
        Performs 1 epochs to estimate buffers average using train dataset.

        :param loader: train dataset loader for buffers average estimation.
        :param model: model being update
        :return: None
    """
    if not check_bn(model):
        return
    model.train()
    momenta = {}
    model.apply(reset_bn)
    model.apply(lambda module: _get_momenta(module, momenta))
    device=next(model.parameters()).device

    n = 0
    for input, _ in loader:
        # @TODO: This needs to be device
        input = input.to(device)
        b = input.data.size(0)

        momentum = b / (n + b)
        for module in momenta.keys():
            module.momentum = momentum

        model(input)
        n += b

    model.apply(lambda module: _set_momenta(module, momenta))

import numpy as np
import torch
import torch.nn.functional as F

@torch.no_grad()
def _eval_probs_metrics(probs_list, labels_list):
    probs = np.concatenate(probs_list, axis=0)
    labels = np.concatenate(labels_list, axis=0)
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
def evaluate_function_ensemble_linear(snapshots, loader, device, build_model_fn, weights=None):
    """
    线性池化:  p_mix = sum_j alpha_j * softmax(f_j(x))
    与模型结构无关（VGG/ResNet/WRN均可），不依赖 .fc 属性。
    """
    K = len(snapshots)
    if K == 0:
        return {'acc': 0.0, 'nll': 0.0, 'ece': 0.0}
    # 权重
    if weights is None:
        w = np.ones(K, dtype=np.float64) / float(K)
    else:
        w = np.asarray(weights, dtype=np.float64).reshape(-1)
        assert w.size == K, f"weights length {w.size} != #snapshots {K}"
        s = w.sum()
        w = np.ones(K)/K if (not np.isfinite(s) or s <= 0) else (w / s)

    # 构建模型列表
    models = [build_model_fn() for _ in range(K)]
    for m, sd in zip(models, snapshots):
        m.load_state_dict(sd, strict=True)
        m.to(device).eval()

    probs_all, labels_all = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        mix = None  # 见第一轮
        for j, m in enumerate(models):
            p = F.softmax(m(x), dim=1)  # [B, C]
            if mix is None:
                mix = torch.zeros_like(p)  # [B, C]
            mix.add_(p, alpha=float(w[j]))
        probs_all.append(mix.detach().cpu().numpy())
        labels_all.append(y.numpy())

    return _eval_probs_metrics(probs_all, labels_all)

@torch.no_grad()
def evaluate_function_ensemble_logpool(snapshots, loader, device, build_model_fn, weights=None):
    """
    对数池化（几何平均）:  p_mix ∝ exp( sum_j alpha_j * log_softmax(f_j(x)) )
    同样与模型结构无关。
    """
    K = len(snapshots)
    if K == 0:
        return {'acc': 0.0, 'nll': 0.0, 'ece': 0.0}
    # 权重
    if weights is None:
        w = np.ones(K, dtype=np.float64) / float(K)
    else:
        w = np.asarray(weights, dtype=np.float64).reshape(-1)
        assert w.size == K, f"weights length {w.size} != #snapshots {K}"
        s = w.sum()
        w = np.ones(K)/K if (not np.isfinite(s) or s <= 0) else (w / s)

    models = [build_model_fn() for _ in range(K)]
    for m, sd in zip(models, snapshots):
        m.load_state_dict(sd, strict=True)
        m.to(device).eval()

    probs_all, labels_all = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        log_mix = None
        for j, m in enumerate(models):
            logp = F.log_softmax(m(x), dim=1).double()  # [B, C]
            if log_mix is None:
                log_mix = torch.zeros_like(logp)         # [B, C]
            log_mix.add_(logp, alpha=float(w[j]))
        probs = torch.softmax(log_mix, dim=1).float()
        probs_all.append(probs.detach().cpu().numpy())
        labels_all.append(y.numpy())

    return _eval_probs_metrics(probs_all, labels_all)


