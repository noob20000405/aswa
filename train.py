# ===================== W,b 空间 + HAC(QS) + aug-val =====================

import math, random
import torch.nn as nn

EVAL_AUG_SEED = 12345  # 确保不同快照看到相同增广

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
    """
    用带增强的 loaders['val']，抓到最后 Linear 的输入特征 Z ∈ [N,D]。
    与 SWA/ASWA 验证一致；每次遍历前固定 RNG 确保增广一致。
    """
    _fix_aug_rng()
    name, last = _find_last_linear(body_model)
    feats = []
    last_in = None
    def _hook(m, inp, out):
        nonlocal last_in
        # inp[0] shape: [B, D]
        last_in = inp[0].detach()
    h = last.register_forward_hook(_hook)
    body_model.eval()
    for x, _ in val_loader:
        x = x.to(device, non_blocking=True)
        _ = body_model(x)
        feats.append(last_in.flatten(1).cpu())
    h.remove()
    Z = torch.cat(feats, dim=0).double().numpy()  # [N,D]
    return Z, name

def _extract_last_linear_from_sd(sd: dict, last_name: str):
    W = sd[f"{last_name}.weight"].cpu().numpy()  # [C,D]
    b = sd[f"{last_name}.bias"].cpu().numpy()    # [C]
    return W.astype(np.float64), b.astype(np.float64)

def _stack_last_linear_from_snaps(snapshots, last_name: str):
    W_list, b_list = [], []
    for sd in snapshots:
        Wi, bi = _extract_last_linear_from_sd(sd, last_name)
        W_list.append(Wi[None, ...])  # [1,C,D]
        b_list.append(bi[None, ...])  # [1,C]
    W = np.concatenate(W_list, axis=0).astype(np.float64)  # [K,C,D]
    b = np.concatenate(b_list, axis=0).astype(np.float64)  # [K,C]
    return W, b

@torch.no_grad()
def _build_M_HAC_QS_trueprob_aug_val(snapshots, val_loader, build_model_fn, device, y_vec):
    """
    真·HAC(QS)：在带增强的 val 上，计算每个快照对真类的概率向量 f_j，
    构造 F[K,N]，行中心化→HAC(QS)→M[K,K]，再做 PSD 投影。
    """
    # ---- 先缓存真类概率：确保每个快照看到同样增广 ----
    K = len(snapshots); N = y_vec.size
    P_true = np.zeros((K, N), dtype=np.float64)
    for k, sd in enumerate(snapshots):
        _fix_aug_rng()
        m = build_model_fn()
        m.load_state_dict(sd, strict=True); m.to(device).eval()
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

    # ---- HAC(QS) ----
    def _center(X): return X - X.mean(axis=1, keepdims=True)  # 按行中心化
    def _qs_kernel(x):
        if x == 0.0: return 1.0
        a = 6.0*math.pi*x/5.0
        return (25.0/(12.0*math.pi**2*x**2))*(math.sin(a)/a - math.cos(a))
    def _hac_qs_rho(Fc, H=None):
        # Fc: [K,N], 行中心化
        raw = (Fc @ Fc.T) / max(1, Fc.shape[1])  # [K,K]
        gamma0 = float(np.mean(np.diag(raw)))
        if H is None:
            H = max(1, int(round((Fc.shape[0])**(1/3))))
        rho = [1.0]
        for h in range(1, H+1):
            # 取 k-step 对角线均值 / gamma0
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
        w, V = np.linalg.eigh(M)
        w = np.maximum(w, eps)
        return (V * w) @ V.T

    F = P_true.astype(np.float64)          # [K,N]
    Fc = _center(F)
    rho, _ = _hac_qs_rho(Fc, H=None)
    M = _psd_project(_build_M_from_rho(K, rho))
    return M  # [K,K]

@torch.no_grad()
def _build_crit_subset_indices_aug(ref_model, val_loader, frac, device):
    """
    与你原逻辑一致：用 ref_model 的 margin 选困难子集，但在带增强的 val 上；
    每次遍历前固定 RNG，以便与快照评估一致。
    """
    utils.bn_update(loaders['train'], ref_model)  # 保持源码一致性
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
                                         steps=150, lr=0.3, lam=1e-3, eta=1e-2, topk=8, log_every=50):
    """
    目标与 WRN 版一致：在 W,b 空间学习类条件 A[K,C]（列 softmax 到 simplex），
    logits = Z @ W_mix^T + b_mix，CE 仅在困难子集 S 上；外加 L2 到均匀与 HAC 二次正则。
    """
    K, C, D = W_stack.shape
    Zt = torch.from_numpy(Z_val).to(torch.float64)
    yt = torch.from_numpy(y_val).to(torch.long)
    Wt = torch.from_numpy(W_stack).to(torch.float64)   # [K,C,D]
    bt = torch.from_numpy(b_stack).to(torch.float64)   # [K,C]
    Mt = torch.from_numpy(M).to(torch.float64)         # [K,K]
    U  = torch.full((K, C), 1.0/float(K), dtype=torch.float64)

    if S_idx is not None:
        St = torch.from_numpy(np.asarray(S_idx, dtype=np.int64))
    else:
        St = None

    phi = torch.zeros(K, C, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.Adam([phi], lr=lr)

    best_obj, best_A = float('inf'), None
    for t in range(1, steps+1):
        A = torch.softmax(phi, dim=0)                 # [K,C]
        # 合成头
        Wc = torch.einsum('kc,kcd->cd', A, Wt)       # [C,D]
        bc = torch.einsum('kc,kc->c', A, bt)         # [C]
        logits = Zt @ Wc.t() + bc.unsqueeze(0)       # [N,C]
        if St is not None:
            ce = F.cross_entropy(logits.index_select(0, St), yt.index_select(0, St))
        else:
            ce = F.cross_entropy(logits, yt)
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
    return A_np  # [K,C]

@torch.no_grad()
def _replace_last_linear(model, last_name, W_mix, b_mix):
    last = _get_module_by_name(model, last_name)
    last.weight.copy_(torch.from_numpy(W_mix).to(last.weight.dtype).to(last.weight.device))
    last.bias.copy_(  torch.from_numpy(b_mix).to(last.bias.dtype).to(last.bias.device))

@torch.no_grad()
def _eval_function_ensemble_from_alpha_bar(snapshots, alpha_bar, loader, build_model_fn, device, mode='linear'):
    """
    与你原先的 _eval_function_ensemble 等价，但只支持 ᾱ[K]，并保证每个快照遍历前固定增广 RNG（若 loader 为 val）。
    Test loader 通常无增广，此处不强制设 RNG。
    """
    K = len(snapshots)
    alpha_bar = np.asarray(alpha_bar, dtype=np.float64)
    mix_accum = None
    labels_all = []
    for j in range(K):
        # 如果是验证集（带增广），我们固定 RNG；测试集（通常无增广）不强制。
        try:
            _fix_aug_rng()
        except Exception:
            pass
        m = build_model_fn(); m.load_state_dict(snapshots[j], strict=True)
        m.to(device).eval()
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
