# -*- coding: utf-8 -*-
"""v6 训练、均衡采样、源验证记录、记录级评估与 checkpoint。

由 v5 `trainer.py` 迁移（见 `docs/v5复用与迁移清单.md` §2）：删 `train_source_ce_target_bn`
与 `train_target_only`；`train_adapt` 去掉中值带宽分支（固定核唯一）；
`save_checkpoint` 无条件写 `checkpoint_kind`/`checkpoint_epoch`（v5 缺 101 个的教训）。
保留 `train_source`（`source_continue` 的引擎）与 `train_adabn`（无标注 BN 控制）。

要点：内置权重做四步均衡；固定 S_ref 预算保证各臂同更新数；每轮记录源验证 DE/FE
记录级 Macro-F1 均值，但 v6 正式权重恒取末轮 `last.pt`（selection_rule=last，源验证只记录不择优）；
记录内做概率平均；checkpoint 用 schema_version=1 且不静默覆盖。
"""
from copy import deepcopy
import csv
import json
import math
import random
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from dataset import LABEL_NAMES
from model import classwise_mmd_loss

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SCHEMA_VERSION = 1  # v5/v6 共用同一 schema，读 v5 权重走 load_checkpoint 这一条路。


def seed_everything(seed):
    """固定 Python、NumPy、PyTorch 与 cuDNN 随机性，保证同 seed 可重复。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _feature_len(feature):
    # 统一取特征集窗口数；空集由调用方明确报错，不返回假长度。
    return int(len(feature.y))


def balanced_window_weights(feature):
    """返回 1/(R[channel,class] * W[record]) 的每窗口权重。

    四步均衡：先按通道分开，再按通道内类别均衡，再按记录均衡，最后记录内窗口均匀抽取。
    统计只来自当前传入的训练特征集；源域含四类，目标域只对实际存在的三类故障均衡，不补造 Normal。
    """
    if _feature_len(feature) == 0:
        raise ValueError("空特征集不能计算采样权重")
    keys = np.asarray(feature.record_keys, dtype=object)
    channels = np.asarray(feature.channels, dtype=object)
    labels = feature.y.detach().cpu().numpy()
    record_pairs = {(str(channel), str(key), int(label))
                    for channel, key, label in zip(channels, keys, labels)}
    record_counts = {}
    window_counts = {}
    for channel, key, label in record_pairs:
        group = (str(channel), int(label))
        record_counts[group] = record_counts.get(group, 0) + 1
        record_key = (str(channel), str(key))
        window_counts[record_key] = int(np.sum((channels == channel) & (keys == key)))
    weights = np.empty(len(labels), dtype=np.float64)
    for index, (channel, key, label) in enumerate(zip(channels, keys, labels)):
        group = (str(channel), int(label))
        record_key = (str(channel), str(key))
        weights[index] = 1.0 / (record_counts[group] * window_counts[record_key])
    return weights


def weight_totals(feature):
    """返回每个 channel×class 和每个 channel×record 的总权重，供验收使用。"""
    weights = balanced_window_weights(feature)
    by_class, by_record = {}, {}
    for weight, key, channel, label in zip(weights, feature.record_keys, feature.channels,
                                           feature.y.tolist()):
        by_class[(str(channel), int(label))] = by_class.get((str(channel), int(label)), 0.0) + weight
        by_record[(str(channel), str(key))] = by_record.get((str(channel), str(key)), 0.0) + weight
    return by_class, by_record


def make_weighted_loader(feature, batch_size, num_samples, seed):
    """用 WeightedRandomSampler 构造均衡采样器，返回 (DataLoader, 权重)。

    有放回采样才能压平长记录与多数类；配了 sampler 不再配 shuffle；
    drop_last=True 保证每步形状一致。
    """
    if _feature_len(feature) == 0:
        raise ValueError("空特征集不能创建采样器")
    if batch_size <= 0 or num_samples <= 0:
        raise ValueError("batch_size 和 num_samples 必须为正数")
    weights = torch.as_tensor(balanced_window_weights(feature), dtype=torch.double)
    sampler = WeightedRandomSampler(
        weights, num_samples=int(num_samples), replacement=True,
        generator=torch.Generator().manual_seed(int(seed)))
    data = TensorDataset(feature.X, feature.y)
    return DataLoader(data, batch_size=batch_size, sampler=sampler, drop_last=True), weights


def steps_per_epoch(s_ref, batch_size):
    """返回固定步数 steps_per_epoch=ceil(S_ref/batch_size)。

    S_ref 只取源训练 DE 窗口数（512 点口径），与当前训练的是 DE/FE 还是 joint 无关；
    源续训与两个适配臂共用同一 S_ref，epochs 相同时总更新数相同。
    """
    if s_ref <= 0 or batch_size <= 0:
        raise ValueError("s_ref 和 batch_size 必须为正数")
    return int(math.ceil(s_ref / batch_size))


def _record_groups(feature):
    # 记录身份是"文件键+通道"二元组：同一文件的 DE/FE 是两条独立记录。
    groups = {}
    for index, (key, channel) in enumerate(zip(feature.record_keys, feature.channels)):
        groups.setdefault((str(key), str(channel)), []).append(index)
    return groups


def _class_metrics(true, pred, n_classes=4):
    """按记录级预测计算 Accuracy、Macro-F1、逐类召回、4x4 混淆与样本数。

    目标测试没有 Normal，该类召回/F1 记 None（JSON null），Macro-F1 只对有样本类平均；
    故障被判成 Normal 仍计错，不删除也不改判。
    """
    true = np.asarray(true, dtype=np.int64)
    pred = np.asarray(pred, dtype=np.int64)
    confusion = np.zeros((n_classes, n_classes), dtype=np.int64)
    for actual, guessed in zip(true, pred):
        confusion[actual, guessed] += 1
    support = confusion.sum(axis=1)
    recall, f1 = [], []
    for label in range(n_classes):
        tp = float(confusion[label, label])
        fp = float(confusion[:, label].sum() - confusion[label, label])
        fn = float(confusion[label, :].sum() - confusion[label, label])
        recall.append(None if support[label] == 0 else tp / support[label])
        denominator = 2 * tp + fp + fn
        f1.append(None if support[label] == 0 else (2 * tp / denominator if denominator else 0.0))
    valid_f1 = [value for value in f1 if value is not None]
    return {
        "accuracy": float(np.mean(true == pred)) if len(true) else 0.0,
        "macro_f1": float(np.mean(valid_f1)) if valid_f1 else None,
        "recall": recall,
        "f1": f1,
        "support": support.astype(int).tolist(),
        "confusion_matrix": confusion.tolist(),
    }


def evaluate_feature(model, feature, batch_size=256):
    """记录内概率平均后计算同一套指标（model.eval() + no_grad，测试前向不更新 BN）。"""
    if _feature_len(feature) == 0:
        raise ValueError("评估特征集为空")
    model.eval()
    probabilities = []
    with torch.no_grad():
        for start in range(0, len(feature.X), batch_size):
            logits = model(feature.X[start:start + batch_size].to(DEVICE))[0]
            if not torch.isfinite(logits).all():
                raise ValueError("评估模型输出包含非有限值，不能生成诊断指标")
            batch_probabilities = torch.softmax(logits, dim=1)
            if not torch.isfinite(batch_probabilities).all():
                raise ValueError("评估概率包含非有限值，不能生成诊断指标")
            probabilities.append(batch_probabilities.cpu())
    window_probs = torch.cat(probabilities).numpy()
    records = _record_groups(feature)
    record_probs, record_true, record_keys, record_window_counts = [], [], [], []
    for (key, channel), indices in records.items():
        record_probs.append(window_probs[indices].mean(axis=0))
        record_true.append(int(feature.y[indices[0]]))
        record_keys.append(f"{key}::{channel}")
        record_window_counts.append(len(indices))
    record_probs = np.asarray(record_probs)
    record_true = np.asarray(record_true)
    record_pred = record_probs.argmax(axis=1)
    metrics = _class_metrics(record_true, record_pred)
    metrics.update({
        "record_keys": record_keys,
        "predictions": record_pred.astype(int).tolist(),
        "probabilities": record_probs.tolist(),
        "true": record_true.astype(int).tolist(),
        "window_counts": [int(value) for value in record_window_counts],
        "window_count": len(feature.X),
    })
    return metrics


def select_validation_score(metrics_by_channel):
    """按 DE/FE 记录级 Macro-F1 平均给分，平局再比较 Accuracy（v6 只记录，不据此择优）。"""
    required = ("DE", "FE")
    if any(channel not in metrics_by_channel for channel in required):
        raise ValueError("源验证必须同时提供 DE 和 FE 指标")
    f1_values = [metrics_by_channel[channel]["macro_f1"] for channel in required]
    acc_values = [metrics_by_channel[channel]["accuracy"] for channel in required]
    if any(value is None for value in f1_values):
        raise ValueError("源验证必须包含四类真实样本")
    return float(np.mean(f1_values)), float(np.mean(acc_values))


def _save_progress(path, best, history, updates, last=None):
    """保留已完成轮次，原子替换；中断后仍须整目录重跑，不作断点续训。"""
    if path is not None:
        path = Path(path)
        temporary = path.with_suffix(".tmp")
        torch.save({"best": best, "history": history, "updates": updates, "last": last}, temporary)
        temporary.replace(path)


def train_source(model, train_feature, val_features, batch_size=64, epochs=30,
                 lr=1e-3, seed=0, s_ref=None, progress_path=None):
    """固定步数的源 CE 训练；每轮记录源验证，跑满不早停。

    v6 用它跑 `source_continue`（源续训控制）：起点为同折同种子的 joint `last.pt`，
    只喂源域，不接触任何目标数据。`best` 仍是源验证最优（保留原义），
    正式权重由 train.py 按 `selection_rule=last` 取 `last`。
    """
    if s_ref is None:
        raise ValueError("必须显式提供本折源 DE 窗口数 s_ref")
    s_ref = int(s_ref)
    steps = steps_per_epoch(s_ref, batch_size)
    loader, _ = make_weighted_loader(train_feature, batch_size, steps * batch_size, seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best = None
    last = None
    history = []
    for epoch in range(1, epochs + 1):
        started = perf_counter()
        model.train()
        ce_sum = 0.0
        for xb, yb in loader:
            logits = model(xb.to(DEVICE))[0]
            loss = F.cross_entropy(logits, yb.to(DEVICE))
            if not torch.isfinite(loss):
                raise FloatingPointError(f"源训练第 {epoch} 轮损失非有限，停止更新")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            ce_sum += float(loss.item())
        val_metrics = {channel: evaluate_feature(model, feature) for channel, feature in val_features.items()}
        score, accuracy = select_validation_score(val_metrics)
        row = {"epoch": epoch, "ce_source": ce_sum / steps,
               "val_macro_f1_mean": score, "val_accuracy_mean": accuracy,
               "val_macro_f1_DE": val_metrics["DE"]["macro_f1"],
               "val_accuracy_DE": val_metrics["DE"]["accuracy"],
               "val_macro_f1_FE": val_metrics["FE"]["macro_f1"],
               "val_accuracy_FE": val_metrics["FE"]["accuracy"]}
        history.append(row)
        if best is None or (score, accuracy) > (best["score"], best["accuracy"]):
            best = {"score": score, "accuracy": accuracy, "epoch": epoch,
                    "state_dict": deepcopy(model.state_dict()), "metrics": val_metrics}
        last = {"epoch": epoch, "state_dict": deepcopy(model.state_dict()),
                "score": score, "accuracy": accuracy, "metrics": val_metrics}
        _save_progress(progress_path, best, history, steps * epoch, last)
        print(f"源训练 {epoch}/{epochs}：CE={row['ce_source']:.4f}，源验证 F1={score:.4f}，"
              f"Accuracy={accuracy:.4f}，best={best['epoch']}，耗时={perf_counter() - started:.1f}秒", flush=True)
    model.load_state_dict(best["state_dict"])
    return {"model": model, "best": best, "last": last, "history": history,
            "steps_per_epoch": steps, "updates": steps * epochs}


def train_source_continue(model, train_feature, val_features, batch_size=64, epochs=30,
                          lr=1e-4, seed=0, s_ref=None, progress_path=None):
    """仅源 CE 续训控制，与适配臂共用同一 S_ref/steps/epochs/lr/bs/seed。

    为什么：从同一 joint 末轮出发只多做源 CE 更新，用来剥离"额外步数本身"带来的变化；
    `CE − 源续训` 才是标签本身值多少。它只对齐更新步数，不声称匹配全部 FLOPs。
    """
    return train_source(model, train_feature, val_features, batch_size, epochs, lr, seed, s_ref, progress_path)


def train_adapt(model, source_feature, target_feature, val_features, batch_size=64,
                epochs=30, lr=1e-4, lambda_mmd=0.0, seed=0, s_ref=None, progress_path=None):
    """源 CE + 目标 CE（+ 共有真实故障类 MMD）；源/目标各自均衡采样。

    固定核 sigma=(1,5,10)、biased 估计、只对 Ball/IR/OR 逐类等权平均、
    任一域不足 2 样本跳类、无有效类取零且不回退全局；lambda_mmd=0 即 CE 臂。
    目标一次前向同时返回 logits 与特征，CE 与 MMD 共用，不额外前向；
    优化器新建（Adam 1e-4），不继承源阶段状态；每轮记录源验证，正式取末轮。
    """
    if len(target_feature.X) == 0:
        raise ValueError("目标适配特征为空")
    if s_ref is None:
        raise ValueError("必须显式提供本折源 DE 窗口数 s_ref")
    s_ref = int(s_ref)
    steps = steps_per_epoch(s_ref, batch_size)
    source_loader, _ = make_weighted_loader(source_feature, batch_size, steps * batch_size, seed)
    target_loader, _ = make_weighted_loader(target_feature, batch_size, steps * batch_size, seed + 1)
    target_iterator = iter(target_loader)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best, last, history = None, None, []
    for epoch in range(1, epochs + 1):
        started = perf_counter()
        model.train()
        totals = [0.0, 0.0, 0.0]
        for xb_s, yb_s in source_loader:
            try:
                xb_t, yb_t = next(target_iterator)
            except StopIteration:
                target_iterator = iter(target_loader)
                xb_t, yb_t = next(target_iterator)
            logits_s, _, feat_s = model(xb_s.to(DEVICE), return_feat=True)
            logits_t, _, feat_t = model(xb_t.to(DEVICE), return_feat=True)
            yb_s, yb_t = yb_s.to(DEVICE), yb_t.to(DEVICE)
            ce_source = F.cross_entropy(logits_s, yb_s)
            ce_target = F.cross_entropy(logits_t, yb_t)
            # 普通 CE，不叠逆频率类别权重：均衡已由两域采样器承担。
            if lambda_mmd:
                mmd = classwise_mmd_loss(feat_s, yb_s, feat_t, yb_t)
            else:
                mmd = feat_s.new_zeros(())
            loss = ce_source + ce_target + float(lambda_mmd) * mmd
            if not torch.isfinite(loss):
                raise FloatingPointError(f"适配第 {epoch} 轮损失非有限，停止更新")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            totals[0] += float(ce_source.item())
            totals[1] += float(ce_target.item())
            totals[2] += float(mmd.item())
        val_metrics = {channel: evaluate_feature(model, feature) for channel, feature in val_features.items()}
        score, accuracy = select_validation_score(val_metrics)
        history.append({"epoch": epoch, "ce_source": totals[0] / steps,
                        "ce_target": totals[1] / steps, "mmd": totals[2] / steps,
                        "val_macro_f1_mean": score, "val_accuracy_mean": accuracy,
                        "val_macro_f1_DE": val_metrics["DE"]["macro_f1"],
                        "val_accuracy_DE": val_metrics["DE"]["accuracy"],
                        "val_macro_f1_FE": val_metrics["FE"]["macro_f1"],
                        "val_accuracy_FE": val_metrics["FE"]["accuracy"]})
        if best is None or (score, accuracy) > (best["score"], best["accuracy"]):
            best = {"score": score, "accuracy": accuracy, "epoch": epoch,
                    "state_dict": deepcopy(model.state_dict()), "metrics": val_metrics}
        last = {"epoch": epoch, "state_dict": deepcopy(model.state_dict()),
                "score": score, "accuracy": accuracy, "metrics": val_metrics}
        _save_progress(progress_path, best, history, steps * epoch, last)
        print(f"适配 {epoch}/{epochs}：源CE={totals[0] / steps:.4f}，目标CE={totals[1] / steps:.4f}，"
              f"MMD={totals[2] / steps:.4f}，源验证 F1={score:.4f}，best={best['epoch']}，"
              f"耗时={perf_counter() - started:.1f}秒", flush=True)
    model.load_state_dict(best["state_dict"])
    return {"model": model, "best": best, "last": last, "history": history,
            "steps_per_epoch": steps, "updates": steps * epochs}


def _set_bn_train_only(model):
    """整体置 eval 后仅 BN 层置 train，供纯 AdaBN 使用（Dropout 保持关闭）。"""
    model.eval()
    for module in model.modules():
        if isinstance(module, nn.BatchNorm1d):
            module.train()


def _check_bn_momentum(model, expected=0.1):
    """检查全部 BN 层 momentum 是否为既定值，不符直接报错（沿用既有值，不重置）。"""
    found = False
    for module in model.modules():
        if isinstance(module, nn.BatchNorm1d):
            found = True
            if float(module.momentum) != float(expected):
                raise ValueError(f"BN momentum 必须为 {expected}，实际为 {module.momentum}")
    if not found:
        raise ValueError("模型中没有 BN 层，无法执行 BN 控制")


def _learnable_state_snapshot(model):
    # 快照全部 state；running_mean/running_var/num_batches_tracked 是允许变化的缓冲区。
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def _assert_only_bn_buffers_changed(before, model):
    """断言除 BN 缓冲区外所有参数不变，否则报错（AdaBN 不允许有权重更新）。"""
    after = model.state_dict()
    changed = []
    for key, old in before.items():
        if key.endswith(("running_mean", "running_var", "num_batches_tracked")):
            continue
        if not torch.equal(after[key].detach().cpu(), old):
            changed.append(key)
    if changed:
        raise AssertionError(f"AdaBN 不应改变可学习参数，变化键：{changed[:8]}")


def train_adabn(model, target_feature, val_features, batch_size=64, epochs=30,
                seed=0, s_ref=None, progress_path=None):
    """纯 AdaBN：仅对目标标注池做无梯度前向更新 BN 运行统计，不做任何参数更新。

    目标采样器与同折同种子的 CE 共用 seed+1，目标前向批次数 steps×epochs 与 CE 相同，
    但不称等优化步数（`updates` 恒 0）；每轮源验证只记录描述，正式取末轮；
    结束校验可学习参数逐键不变。
    """
    if len(target_feature.X) == 0:
        raise ValueError("目标适配特征为空")
    if s_ref is None:
        raise ValueError("必须显式提供本折源 DE 窗口数 s_ref")
    s_ref = int(s_ref)
    steps = steps_per_epoch(s_ref, batch_size)
    _check_bn_momentum(model)
    before = _learnable_state_snapshot(model)
    target_loader, _ = make_weighted_loader(target_feature, batch_size, steps * batch_size, seed + 1)
    target_iterator = iter(target_loader)
    best, last, history, forwards = None, None, [], 0
    for epoch in range(1, epochs + 1):
        started = perf_counter()
        _set_bn_train_only(model)
        with torch.no_grad():
            for _ in range(steps):
                try:
                    xb_t, _ = next(target_iterator)
                except StopIteration:
                    target_iterator = iter(target_loader)
                    xb_t, _ = next(target_iterator)
                model(xb_t.to(DEVICE))
                forwards += 1
        val_metrics = {channel: evaluate_feature(model, feature) for channel, feature in val_features.items()}
        score, accuracy = select_validation_score(val_metrics)
        history.append({"epoch": epoch, "target_forward_batches": steps,
                        "target_forward_batches_cum": forwards,
                        "val_macro_f1_mean": score, "val_accuracy_mean": accuracy,
                        "val_macro_f1_DE": val_metrics["DE"]["macro_f1"],
                        "val_accuracy_DE": val_metrics["DE"]["accuracy"],
                        "val_macro_f1_FE": val_metrics["FE"]["macro_f1"],
                        "val_accuracy_FE": val_metrics["FE"]["accuracy"]})
        if best is None or (score, accuracy) > (best["score"], best["accuracy"]):
            best = {"score": score, "accuracy": accuracy, "epoch": epoch,
                    "state_dict": deepcopy(model.state_dict()), "metrics": val_metrics}
        last = {"epoch": epoch, "state_dict": deepcopy(model.state_dict()),
                "score": score, "accuracy": accuracy, "metrics": val_metrics}
        _save_progress(progress_path, best, history, 0, last)
        print(f"AdaBN {epoch}/{epochs}：目标前向累计 {forwards} 批，源验证 F1={score:.4f}，"
              f"耗时={perf_counter() - started:.1f}秒", flush=True)
    model.eval()
    _assert_only_bn_buffers_changed(before, model)
    model.load_state_dict(best["state_dict"])
    return {"model": model, "best": best, "last": last, "history": history,
            "steps_per_epoch": steps, "updates": 0, "target_forwards": forwards}


def save_checkpoint(path, model, config):
    """保存 schema_version=1 的权重与配置；已存在则报错不覆盖。

    v6 无条件写 `checkpoint_kind`（文件名推 best/last）与 `checkpoint_epoch`，
    避免 v5 那种"只能靠文件名推断末轮"的 101 个流程。
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"不覆盖已有 checkpoint：{path}")
    payload = {"schema_version": SCHEMA_VERSION, "state_dict": model.state_dict(), **config}
    if path.name in ("best.pt", "last.pt"):
        kind = path.stem
        payload["checkpoint_kind"] = kind
        payload["checkpoint_epoch"] = config[f"{kind}_epoch"]
    torch.save(payload, path)


def load_checkpoint(path, model):
    """校验 schema_version 后加载权重，返回 checkpoint 配置。

    v6 只按状态字典加载 v5 的 `joint` `last.pt`，其余配置字段由 train.py 逐条核对。
    """
    payload = torch.load(path, map_location=DEVICE, weights_only=False)
    if payload.get("schema_version") != SCHEMA_VERSION or "state_dict" not in payload:
        raise ValueError("checkpoint 版本或内容不匹配")
    model.load_state_dict(payload["state_dict"])
    return payload


def write_history(path, history):
    """把每轮训练记录写成 history.csv；已存在则报错不覆盖。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"不覆盖已有 history：{path}")
    if not history:
        return
    fields = sorted({key for row in history for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(history)


def dump_json(path, value):
    """把配置或指标写成 UTF-8 JSON；已存在则报错不覆盖。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"不覆盖已有 JSON：{path}")
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def _self_check():
    from types import SimpleNamespace
    feature = SimpleNamespace(
        X=torch.zeros(8, 1, 256), y=torch.tensor([0, 0, 1, 1, 2, 2, 3, 3]),
        record_keys=np.array(["a", "a", "b", "b", "c", "c", "d", "d"], dtype=object),
        channels=np.array(["DE"] * 8, dtype=object),
    )
    totals = weight_totals(feature)
    assert set(totals[0].values()) == {1.0}
    assert set(totals[1].values()) == {1.0}
    assert steps_per_epoch(9983, 64) == math.ceil(9983 / 64) == 156
    metrics = _class_metrics([1, 2, 3], [1, 2, 3])
    assert metrics["macro_f1"] == 1.0 and metrics["recall"][0] is None
    print("ALL TRAINER CHECKS PASSED")


if __name__ == "__main__":
    _self_check()
