# -*- coding: utf-8 -*-
"""v6 表征构建：分段信号 → logspec512 特征，直接计算，不建缓存。

由 v5 `preprocessing.py` 迁移（见 `docs/v5复用与迁移清单.md` §2），做减法：
`REPRESENTATIONS` 收成 `("logspec512",)`；删除 `logspec4096` 与包络全链路
（`ENVELOPE_BANDS`/`ENVELOPE_FILTER_*`/`select_envelope_band`/`apply_bandpass`/
`envelope_spectrum`/`build_envelope_feature_set`）以及 `predict_features`（本轮无预测交付）。
`build_repr_config` 仍输出 `band/filter/scipy_version=None` 三个空字段，
唯一理由是 v5 源 checkpoint 的 `repr_config` 逐字段含它们，`config_from_saved`
要能原样校验通过并复用 v5 的 `joint` 权重。

口径（与 v5 完全相同，冻结）：512 点窗口 → 乘 Hanning → rFFT 取幅度 + 1e-9 →
在 0～6000 Hz 全频段用 `linspace(0, 末索引, 256).astype(int)` 等距取 256 点 →
取 log → 逐窗口跨频标准化（eps=1e-9）→ (N,1,256) float32。

报错约定：空信号、非有限值、不足一个窗口、形状不符均直接报错，不填充修复。
"""
from dataclasses import dataclass

import numpy as np
import torch

#: v6 单表征协议：只允许 logspec512。
REPRESENTATIONS = ("logspec512",)
#: 统一采样率（Hz）；48k 来源在 dataset.py 里已降到该采样率。
TARGET_FS = 12000


@dataclass
class FeatureSet:
    X: torch.Tensor
    y: torch.Tensor
    record_keys: np.ndarray
    channels: np.ndarray


def build_repr_config(seg_len=512, nbin=256):
    """构造 logspec512 表征配置；窗长只允许 512，分箱固定 256。

    返回可 JSON 序列化的配置字典，`input_dim=256`、`n_channels=1` 供模型构造输入层。
    """
    seg_len, nbin = int(seg_len), int(nbin)
    if seg_len != 512 or nbin != 256:
        raise ValueError("v6 表征固定 logspec512：seg_len=512、nbin=256")
    df = TARGET_FS / seg_len
    return {
        "representation": "logspec512",
        "repr_kind": "logspec",
        "seg_len": seg_len,
        "window": "hanning",
        "nbin": 256,
        "log_eps": 1e-9,
        "std_eps": 1e-9,
        "seg_std": True,
        "n_channels": 1,
        "input_dim": 256,
        "band": None,
        "filter": None,
        "scipy_version": None,
        "fs": TARGET_FS,
        "freq": {"df": float(df), "bins": 256, "mode": "sparse_fullband", "end_hz": float(TARGET_FS / 2)},
    }


def config_from_saved(saved, require_complete=False):
    """恢复表征配置；旧 512 简写格式允许补默认，新配置逐字段校验。

    v6 只保留 logspec512 一条路径：`require_complete` 仍供正式 checkpoint 拒绝缺配置的产物，
    但不再区分包络的"未选带草稿"。
    """
    if saved is None:
        saved = {}
    if not isinstance(saved, dict):
        raise ValueError("保存的表征配置必须是字典")
    saved = dict(saved)
    legacy = set(saved) <= {"seg_len", "input_dim", "n_channels"}
    if legacy:
        if require_complete:
            raise ValueError("新 checkpoint 缺少完整预处理配置")
        expected = {"seg_len": 512, "input_dim": 256, "n_channels": 1}
        if any(value != expected[key] for key, value in saved.items()):
            raise ValueError("旧格式只兼容 512 点、256 维、单通道表征")
        return build_repr_config()
    if "representation" not in saved:
        raise ValueError("新表征配置缺少 representation，不能按旧格式补默认")
    if "seg_len" not in saved or "nbin" not in saved:
        raise ValueError("新表征配置缺少窗长或频点数量")
    restored = build_repr_config(saved["seg_len"], saved["nbin"])
    for name, value in restored.items():
        if name not in saved or saved[name] != value:
            raise ValueError(f"保存的表征配置缺失或与当前实现、环境不一致：{name}")
    return {**saved, **restored}


def representation_of(config):
    """返回配置的表征名；缺省按 logspec512 处理。"""
    if config is None:
        return "logspec512"
    name = config.get("representation")
    if name is None:
        return "logspec512"
    if name not in REPRESENTATIONS:
        raise ValueError(f"未知表征：{name}")
    return name


def _shape_std(values, eps=1e-9):
    """逐窗口跨频带标准化：(N,D)→(N,D)，每个窗口独立做 z-score。

    按窗口做而不跨窗口共享均值方差：每个窗口的能量量级不同，跨窗口统计会把
    大能量窗口的幅值差异带进所有样本；窗口内标准化只保留窗内相对谱形。
    """
    values = np.asarray(values, dtype=np.float64)
    return (values - values.mean(axis=1, keepdims=True)) / (values.std(axis=1, keepdims=True) + eps)


def extract_logspec(signals, config=None):
    """logspec 特征提取：分段信号 (N,512) → (N,1,256) float32。

    Hanning → rFFT 幅度 + log_eps → 全频段等距取 256 点（含端点、int 截断）→
    log → 窗口内标准化。形状不符或含非有限值直接报错。
    """
    cfg = config or build_repr_config()
    representation_of(cfg)
    seg_len = int(cfg["seg_len"])
    signals = np.asarray(signals, dtype=np.float64)
    if signals.ndim != 2 or signals.shape[1] != seg_len:
        raise ValueError(f"signals 必须为 (N,{seg_len})")
    if signals.size and not np.isfinite(signals).all():
        raise ValueError("signals 包含非有限值")
    amp = np.abs(np.fft.rfft(signals * np.hanning(seg_len), axis=1)) + float(cfg["log_eps"])
    pick = np.linspace(0, amp.shape[1] - 1, int(cfg["nbin"])).astype(int)
    selected = amp[:, pick]
    features = _shape_std(np.log(selected), float(cfg["std_eps"])).astype(np.float32)
    if not np.isfinite(features).all():
        raise ValueError("对数谱计算产生非有限特征，拒绝预测或训练")
    return features[:, None, :]


def build_feature_set(segments, file_keys, channel, config=None):
    """从指定文件键和单通道分段构建 logspec 特征，并保留每行的记录身份。

    保留 record_keys/channels 是为了记录级评估按"同文件同通道"做窗口概率平均；
    一次调用只处理一个通道，DE+FE 对照靠调用方分别构建后 `merge_feature_sets` 合并。
    """
    if channel not in ("DE", "FE"):
        raise ValueError(f"未知通道：{channel}")
    cfg = config or build_repr_config()
    keys = list(file_keys)
    parts, labels, record_keys, channels = [], [], [], []
    for key in keys:
        span = segments.span(key)
        signals = segments.signals[span]
        parts.append(extract_logspec(signals, cfg))
        labels.extend(segments.labels[span].tolist())
        record_keys.extend([str(key)] * len(signals))
        channels.extend([channel] * len(signals))
    if not parts:
        return FeatureSet(
            torch.empty((0, 1, cfg["input_dim"]), dtype=torch.float32),
            torch.empty((0,), dtype=torch.long),
            np.asarray([], dtype=object), np.asarray([], dtype=object),
        )
    return FeatureSet(
        torch.from_numpy(np.concatenate(parts, axis=0)),
        torch.as_tensor(labels, dtype=torch.long),
        np.asarray(record_keys, dtype=object),
        np.asarray(channels, dtype=object),
    )


def build_record_features(records, file_keys, channel, config=None):
    """训练/评估统一预处理入口：切窗 → logspec 特征。

    训练、评估必须用同一入口；v6 只有 logspec 一条路径，不再分发包络。
    """
    import dataset as ds
    cfg = config or build_repr_config()
    segments = ds.build_segments(records, list(file_keys), channel, segment_length=int(cfg["seg_len"]))
    return build_feature_set(segments, list(file_keys), channel, cfg)


def merge_feature_sets(feature_sets):
    """沿样本维合并多个单通道 FeatureSet（DE+FE 作独立单通道样本）。"""
    feature_sets = list(feature_sets)
    if not feature_sets:
        raise ValueError("至少需要一个 FeatureSet")
    return FeatureSet(
        torch.cat([item.X for item in feature_sets], dim=0),
        torch.cat([item.y for item in feature_sets], dim=0),
        np.concatenate([item.record_keys for item in feature_sets]),
        np.concatenate([item.channels for item in feature_sets]),
    )


def _self_check():
    """模块自检：校验 logspec 形状与窗口内标准化均值≈0，不依赖外部数据。"""
    rng = np.random.default_rng(0)
    signals = rng.standard_normal((4, 512))
    features = extract_logspec(signals)
    assert features.shape == (4, 1, 256)
    assert np.max(np.abs(features.mean(axis=2))) < 1e-5
    # v5 源 checkpoint 的 repr_config 必须能原样通过逐字段校验（先查仓库自带的镜像，再查 v5 原目录）。
    from pathlib import Path
    import json
    here = Path(__file__).resolve().parent
    candidates = [here / "weights/v5_joint/f0_s0_joint/config.json",
                  here.parent / "rbop_diagnosis_v5/outputs/stage1_v2_round2/train/f0_s0_joint/config.json"]
    v5_cfg = next((path for path in candidates if path.is_file()), None)
    if v5_cfg is not None:
        saved = json.loads(v5_cfg.read_text(encoding="utf-8"))["repr_config"]
        restored = config_from_saved(saved, require_complete=True)
        assert restored["representation"] == "logspec512"
        print(f"  表征配置回读校验通过：{v5_cfg.name}（{v5_cfg.parent.name}）")
    else:
        print("  跳过表征配置回读（未找到 v5 joint config.json）")
    print(f"  数据根：{__import__('dataset').default_data_root()}")
    print("ALL PREPROCESSING CHECKS PASSED")


if __name__ == "__main__":
    _self_check()
