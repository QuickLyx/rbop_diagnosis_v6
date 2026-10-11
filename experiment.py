# -*- coding: utf-8 -*-
"""v6 数据准备：源四折特征、S_ref 参考预算与按冻结池构建的目标适配特征。

由 v5 `experiment.py` 迁移（见 `docs/v5复用与迁移清单.md` §2）：
删 `prepare_target`（v1 配额档）、`prepare_target_v2`、`split_payload`、`split_payload_v2`
与包络选带分支（`resolve_source_allowed_channels` 的通道参数一并收窄为 joint），
新增 `prepare_target_v6` 与 `split_payload`（v6 语义：目标部分读冻结池）。

要点：一次只准备一个 fold 的源训练+验证特征（验证恒为 DE/FE 两份）；
目标只给池内文件建特征，`test` 只保留文件列表不读信号；
split.json 只存文件列表与种子；S_ref 为本折源训练 DE 窗口数（512 点口径）。
"""
from dataclasses import dataclass
import json
from pathlib import Path

import dataset as ds
import preprocessing as pp

#: 参考窗长：S_ref 恒按 512 点口径统计，与实际表征窗长解耦。
REFERENCE_SEG_LEN = 512


def reference_s_ref(records, train_keys, segment_length=REFERENCE_SEG_LEN):
    """按原始信号统计源训练 DE 通道的参考窗口数，不依赖特征窗长。"""
    by_key = {record["file_key"]: record for record in records}
    total = 0
    for key in list(train_keys):
        record = by_key.get(str(key))
        if record is None:
            raise KeyError(f"未知文件键：{key}")
        signal = ds.read_record_signal(record, "DE")
        n_windows = len(signal) // int(segment_length)
        if n_windows < 1:
            raise ValueError(f"文件不足一个参考窗口：{key}::DE")
        total += int(n_windows)
    if total <= 0:
        raise ValueError("参考 S_ref 必须为正数")
    return int(total)


@dataclass
class SourceData:
    # 一次 fold 的源数据容器：records 为全部 56 个源文件记录；fold 为选中折的文件键；
    # train 为源训练特征（DE+FE 沿样本维合并）；val 恒为 {"DE":…, "FE":…}；
    # s_ref 为本折源训练 DE 通道按 512 点口径的参考窗口数；repr_config 为实际表征配置。
    records: list
    fold: dict
    train: pp.FeatureSet
    val: dict
    s_ref: int
    repr_config: dict = None


def _channel_features(records, keys, channel, repr_config=None):
    # 单通道特征统一入口：v6 只有 logspec 一条路径。
    cfg = pp.config_from_saved(repr_config) if repr_config is not None else pp.build_repr_config()
    return pp.build_record_features(records, list(keys), channel, cfg)


def prepare_source(data_root=None, fold=0, split_seed=0, repr_config=None, build_train=True):
    """构建一个 fold 的源训练（DE+FE 合并）和 DE/FE 验证特征。

    `build_train=False` 时 train 为 None，仍计算 512 点参考预算并构建描述性验证特征
    （adabn 只用验证描述，不做源重放）。通道恒为 joint，因为 v6 不重训源模型、
    所有控制与适配都从 joint 起点出发。
    """
    records = ds.enumerate_source_files(data_root)
    folds = ds.split_source_files(records, split_seed)
    if not 0 <= fold < len(folds):
        raise ValueError(f"fold 必须在 0..3 内：{fold}")
    selected = folds[fold]
    cfg = pp.config_from_saved(repr_config) if repr_config is not None else pp.build_repr_config()
    train = None
    if build_train:
        parts = [_channel_features(records, selected["train"], channel, cfg) for channel in ("DE", "FE")]
        train = pp.merge_feature_sets(parts)
    val = {
        "DE": _channel_features(records, selected["val"], "DE", cfg),
        "FE": _channel_features(records, selected["val"], "FE", cfg),
    }
    return SourceData(records, selected, train, val, reference_s_ref(records, selected["train"]), cfg)


def prepare_target_v6(data_root=None, target_source=None, selection_path=None, repr_config=None):
    """只给冻结标注池内文件建 DE/FE 适配特征，test 列表不读信号。

    输入：数据根、目标来源、`selection.json` 路径、表征配置。
    输出：{"records", "source_records", "source", "split"(pool/test/计数), "adapt"}。
    为什么：v6 的"适配档"就是选标协议产出的唯一一份池，与 fold/seed/tsplit 无关；
    池外文件一律不进训练、BN 更新或对齐，所以这里只把 pool 传给特征构建。
    """
    if target_source not in ds.TARGET_DATASETS:
        raise ValueError(f"目标来源必须为 {ds.TARGET_DATASETS} 之一")
    records = ds.enumerate_target_files(data_root)
    splits = ds.target_split_v6(data_root, selection_path)
    split = splits[target_source]
    source_records = [record for record in records if record["dataset"] == target_source]
    cfg = pp.config_from_saved(repr_config) if repr_config is not None else pp.build_repr_config()
    adapt_de = _channel_features(source_records, split["pool"], "DE", cfg)
    adapt_fe = _channel_features(source_records, split["pool"], "FE", cfg)
    return {
        "records": records,
        "source_records": source_records,
        "source": target_source,
        "selection_path": str(selection_path or ds.DEFAULT_SELECTION_PATH),
        "split": {"pool": split["pool"], "test": split["test"],
                  "pool_file_count": len(split["pool"]),
                  "test_file_count": len(split["test"]),
                  "actual_label_ratio": len(split["pool"]) / len(split["all"]),
                  "counts": split["pool_class_counts"],
                  "test_counts": split["test_class_counts"],
                  "test_group_counts": split["test_group_counts"],
                  "empty_test_groups": split["empty_groups"]},
        "adapt": pp.merge_feature_sets((adapt_de, adapt_fe)),
    }


def split_payload(data_root=None, fold=0, split_seed=0, selection_path=None):
    """生成 v6 split.json 内容：源四折 + 选中折 + 由冻结池决定的目标划分，不读窗口特征。"""
    source_records = ds.enumerate_source_files(data_root)
    target_records = ds.enumerate_target_files(data_root)
    folds = ds.split_source_files(source_records, split_seed)
    target = ds.target_split_v6(data_root, selection_path)
    return {
        "source": {"split_seed": split_seed, "folds": folds, "selected_fold": fold},
        "target": {"protocol": "v6", "selection": str(selection_path or ds.DEFAULT_SELECTION_PATH),
                   "splits": target},
    }


def write_split(path, payload):
    # split.json 直接覆盖写（train.py 已保证输出目录为空）。
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _self_check():
    """手动自检：只查划分与 S_ref，不构建特征（特征构建由冒烟流程覆盖）。"""
    root = ds.default_data_root()
    folds = ds.split_source_files(ds.enumerate_source_files(root), 0)
    assert len(folds) == 4
    for fold, selected in enumerate(folds):
        s_ref = reference_s_ref(ds.enumerate_source_files(root), selected["train"])
        assert s_ref > 0
    splits = ds.target_split_v6(root)
    for source, split in splits.items():
        assert set(split["pool"]).isdisjoint(split["test"])
        assert len(split["pool"]) + len(split["test"]) == len(split["all"])
    print("ALL EXPERIMENT CHECKS PASSED")


if __name__ == "__main__":
    _self_check()
