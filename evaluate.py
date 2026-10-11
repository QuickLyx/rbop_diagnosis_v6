# -*- coding: utf-8 -*-
"""v6 独立留出评估与配对汇总。

由 v5 `evaluate.py` 迁移（见 `docs/v5复用与迁移清单.md` §2）：删 v1/v2 双协议分支、
`_summarize_runs_v1`、跨表征配对与四档臂映射；测试列表一律来自 `selection.json`
（`dataset.target_split_v6`），与 fold/seed 无关，所以评估命令不再接受目标划分种子。
配对链固定六对，训练前写死、中途不增删。

评估纪律：测试前向无梯度且不更新 BN（结束逐键校验权重与 BN 未变）；
记录级指标先按"同文件同通道"做窗口概率平均再单次判类；
`Macro-F1` 按有真值的类平均，无正常测试则正常召回留空；标准差一律 `ddof=1`。
"""
import argparse
import csv
import json
from pathlib import Path

import torch

import dataset as ds
import preprocessing as pp
from model import MSRSABiLSTM
from trainer import DEVICE, dump_json, evaluate_feature, load_checkpoint

#: v6 六对配对链：(候选臂, 基线臂, 名称)，训练前写死（协议 §7）。
PAIRS = (
    ("source_continue", "joint", "源续训 vs joint（额外更新步数本身）"),
    ("adabn", "joint", "adabn vs joint（无标注 BN 重估）"),
    ("ce", "joint", "CE vs joint（有监督适配总量）"),
    ("ce", "source_continue", "CE vs 源续训（标签本身值多少）"),
    ("ce", "adabn", "CE vs adabn（标签 vs 无监督 BN）"),
    ("ce_mmd", "ce", "CE+MMD vs CE（对齐项值多少）"),
)
#: 达标判据（训练前写死，事后不得改）。
CRITERION_ACCURACY = 0.95
ARM_BY_MODE = {"source": "joint", "source_continue": "source_continue", "adabn": "adabn",
               "ce_adapt": "ce", "mmd_adapt": "ce_mmd", "supcon_adapt": "supcon"}


def _model_from_checkpoint(checkpoint_path, require_v6=True):
    """装载权重并返回 (model, checkpoint 配置)。"""
    model = MSRSABiLSTM().to(DEVICE)
    config = load_checkpoint(checkpoint_path, model)
    repr_config = pp.config_from_saved(config.get("repr_config"), require_complete=True)
    if repr_config["representation"] != "logspec512":
        raise ValueError("v6 只评估 logspec512 权重")
    model.repr_config = repr_config
    if require_v6 and config.get("checkpoint_kind") not in ("best", "last"):
        raise ValueError("v6 自产 checkpoint 必须写明 checkpoint_kind")
    return model, config


def evaluate_groups(model, repr_config, data_root, sources):
    """在 v6 冻结测试列表上做记录级评估，返回 (metrics, predictions)。

    只读 test 文件信号，不读池内文件的测试视图；每个文件×通道是一条记录。
    """
    splits = ds.target_split_v6(data_root)
    records = ds.enumerate_target_files(data_root)
    result, predictions = {}, []
    for source in sources:
        if source not in ds.TARGET_DATASETS:
            raise ValueError(f"未知目标来源：{source}")
        test_keys = list(splits[source]["test"])
        if not test_keys:
            raise ValueError(f"{source} 测试列表为空")
        source_records = [record for record in records if record["dataset"] == source]
        if set(test_keys) & set(splits[source]["pool"]):
            raise ValueError(f"{source} 标注池文件不得进入测试")
        for channel in ("DE", "FE"):
            feature = pp.build_record_features(source_records, test_keys, channel, repr_config)
            metrics = evaluate_feature(model, feature)
            if len(metrics["record_keys"]) != len(test_keys):
                raise ValueError("记录数应等于测试文件数")
            result[f"{source}/{channel}"] = {
                key: value for key, value in metrics.items()
                if key not in ("record_keys", "predictions", "probabilities", "true")
            }
            for key, pred, probs, true, count in zip(
                    metrics["record_keys"], metrics["predictions"], metrics["probabilities"],
                    metrics["true"], metrics["window_counts"]):
                predictions.append({"target_source": source, "test_channel": channel,
                                    "file": key.rsplit("::", 1)[0], "true": int(true),
                                    "pred": int(pred), "window_count": int(count),
                                    **{f"prob_{label}": float(probs[index])
                                       for index, label in enumerate(ds.LABEL_NAMES)}})
    return result, predictions


def _write_predictions(path, predictions):
    fields = ["target_source", "test_channel", "file", "true", "pred", "window_count",
              *[f"prob_{label}" for label in ds.LABEL_NAMES]]
    with Path(path).open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(predictions)


def write_eval_output(output_dir, model, metrics, predictions, info, before):
    """落 metrics.json / predictions.csv / eval_info.json，并校验前向未改动权重。"""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    for name in ("metrics.json", "predictions.csv", "eval_info.json"):
        if (output / name).exists():
            raise FileExistsError(f"不覆盖已有评估结果：{output / name}")
    assert all(torch.equal(value.detach().cpu(), before[key]) for key, value in model.state_dict().items())
    dump_json(output / "metrics.json", metrics)
    _write_predictions(output / "predictions.csv", predictions)
    dump_json(output / "eval_info.json", info)


def evaluate_run(run_dir, output_dir=None, data_root=None):
    """评估一个 v6 训练目录：正式权重恒为本目录 last.pt，臂为 ce/ce_mmd/adabn/source_continue。"""
    run_dir = Path(run_dir)
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    if config.get("protocol") != "v6":
        raise ValueError("v6 评估只接受 v6 训练目录，禁止跨协议相减")
    if (config.get("selection_rule") != "last" or config.get("selected_checkpoint") != "last.pt"
            or config.get("selected_epoch") != config.get("epochs")):
        raise ValueError("正式评估要求训练已跑满并选取末轮")
    checkpoint_path = run_dir / "last.pt"
    model, stored = _model_from_checkpoint(checkpoint_path)
    if stored.get("checkpoint_kind") != "last" or stored.get("checkpoint_epoch") != config["epochs"]:
        raise ValueError("last.pt 实际保存的不是本次运行末轮权重")
    if json.loads(json.dumps({key: stored.get(key) for key in config})) != config:
        raise ValueError("checkpoint 内配置与训练配置不一致，禁止混用模型")
    before = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    sources = [config["target_source"]] if config.get("target_source") else list(ds.TARGET_DATASETS)
    metrics, predictions = evaluate_groups(model, stored["repr_config"],
                                          data_root or config.get("data_root"), sources)
    output = Path(output_dir) if output_dir is not None else run_dir
    if output.resolve() == run_dir.resolve():
        raise ValueError("v6 评估必须写入独立输出目录，不覆盖训练产物")
    info = {
        "run_dir": str(run_dir.resolve()),
        "checkpoint": "last.pt",
        "protocol": "v6",
        "arm": config["arm"],
        "mode": config["mode"],
        "fold": config["fold"],
        "seed": config["seed"],
        "split_seed": config["split_seed"],
        "trained_target_source": config.get("target_source"),
        "evaluated_sources": sources,
        "test_files": {group: sorted(row["file"] for row in predictions
                                     if f"{row['target_source']}/{row['test_channel']}" == group)
                       for group in metrics},
        "config": config,
    }
    write_eval_output(output, model, metrics, predictions, info, before)
    _print_groups(run_dir.name, metrics)
    return metrics


def evaluate_joint(checkpoint_dir, output_dir, data_root=None):
    """基线：直接复用 v5 的 joint 源 `last.pt`，在 v6 列表上做一次无梯度前向。

    v5 目录只读；eval_info 里的 config 由 v5 config 补齐 v6 字段（arm=joint、协议 v6），
    并显式记下起点文件名与 selection_rule=last 这条跨版本依赖。
    """
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_path = checkpoint_dir / "last.pt"
    v5_config = json.loads((checkpoint_dir / "config.json").read_text(encoding="utf-8"))
    if (v5_config.get("mode") != "source" or v5_config.get("channels") != "joint"
            or v5_config.get("selection_rule") != "last"):
        raise ValueError("基线起点必须是 joint 源且 selection_rule=last")
    if v5_config.get("representation") != "logspec512":
        raise ValueError("v6 基线只用 logspec512 源权重")
    model, stored = _model_from_checkpoint(checkpoint_path, require_v6=False)
    before = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    sources = list(ds.TARGET_DATASETS)
    metrics, predictions = evaluate_groups(model, stored["repr_config"], data_root, sources)
    config = dict(v5_config)
    config.update({
        "task": "v6_round1", "protocol": "v6", "arm": "joint", "mode": "source",
        "channels": "joint", "selected_checkpoint": "last.pt",
        "selection_rule": "last", "selected_epoch": v5_config["last_epoch"],
        "target_source": None, "source_checkpoint": str(checkpoint_path.resolve()),
        "v5_checkpoint_kind_absent": True,
    })
    info = {
        "run_dir": str(checkpoint_dir.resolve()),
        "checkpoint": "last.pt",
        "protocol": "v6",
        "arm": "joint",
        "mode": "source",
        "fold": config["fold"],
        "seed": config["seed"],
        "split_seed": config["split_seed"],
        "trained_target_source": None,
        "evaluated_sources": sources,
        "test_files": {group: sorted(row["file"] for row in predictions
                                     if f"{row['target_source']}/{row['test_channel']}" == group)
                       for group in metrics},
        "config": config,
    }
    output = Path(output_dir)
    write_eval_output(output, model, metrics, predictions, info, before)
    _print_groups(output.name, metrics)
    return metrics


def _print_groups(label, metrics):
    for group, values in metrics.items():
        print(f"{label}，{group}：记录数={sum(values['support'])}，"
              f"Accuracy={values['accuracy']:.4f}，Macro-F1={values['macro_f1']:.4f}", flush=True)


def _mean_std(values):
    """均值与样本标准差（ddof=1）；单样本记 0.0。"""
    n = len(values)
    if n == 0:
        raise ValueError("空序列不能计算均值")
    mean = float(sum(values) / n)
    if n == 1:
        return mean, 0.0
    variance = sum((float(value) - mean) ** 2 for value in values) / (n - 1)
    return mean, float(variance ** 0.5)


def _load_entry(directory):
    directory = Path(directory)
    metrics = json.loads((directory / "metrics.json").read_text(encoding="utf-8"))
    info = json.loads((directory / "eval_info.json").read_text(encoding="utf-8"))
    config = info.get("config", {})
    if config.get("protocol") != "v6":
        raise ValueError(f"{directory} 不是 v6 评估目录，禁止跨协议相减")
    saved = info.get("test_files")
    if saved is None or set(saved) != set(metrics):
        raise ValueError("eval_info 的测试文件列表与指标组不一致")
    for group, files in saved.items():
        if len(files) != len(set(files)) or len(files) != sum(metrics[group]["support"]):
            raise ValueError("测试文件数与支持记录数不一致，或存在重复记录")
    files = {group: tuple(sorted(items)) for group, items in saved.items()}
    return directory, config, metrics, files


def summarize_runs(run_dirs, output):
    """v6 汇总：逐格明细 + 按 (臂×来源×通道) 的均值±样本标准差 + 六对配对差值 + 判据核对。

    cell 为 (split_seed, fold, seed)；配对只在同一 cell、同一 `来源/通道` 分组且
    双方测试文件列表逐项一致时计算，负增量完整保留；两个适配臂还要求训练来源相同。
    """
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"不覆盖已有汇总目录：{output}")
    entries = [_load_entry(directory) for directory in run_dirs]
    if not entries:
        raise ValueError("汇总需要至少一个已评估目录")
    output.mkdir(parents=True, exist_ok=True)
    detail, seen, cells = [], {}, {}
    for directory, config, metrics, files in entries:
        identity = (config.get("arm"), config.get("target_source"),
                    config.get("split_seed"), config.get("fold"), config.get("seed"))
        if identity in seen:
            raise ValueError(f"同一 cell 出现重复臂 {identity}，不能择优汇总：{directory} 与 {seen[identity]}")
        seen[identity] = directory
        cell = (config.get("split_seed"), config.get("fold"), config.get("seed"))
        cells.setdefault(cell, {})[identity[:2]] = (metrics, files)
        for group, values in metrics.items():
            source, channel = group.split("/")
            detail.append({
                "eval_dir": directory.name, "arm": config.get("arm"), "mode": config.get("mode"),
                "trained_target_source": config.get("target_source"),
                "split_seed": config.get("split_seed"), "fold": config.get("fold"),
                "seed": config.get("seed"), "target_source": source, "test_channel": channel,
                "pool_file_count": config.get("pool_file_count"),
                "actual_label_ratio": config.get("actual_label_ratio"),
                "s_ref": config.get("s_ref"), "steps_per_epoch": config.get("steps_per_epoch"),
                "updates": config.get("updates"), "lambda_mmd": config.get("lambda_mmd"),
                "accuracy": values["accuracy"], "macro_f1": values["macro_f1"],
                **{f"recall_{name}": values["recall"][index]
                   for index, name in enumerate(ds.LABEL_NAMES)},
                "records": sum(values["support"]),
                "metrics_path": (directory / "metrics.json").as_posix(),
            })
    grouped = []
    buckets = {}
    for row in detail:
        key = (row["arm"], row["trained_target_source"], row["split_seed"],
               row["target_source"], row["test_channel"])
        buckets.setdefault(key, []).append(row)
    for key, members in sorted(buckets.items(), key=lambda item: str(item[0])):
        arm, trained_source, split_seed, source, channel = key
        acc_mean, acc_std = _mean_std([row["accuracy"] for row in members])
        f1_mean, f1_std = _mean_std([row["macro_f1"] for row in members])
        grouped.append({"arm": arm, "trained_target_source": trained_source,
                        "split_seed": split_seed, "target_source": source, "test_channel": channel,
                        "n_runs": len(members),
                        "folds": ";".join(str(value) for value in sorted({row["fold"] for row in members})),
                        "seeds": ";".join(str(value) for value in sorted({row["seed"] for row in members})),
                        "accuracy_mean": acc_mean, "accuracy_std": acc_std,
                        "macro_f1_mean": f1_mean, "macro_f1_std": f1_std,
                        # v5 缺陷修正：按实际成员写记录数，不再只取 members[0] 的单值。
                        "records_per_run": ";".join(str(value) for value in
                                                    sorted({row["records"] for row in members}))})
    comparisons = []
    for cell in sorted(cells):
        split_seed, fold, seed = cell
        indexed = cells[cell]
        for candidate_arm, baseline_arm, name in PAIRS:
            for adaptation in (None, *ds.TARGET_DATASETS):
                candidate = (candidate_arm, adaptation if _arm_needs_source(candidate_arm) else None)
                baseline = (baseline_arm, adaptation if _arm_needs_source(baseline_arm) else None)
                if candidate not in indexed or baseline not in indexed:
                    continue
                candidate_metrics, candidate_files = indexed[candidate]
                baseline_metrics, baseline_files = indexed[baseline]
                for group in candidate_metrics.keys() & baseline_metrics.keys():
                    if candidate_files[group] != baseline_files[group]:
                        raise ValueError("配对比较的实际测试文件不同，不能仅凭同折同种子相减")
                    current, previous = candidate_metrics[group], baseline_metrics[group]
                    comparisons.append({"split_seed": split_seed, "fold": fold, "seed": seed,
                                        "comparison": name, "candidate_arm": candidate_arm,
                                        "baseline_arm": baseline_arm, "adaptation_source": adaptation,
                                        "group": group,
                                        "delta_accuracy_pp": 100 * (current["accuracy"] - previous["accuracy"]),
                                        "delta_macro_f1_pp": 100 * (current["macro_f1"] - previous["macro_f1"])})
    comparisons.sort(key=lambda row: (row["group"], row["comparison"], row["fold"], row["seed"]))
    criteria = _criterion_rows(detail)
    _write_csv(output / "detail.csv", detail)
    _write_csv(output / "grouped.csv", grouped)
    _write_csv(output / "paired_comparisons.csv", comparisons)
    _write_csv(output / "criterion.csv", criteria)
    _write_report(output / "report.md", detail, grouped, comparisons, criteria, entries)
    print(f"汇总完成：{output / 'report.md'}，{len(detail)} 行明细、{len(grouped)} 行分组、"
          f"{len(comparisons)} 行配对差值", flush=True)
    return detail, grouped, comparisons, criteria


def _arm_needs_source(arm):
    return arm in ("adabn", "ce", "ce_mmd")


def _criterion_rows(detail):
    """按来源给记录级准确率的均值、最差格与过线格比例，对照 ≥95% 的达标判据。"""
    rows = []
    for arm in sorted({row["arm"] for row in detail}):
        per_source = {}
        for source in ds.TARGET_DATASETS:
            values = [row["accuracy"] for row in detail
                      if row["arm"] == arm and row["target_source"] == source]
            if not values:
                continue
            mean, std = _mean_std(values)
            per_source[source] = (mean, std, values)
            rows.append({"arm": arm, "target_source": source, "n_cells": len(values),
                         "accuracy_mean": mean, "accuracy_std": std,
                         "worst_cell": min(values), "cells_over_95": sum(v >= CRITERION_ACCURACY for v in values),
                         "pass_95": mean >= CRITERION_ACCURACY})
        if len(per_source) == len(ds.TARGET_DATASETS):
            macro = float(sum(value[0] for value in per_source.values()) / len(per_source))
            all_values = [value for item in per_source.values() for value in item[2]]
            rows.append({"arm": arm, "target_source": "宏平均", "n_cells": len(all_values),
                         "accuracy_mean": macro, "accuracy_std": _mean_std(all_values)[1] if len(all_values) > 1 else 0.0,
                         "worst_cell": min(all_values),
                         "cells_over_95": sum(v >= CRITERION_ACCURACY for v in all_values),
                         "pass_95": macro >= CRITERION_ACCURACY})
    return rows


def _write_csv(path, rows):
    if not rows:
        return
    with Path(path).open("x", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_report(path, detail, grouped, comparisons, criteria, entries):
    text = ["# v6 第一轮结果汇总", "",
            f"共 {len(entries)} 个已评估目录，{len(detail)} 行逐格明细，{len(grouped)} 行分组，"
            f"{len(comparisons)} 行配对差值。", "",
            "达标判据：记录级准确率按来源的均值 ≥95%（训练前写死）。逐格值、最差格与过线格比例全部保留；"
            "标准差为样本标准差（ddof=1），描述运行间波动而不是独立测试样本的置信区间。", "",
            "## 判据核对", "",
            "| 臂 | 来源 | 格数 | 准确率均值 (%) | 样本标准差 (%) | 最差格 (%) | 过 95% 格数 | 达标 |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |"]
    for row in criteria:
        text.append(f"| {row['arm']} | {row['target_source']} | {row['n_cells']} | "
                    f"{100 * row['accuracy_mean']:.2f} | {100 * row['accuracy_std']:.2f} | "
                    f"{100 * row['worst_cell']:.2f} | {row['cells_over_95']} | "
                    f"{'是' if row['pass_95'] else '否'} |")
    text += ["", "## 分组均值（臂 × 训练来源 × 测试来源 × 通道）", "",
             "| 臂 | 训练来源 | 测试来源 | 通道 | 格数 | 准确率均值 (%) | 标准差 (%) | Macro-F1 均值 (%) | 记录数 |",
             "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- |"]
    for row in grouped:
        text.append(f"| {row['arm']} | {row['trained_target_source'] or '不限'} | {row['target_source']} | "
                    f"{row['test_channel']} | {row['n_runs']} | {100 * row['accuracy_mean']:.2f} | "
                    f"{100 * row['accuracy_std']:.2f} | {100 * row['macro_f1_mean']:.2f} | "
                    f"{row['records_per_run']} |")
    text += ["", "## 六对配对链（逐格差值）", "",
             "| 折 | seed | 比较 | 组 | ΔAccuracy (pp) | ΔMacro-F1 (pp) |",
             "| --- | --- | --- | --- | ---: | ---: |"]
    for row in comparisons:
        text.append(f"| {row['fold']} | {row['seed']} | {row['comparison']} | {row['group']} | "
                    f"{row['delta_accuracy_pp']:+.2f} | {row['delta_macro_f1_pp']:+.2f} |")
    negatives = [row for row in comparisons
                 if row["delta_accuracy_pp"] < 0 or row["delta_macro_f1_pp"] < 0]
    text += ["", "## 负增量", "", "以下负差值全部保留，不择优选报告。"]
    text += [f"- fold={row['fold']} seed={row['seed']} {row['group']}，{row['comparison']}："
             f"Accuracy {row['delta_accuracy_pp']:+.2f}、Macro-F1 {row['delta_macro_f1_pp']:+.2f} 个百分点。"
             for row in negatives] or ["- 本轮未观察到负增量。"]
    text += ["", "## 结论边界", "",
             "- 标注池与 fold/seed 无关（选标只用无标签信号），所以本轮没有划分轴可分层报告；"
             "看不到 tsplit 维度的方向反转，是协议 §4 塌缩的既定代价。",
             "- 目标域无 Normal 类，正常召回为 null，Macro-F1 只平均有真值的三类，不能验证正常误报率。",
             "- 同文件的 DE/FE 是两条记录但不是独立采集，逐格值不是独立样本。",
             "- 本轮 `source_continue` 只对齐额外更新步数，不声称匹配全部 FLOPs。",
             "- 跨版本纪律：v6 测试列表与 v5 不同，禁止与 v5 任何数字相减。", ""]
    path.write_text("\n".join(text), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="v6 独立留出评估与六对配对汇总")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--run-dir", help="v6 训练目录")
    mode.add_argument("--joint-dir", help="v5 joint 源目录（基线，只读）")
    mode.add_argument("--summary-runs", nargs="+", help="已评估目录列表")
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--output", default=None, help="评估输出目录或汇总目录")
    args = parser.parse_args()
    if args.run_dir:
        if args.output is None:
            parser.error("--run-dir 评估必须指定 --output 独立目录")
        evaluate_run(args.run_dir, args.output, args.data_root)
    elif args.joint_dir:
        if args.output is None:
            parser.error("--joint-dir 基线评估必须指定 --output 目录")
        evaluate_joint(args.joint_dir, args.output, args.data_root)
    else:
        if not args.output:
            parser.error("汇总必须指定 --output")
        summarize_runs(args.summary_runs, args.output)


if __name__ == "__main__":
    main()
