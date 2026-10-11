# -*- coding: utf-8 -*-
"""v6 训练入口：一次流程一个目录，四种臂（source_continue / adabn / ce_adapt / mmd_adapt）。

由 v5 `train.py` 收窄而来（见 `docs/v5复用与迁移清单.md` §2）：
删 `--protocol`、`--adapt-arm`、`--label-ratio`、`--target-split-seed`、`--representation`、
`--mmd-bandwidth` 与 `source`/`target_only`/`src_ce_target_bn` 三个 mode；
表征写死 logspec512，MMD 写死固定核，目标划分只读 `selection.py` 的冻结池。
v6 不重训源模型：所有流程都从 v5 的 `joint` `last.pt` 出发（`--source-checkpoint`）。

落盘：`split.json`、`best.pt`、`last.pt`、`history.csv`、`config.json`
（适配臂另写 `target_split.json`）。新目录必写 `checkpoint_kind`/`checkpoint_epoch`。
留出测试不在这里发生，由 `evaluate.py` 显式执行。
"""
import argparse
import math
from pathlib import Path

import dataset as ds
import experiment as ex
import preprocessing as pp
from model import MSRSABiLSTM
from trainer import (DEVICE, dump_json, load_checkpoint, save_checkpoint, seed_everything,
                     train_adabn, train_adapt, train_source_continue, write_history)

#: v6 冻结的训练参数，CLI 不再开放修改。
FIXED_LAMBDA_MMD = 0.3
FIXED_MMD_SIGMA = (1, 5, 10)
FIXED_EPOCHS = 30
FIXED_BATCH_SIZE = 64
FIXED_LR = 1e-4


def build_parser():
    parser = argparse.ArgumentParser(description="RBOP v6 训练（单表征 logspec512）")
    parser.add_argument("--data-root", default=str(ds.default_data_root()))
    parser.add_argument("--output", required=True)
    parser.add_argument("--mode", choices=["source_continue", "adabn", "ce_adapt", "mmd_adapt"],
                        required=True)
    # 四种臂：source_continue=仅源 CE 续训（不接触目标）；adabn=仅目标前向更新 BN（无梯度）；
    # ce_adapt=源重放+目标 CE；mmd_adapt=再加共有故障类 MMD。
    parser.add_argument("--target-source", choices=list(ds.TARGET_DATASETS), default=None)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--source-checkpoint", required=True, help="同 fold 同 seed 的 v5 joint last.pt")
    parser.add_argument("--selection", default=str(ds.DEFAULT_SELECTION_PATH),
                        help="冻结标注池 selection.json")
    parser.add_argument("--lambda-mmd", type=float, default=FIXED_LAMBDA_MMD)
    parser.add_argument("--num-threads", type=int, default=4)
    return parser


def _validate(args):
    """校验一次流程的参数组合，不合法直接退出。"""
    if args.fold not in range(4):
        raise SystemExit("--fold 必须为 0、1、2 或 3")
    if args.split_seed != 0:
        raise SystemExit("v6 源划分固定 split_seed=0")
    if args.mode == "source_continue":
        if args.target_source is not None:
            raise SystemExit("source_continue 不接触目标来源")
    elif args.target_source is None:
        raise SystemExit(f"{args.mode} 必须指定 --target-source")
    if Path(args.source_checkpoint).name != "last.pt":
        raise SystemExit("v6 起点必须是 joint 的 last.pt，不能用 best.pt 或其他文件冒充末轮起点")
    if args.mode == "mmd_adapt" and (not math.isfinite(args.lambda_mmd) or args.lambda_mmd <= 0):
        raise SystemExit("mmd_adapt 要求 --lambda-mmd 为有限正数")
    if args.mode == "ce_adapt":
        args.lambda_mmd = 0.0
    if args.mode != "mmd_adapt":
        args.lambda_mmd = 0.0
    if args.num_threads <= 0:
        raise SystemExit("--num-threads 必须为正数")


def _model():
    return MSRSABiLSTM().to(DEVICE)


def run(args):
    """执行一次训练流程并落盘全部产物，返回 config。

    顺序：校验组合 → 定种子 → 拒绝非空输出目录 → 写 split.json → 读起点 checkpoint
    并逐字段核对（同折/同种子/同预算/末轮/logspec512）→ 按起点配置准备源特征 →
    适配臂再准备池内目标特征 → 训练 → 落 best.pt/last.pt/history.csv/config.json。
    """
    import torch
    _validate(args)
    torch.set_num_threads(args.num_threads)
    seed_everything(args.seed)
    output = Path(args.output)
    if output.exists() and any(output.iterdir()):
        raise SystemExit(f"输出目录已有内容，不静默覆盖：{output}")
    output.mkdir(parents=True, exist_ok=True)
    ex.write_split(output / "split.json",
                   ex.split_payload(args.data_root, args.fold, args.split_seed, args.selection))

    model = _model()
    init_path = str(args.source_checkpoint)
    initial = load_checkpoint(init_path, model)
    # 跨版本只认这一条依赖：v5 joint 的 last.pt + selection_rule=last（v5 无 checkpoint_kind）。
    if initial.get("mode") != "source" or initial.get("channels") != "joint":
        raise ValueError("v6 起点必须是 joint 源模型")
    if initial.get("selection_rule") != "last":
        raise ValueError("v6 起点必须按 selection_rule=last 认定末轮权重")
    if initial.get("last_epoch") != FIXED_EPOCHS or initial.get("epochs") != FIXED_EPOCHS:
        raise ValueError("起点不是跑满 30 轮的末轮快照")
    if initial.get("updates") != FIXED_EPOCHS * initial.get("steps_per_epoch", -1):
        raise ValueError("起点的更新预算不符合冻结协议")
    init_repr = pp.config_from_saved(initial.get("repr_config"), require_complete=True)
    for field in ("representation", "input_repr"):
        if field in initial and initial[field] != init_repr["representation"]:
            raise ValueError("源 checkpoint 顶层表征声明与内部配置不一致")
    source = ex.prepare_source(args.data_root, args.fold, args.split_seed, init_repr,
                               build_train=args.mode != "adabn")
    expected = {"fold": args.fold, "seed": args.seed, "split_seed": args.split_seed,
                "batch_size": FIXED_BATCH_SIZE, "s_ref": source.s_ref}
    if any(initial.get(key) != value for key, value in expected.items()):
        raise ValueError("v6 后续训练必须使用同折、同种子、同预算的 joint 源 last.pt 作为起点")

    target = None
    if args.mode in ("adabn", "ce_adapt", "mmd_adapt"):
        target = ex.prepare_target_v6(args.data_root, args.target_source, args.selection, init_repr)

    if args.mode == "source_continue":
        result = train_source_continue(model, source.train, source.val, FIXED_BATCH_SIZE,
                                       FIXED_EPOCHS, FIXED_LR, args.seed, source.s_ref,
                                       progress_path=output / "progress.pt")
    elif args.mode == "adabn":
        result = train_adabn(model, target["adapt"], source.val, FIXED_BATCH_SIZE, FIXED_EPOCHS,
                             args.seed, source.s_ref, progress_path=output / "progress.pt")
    else:
        result = train_adapt(model, source.train, target["adapt"], source.val, FIXED_BATCH_SIZE,
                             FIXED_EPOCHS, FIXED_LR, args.lambda_mmd, args.seed, source.s_ref,
                             progress_path=output / "progress.pt")

    config = {
        "schema_version": 1,
        "task": "v6_round1",
        "protocol": "v6",
        "selection_rule": "last",
        "selected_epoch": FIXED_EPOCHS,
        "selected_checkpoint": "last.pt",
        # 正式权重的身份直接写进 config.json（v5 缺 101 个流程的教训：只能靠文件名推断末轮）。
        # best.pt/last.pt 的 payload 里 save_checkpoint 会按各自文件再写一次，不冲突。
        "checkpoint_kind": "last",
        "checkpoint_epoch": FIXED_EPOCHS,
        "data_root": str(Path(args.data_root).resolve()),
        "selection_path": str(Path(args.selection).resolve()),
        "num_threads": args.num_threads,
        "best_epoch": result["best"]["epoch"],
        "best_val_macro_f1_mean": result["best"]["score"],
        "best_val_accuracy_mean": result["best"]["accuracy"],
        "last_epoch": result["last"]["epoch"],
        "last_val_macro_f1_mean": result["last"]["score"],
        "last_val_accuracy_mean": result["last"]["accuracy"],
        "mode": args.mode,
        "arm": {"source_continue": "source_continue", "adabn": "adabn",
                "ce_adapt": "ce", "mmd_adapt": "ce_mmd"}[args.mode],
        "channels": "joint",
        "target_source": args.target_source,
        "pool_file_count": len(target["split"]["pool"]) if target else None,
        "actual_label_ratio": target["split"]["actual_label_ratio"] if target else None,
        "label_class_counts": target["split"]["counts"] if target else None,
        "test_file_count": target["split"]["test_file_count"] if target else None,
        "fold": args.fold,
        "seed": args.seed,
        "split_seed": args.split_seed,
        "source_checkpoint": init_path,
        "source_protocol": initial.get("protocol"),
        "learning_rate": (None if args.mode == "adabn" else FIXED_LR),
        "batch_size": FIXED_BATCH_SIZE,
        "epochs": FIXED_EPOCHS,
        "steps_per_epoch": result["steps_per_epoch"],
        "updates": result["updates"],
        "target_forwards": result.get("target_forwards"),
        "s_ref": source.s_ref,
        "lambda_mmd": args.lambda_mmd,
        "mmd_bandwidth": "fixed",
        "mmd_sigma": list(FIXED_MMD_SIGMA) if args.mode == "mmd_adapt" else None,
        "class_names": ds.LABEL_NAMES,
        "input_repr": init_repr["representation"],
        "representation": init_repr["representation"],
        "repr_config": source.repr_config,
        "model": {"n_channels": 1, "n_classes": 4, "frontend": "spectral_v1"},
    }
    # 正式权重恒为 last.pt；best.pt 保留"源验证最优"原义，不冒写。
    best_model = _model()
    best_model.load_state_dict(result["best"]["state_dict"])
    save_checkpoint(output / "best.pt", best_model, config)
    model.load_state_dict(result["last"]["state_dict"])
    save_checkpoint(output / "last.pt", model, config)
    if target is not None:
        dump_json(output / "target_split.json", target["split"])
    write_history(output / "history.csv", result["history"])
    dump_json(output / "config.json", config)
    print(f"训练已保存：{output / 'last.pt'}，设备：{DEVICE}，更新数：{result['updates']}；尚未执行目标测试", flush=True)
    return config


def main():
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
