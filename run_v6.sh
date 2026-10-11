#!/usr/bin/env bash
# v6 第一轮顺序执行脚本：96 个运行单元（60 完整训练 + 24 无梯度 + 12 基线评估）+ 汇总。
#
# 用法：
#   bash run_v6.sh all         # 按 baseline → continue → adabn → adapt → summary 全跑
#   bash run_v6.sh baseline    # 12 个 joint 基线评估（零训练，用 weights/v5_joint 的末轮权重）
#   bash run_v6.sh continue    # 12 个 source_continue 训练+评估（两来源各评）
#   bash run_v6.sh adabn       # 24 个 adabn 无梯度运行+评估
#   bash run_v6.sh adapt       # 48 个 ce/ce_mmd 训练+评估（2 来源 × 2 臂 × 4 折 × 3 seed）
#   bash run_v6.sh summary     # 六对配对链 + ≥95% 判据核对
#   bash run_v6.sh status      # 只读：完成多少、下一个是什么，看训练跑到哪一步
#   bash run_v6.sh one <type> <source|-> <fold> <seed> [后缀]   # 单独重跑一个单元
#     后缀 retry1 → 产物进正式目录（train/<名>_retry1），作为该单元的重跑；
#     后缀 smoke  → 产物进 outputs/v6_round1/_smoke/，只是管道检查，不进汇总。
#
# 进度：每个单元打印 [i/N]、本单元耗时、累计耗时与按均值推算的剩余分钟数。
# 想留全程日志就自己 tee（脚本不建状态机）：
#   nohup bash run_v6.sh all 2>&1 | tee outputs/v6_round1/logs/run_v6.log &
#   tail -f outputs/v6_round1/logs/run_v6.log
# 失败处理沿用 v5：默认在第一个失败处停并打印重跑命令；已存在的产物自动跳过，
# 所以中断后直接重跑本脚本会继续没跑完的单元。KEEP_GOING=1 可让整批不因单个失败中断。
# 环境变量：PY=解释器路径  DATA_ROOT / V5_JOINT / OUT=覆盖默认位置  SUFFIX=单元目录后缀。
set -euo pipefail

cd "$(dirname "$0")"
ROOT=$PWD
PY=${PY:-python}
DATA_ROOT=${DATA_ROOT:-$ROOT/Data}
V5_JOINT=${V5_JOINT:-$ROOT/weights/v5_joint}
OUT=${OUT:-outputs/v6_round1}
SELECTION=$OUT/selection/selection.json
SOURCES=(12k_FE 48k_DE)
FOLDS=(0 1 2 3)
SEEDS=(0 1 2)

# ---------- 前置检查：缺了立刻停，不要跑到一半才发现 ----------
[ -f "$SELECTION" ] || { echo "缺少冻结标注池：$SELECTION（先跑 $PY selection.py）" >&2; exit 1; }
for d in Normal 12k_DE 12k_FE 48k_DE; do
  [ -d "$DATA_ROOT/$d" ] || { echo "缺少数据目录：$DATA_ROOT/$d" >&2; exit 1; }
done
for f in "${FOLDS[@]}"; do for s in "${SEEDS[@]}"; do
  [ -f "$V5_JOINT/f${f}_s${s}_joint/last.pt" ] || { echo "缺少起点权重：$V5_JOINT/f${f}_s${s}_joint/last.pt" >&2; exit 1; }
done; done
echo "前置检查通过｜数据 $DATA_ROOT｜起点权重 12/12 $V5_JOINT｜标注池 $SELECTION"
echo "设备：$("$PY" -c 'import torch;print(torch.cuda.is_available() and torch.cuda.get_device_name(0) or "cpu", torch.__version__)')"

unit_name () {  # $1=type $2=source $3=fold $4=seed
  if [ "$1" = "joint" ]; then echo "joint_f${3}_s${4}"
  elif [ "$1" = "source_continue" ]; then echo "source_continue_f${3}_s${4}"
  elif [ "$2" = "-" ]; then echo "${1}_f${3}_s${4}"
  else echo "${2}_${1}_f${3}_s${4}"; fi
}

train_done () { local d=$1; [ -f "$d/last.pt" ] && [ -f "$d/config.json" ] && [ -f "$d/history.csv" ]; }
eval_done () { local d=$1; [ -f "$d/metrics.json" ] && [ -f "$d/predictions.csv" ] && [ -f "$d/eval_info.json" ]; }
unit_done () { local type=$1 source=$2 fold=$3 seed=$4 name td ed; name=$(unit_name "$type" "$source" "$fold" "$seed")
  td="$OUT/train/$name"; ed="$OUT/eval/$name"
  if [ "$type" = "joint" ]; then eval_done "$ed"; else train_done "$td" && eval_done "$ed"; fi; }

run_unit () {  # $1=type $2=source $3=fold $4=seed（SUFFIX 非空时产物写到 _smoke 根，不污染正式目录与汇总）
  local type=$1 source=$2 fold=$3 seed=$4 name base td ed args=()
  name=$(unit_name "$type" "$source" "$fold" "$seed")${SUFFIX:+_$SUFFIX}
  base="$OUT"
  if [ "${SUFFIX:-}" = "smoke" ]; then base="$OUT/_smoke"; mkdir -p "$base/train" "$base/eval"; fi
  td="$base/train/$name"; ed="$base/eval/$name"
  if [ "$type" = "joint" ]; then
    if eval_done "$ed"; then echo "  跳过（已评估）$name"; return 0; fi
    "$PY" evaluate.py --joint-dir "$V5_JOINT/f${fold}_s${seed}_joint" --output "$ed" --data-root "$DATA_ROOT"
    return 0
  fi
  if train_done "$td"; then
    echo "  训练产物齐备，跳过训练：$name"
  else
    args=(--mode "$type" --fold "$fold" --seed "$seed" --data-root "$DATA_ROOT"
          --selection "$SELECTION" --output "$td"
          --source-checkpoint "$V5_JOINT/f${fold}_s${seed}_joint/last.pt")
    [ "$source" != "-" ] && args+=(--target-source "$source")
    "$PY" train.py "${args[@]}"
  fi
  if eval_done "$ed"; then
    echo "  跳过（已评估）$name"
  else
    "$PY" evaluate.py --run-dir "$td" --output "$ed" --data-root "$DATA_ROOT"
  fi
}

build_units () {
  UNITS=()
  local source fold seed mode action
  for action in $1; do
    case $action in
      baseline) for fold in "${FOLDS[@]}"; do for seed in "${SEEDS[@]}"; do UNITS+=("joint|-|$fold|$seed"); done; done ;;
      continue) for fold in "${FOLDS[@]}"; do for seed in "${SEEDS[@]}"; do UNITS+=("source_continue|-|$fold|$seed"); done; done ;;
      adabn) for source in "${SOURCES[@]}"; do for fold in "${FOLDS[@]}"; do for seed in "${SEEDS[@]}"; do UNITS+=("adabn|$source|$fold|$seed"); done; done; done ;;
      adapt) for source in "${SOURCES[@]}"; do for mode in ce_adapt mmd_adapt; do
               for fold in "${FOLDS[@]}"; do for seed in "${SEEDS[@]}"; do UNITS+=("$mode|$source|$fold|$seed"); done; done
             done; done ;;
    esac
  done
}

run_all_units () {
  local total=${#UNITS[@]} i=0 unit type source fold seed name t0 el tot avg
  local start; start=$(date +%s)
  echo "== 本次共 $total 个单元"
  for unit in "${UNITS[@]}"; do
    i=$((i + 1)); IFS='|' read -r type source fold seed <<< "$unit"
    t0=$(date +%s)
    echo "------------------------------------------------------------------"
    echo "[$i/$total] $(unit_name "$type" "$source" "$fold" "$seed") 开始 $(date '+%H:%M:%S')"
    if ! run_unit "$type" "$source" "$fold" "$seed"; then
      echo "[$i/$total] $(unit_name "$type" "$source" "$fold" "$seed") 失败 → 重跑该单元：bash run_v6.sh one $type $source $fold $seed retry1"
      if [ "${KEEP_GOING:-0}" = "1" ]; then continue; fi
      exit 1
    fi
    el=$(( $(date +%s) - t0 )); tot=$(( $(date +%s) - start )); avg=$(( tot / i ))
    echo "[$i/$total] 完成，本单元 ${el}s，累计 ${tot}s，剩余约 $(( avg * (total - i) / 60 )) 分钟（按均值推算）"
  done
  echo "== 全部 $total 个单元结束，总耗时 $(( ($(date +%s) - start) / 60 )) 分钟"
}

show_status () {
  build_units "${1:-}"
  local total=${#UNITS[@]} done=0 next="" unit type source fold seed
  for unit in "${UNITS[@]}"; do
    IFS='|' read -r type source fold seed <<< "$unit"
    if unit_done "$type" "$source" "$fold" "$seed"; then
      done=$((done + 1))
    elif [ -z "$next" ]; then
      next="$(unit_name "$type" "$source" "$fold" "$seed")（$type $source f$fold s$seed）"
    fi
  done
  echo "完成 $done/$total 个单元｜下一个：${next:-无}"
  echo "训练目录：$([ -d "$OUT/train" ] && ls "$OUT/train" | wc -l || echo 0) 个；评估目录：$([ -d "$OUT/eval" ] && ls "$OUT/eval" | wc -l || echo 0) 个"
}

do_summary () {
  local dirs=() dir
  for dir in "$OUT"/eval/*; do
    [ -f "$dir/metrics.json" ] && [ -f "$dir/eval_info.json" ] && dirs+=("$dir")
  done
  echo "== 汇总 ${#dirs[@]} 个评估目录"
  "$PY" evaluate.py --summary-runs "${dirs[@]}" --output "$OUT/summary"
}

ACTION=${1:-all}
case $ACTION in
  all) build_units "baseline continue adabn adapt"; run_all_units; do_summary ;;
  baseline) build_units "baseline"; run_all_units ;;
  continue) build_units "continue"; run_all_units ;;
  adabn) build_units "adabn"; run_all_units ;;
  adapt) build_units "adapt"; run_all_units ;;
  summary) do_summary ;;
  status) show_status "${2:-baseline continue adabn adapt}" ;;
  one) if [ $# -lt 5 ]; then echo "用法：bash run_v6.sh one <type> <source|-> <fold> <seed> [后缀]" >&2; exit 2; fi
    SUFFIX=${6:-}; run_unit "$2" "$3" "$4" "$5" ;;
  *) echo "未知动作：$ACTION（all/baseline/continue/adabn/adapt/summary/status/one）" >&2; exit 2 ;;
esac
