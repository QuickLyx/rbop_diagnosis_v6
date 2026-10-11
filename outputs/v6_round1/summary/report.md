# v6 第一轮结果汇总

共 96 个已评估目录，240 行逐格明细，20 行分组，384 行配对差值。

达标判据：记录级准确率按来源的均值 ≥95%（训练前写死）。逐格值、最差格与过线格比例全部保留；标准差为样本标准差（ddof=1），描述运行间波动而不是独立测试样本的置信区间。

## 判据核对

| 臂 | 来源 | 格数 | 准确率均值 (%) | 样本标准差 (%) | 最差格 (%) | 过 95% 格数 | 达标 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| adabn | 12k_FE | 24 | 25.36 | 9.39 | 11.43 | 0 | 否 |
| adabn | 48k_DE | 24 | 76.35 | 8.69 | 60.00 | 0 | 否 |
| adabn | 宏平均 | 48 | 50.86 | 27.28 | 11.43 | 0 | 否 |
| ce | 12k_FE | 24 | 95.00 | 2.56 | 88.57 | 10 | 是 |
| ce | 48k_DE | 24 | 94.79 | 4.54 | 87.50 | 12 | 否 |
| ce | 宏平均 | 48 | 94.90 | 3.65 | 87.50 | 22 | 否 |
| ce_mmd | 12k_FE | 24 | 95.12 | 1.78 | 91.43 | 7 | 是 |
| ce_mmd | 48k_DE | 24 | 94.69 | 4.62 | 87.50 | 13 | 否 |
| ce_mmd | 宏平均 | 48 | 94.90 | 3.47 | 87.50 | 20 | 否 |
| joint | 12k_FE | 24 | 10.24 | 12.03 | 0.00 | 0 | 否 |
| joint | 48k_DE | 24 | 67.81 | 19.06 | 5.00 | 2 | 否 |
| joint | 宏平均 | 48 | 39.03 | 33.09 | 0.00 | 2 | 否 |
| source_continue | 12k_FE | 24 | 6.67 | 8.41 | 0.00 | 0 | 否 |
| source_continue | 48k_DE | 24 | 74.06 | 10.60 | 57.50 | 1 | 否 |
| source_continue | 宏平均 | 48 | 40.36 | 35.35 | 0.00 | 1 | 否 |

## 分组均值（臂 × 训练来源 × 测试来源 × 通道）

| 臂 | 训练来源 | 测试来源 | 通道 | 格数 | 准确率均值 (%) | 标准差 (%) | Macro-F1 均值 (%) | 记录数 |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| adabn | 12k_FE | 12k_FE | DE | 12 | 19.52 | 5.56 | 16.20 | 35 |
| adabn | 12k_FE | 12k_FE | FE | 12 | 31.19 | 8.91 | 29.88 | 35 |
| adabn | 48k_DE | 48k_DE | DE | 12 | 74.79 | 6.78 | 73.89 | 40 |
| adabn | 48k_DE | 48k_DE | FE | 12 | 77.92 | 10.33 | 79.17 | 40 |
| ce | 12k_FE | 12k_FE | DE | 12 | 95.48 | 2.85 | 95.19 | 35 |
| ce | 12k_FE | 12k_FE | FE | 12 | 94.52 | 2.27 | 94.87 | 35 |
| ce | 48k_DE | 48k_DE | DE | 12 | 90.63 | 1.88 | 90.81 | 40 |
| ce | 48k_DE | 48k_DE | FE | 12 | 98.96 | 1.29 | 98.87 | 40 |
| ce_mmd | 12k_FE | 12k_FE | DE | 12 | 95.71 | 1.93 | 95.29 | 35 |
| ce_mmd | 12k_FE | 12k_FE | FE | 12 | 94.52 | 1.47 | 94.85 | 35 |
| ce_mmd | 48k_DE | 48k_DE | DE | 12 | 90.62 | 2.41 | 90.76 | 40 |
| ce_mmd | 48k_DE | 48k_DE | FE | 12 | 98.75 | 1.69 | 98.63 | 40 |
| joint | 不限 | 12k_FE | DE | 12 | 7.38 | 10.08 | 5.70 | 35 |
| joint | 不限 | 12k_FE | FE | 12 | 13.10 | 13.54 | 13.51 | 35 |
| joint | 不限 | 48k_DE | DE | 12 | 68.12 | 9.84 | 66.02 | 40 |
| joint | 不限 | 48k_DE | FE | 12 | 67.50 | 25.74 | 65.84 | 40 |
| source_continue | 不限 | 12k_FE | DE | 12 | 3.10 | 5.09 | 2.92 | 35 |
| source_continue | 不限 | 12k_FE | FE | 12 | 10.24 | 9.70 | 12.41 | 35 |
| source_continue | 不限 | 48k_DE | DE | 12 | 73.12 | 7.62 | 72.97 | 40 |
| source_continue | 不限 | 48k_DE | FE | 12 | 75.00 | 13.23 | 73.43 | 40 |

## 六对配对链（逐格差值）

| 折 | seed | 比较 | 组 | ΔAccuracy (pp) | ΔMacro-F1 (pp) |
| --- | --- | --- | --- | ---: | ---: |
| 0 | 0 | CE vs adabn（标签 vs 无监督 BN） | 12k_FE/DE | +82.86 | +85.61 |
| 0 | 1 | CE vs adabn（标签 vs 无监督 BN） | 12k_FE/DE | +68.57 | +69.35 |
| 0 | 2 | CE vs adabn（标签 vs 无监督 BN） | 12k_FE/DE | +74.29 | +81.45 |
| 1 | 0 | CE vs adabn（标签 vs 无监督 BN） | 12k_FE/DE | +74.29 | +76.25 |
| 1 | 1 | CE vs adabn（标签 vs 无监督 BN） | 12k_FE/DE | +71.43 | +75.37 |
| 1 | 2 | CE vs adabn（标签 vs 无监督 BN） | 12k_FE/DE | +71.43 | +76.80 |
| 2 | 0 | CE vs adabn（标签 vs 无监督 BN） | 12k_FE/DE | +80.00 | +81.16 |
| 2 | 1 | CE vs adabn（标签 vs 无监督 BN） | 12k_FE/DE | +82.86 | +87.13 |
| 2 | 2 | CE vs adabn（标签 vs 无监督 BN） | 12k_FE/DE | +77.14 | +77.33 |
| 3 | 0 | CE vs adabn（标签 vs 无监督 BN） | 12k_FE/DE | +65.71 | +71.43 |
| 3 | 1 | CE vs adabn（标签 vs 无监督 BN） | 12k_FE/DE | +88.57 | +91.40 |
| 3 | 2 | CE vs adabn（标签 vs 无监督 BN） | 12k_FE/DE | +74.29 | +74.50 |
| 0 | 0 | CE vs joint（有监督适配总量） | 12k_FE/DE | +94.29 | +93.70 |
| 0 | 1 | CE vs joint（有监督适配总量） | 12k_FE/DE | +94.29 | +92.58 |
| 0 | 2 | CE vs joint（有监督适配总量） | 12k_FE/DE | +88.57 | +89.83 |
| 1 | 0 | CE vs joint（有监督适配总量） | 12k_FE/DE | +94.29 | +94.58 |
| 1 | 1 | CE vs joint（有监督适配总量） | 12k_FE/DE | +94.29 | +93.73 |
| 1 | 2 | CE vs joint（有监督适配总量） | 12k_FE/DE | +77.14 | +79.25 |
| 2 | 0 | CE vs joint（有监督适配总量） | 12k_FE/DE | +82.86 | +86.49 |
| 2 | 1 | CE vs joint（有监督适配总量） | 12k_FE/DE | +97.14 | +97.24 |
| 2 | 2 | CE vs joint（有监督适配总量） | 12k_FE/DE | +62.86 | +76.77 |
| 3 | 0 | CE vs joint（有监督适配总量） | 12k_FE/DE | +85.71 | +85.24 |
| 3 | 1 | CE vs joint（有监督适配总量） | 12k_FE/DE | +100.00 | +100.00 |
| 3 | 2 | CE vs joint（有监督适配总量） | 12k_FE/DE | +85.71 | +84.39 |
| 0 | 0 | CE vs 源续训（标签本身值多少） | 12k_FE/DE | +94.29 | +93.70 |
| 0 | 1 | CE vs 源续训（标签本身值多少） | 12k_FE/DE | +94.29 | +93.99 |
| 0 | 2 | CE vs 源续训（标签本身值多少） | 12k_FE/DE | +94.29 | +94.06 |
| 1 | 0 | CE vs 源续训（标签本身值多少） | 12k_FE/DE | +91.43 | +91.07 |
| 1 | 1 | CE vs 源续训（标签本身值多少） | 12k_FE/DE | +97.14 | +97.24 |
| 1 | 2 | CE vs 源续训（标签本身值多少） | 12k_FE/DE | +77.14 | +78.72 |
| 2 | 0 | CE vs 源续训（标签本身值多少） | 12k_FE/DE | +94.29 | +93.70 |
| 2 | 1 | CE vs 源续训（标签本身值多少） | 12k_FE/DE | +97.14 | +97.24 |
| 2 | 2 | CE vs 源续训（标签本身值多少） | 12k_FE/DE | +94.29 | +93.72 |
| 3 | 0 | CE vs 源续训（标签本身值多少） | 12k_FE/DE | +88.57 | +88.75 |
| 3 | 1 | CE vs 源续训（标签本身值多少） | 12k_FE/DE | +100.00 | +100.00 |
| 3 | 2 | CE vs 源续训（标签本身值多少） | 12k_FE/DE | +85.71 | +85.00 |
| 0 | 0 | CE+MMD vs CE（对齐项值多少） | 12k_FE/DE | +0.00 | +0.00 |
| 0 | 1 | CE+MMD vs CE（对齐项值多少） | 12k_FE/DE | +0.00 | +0.00 |
| 0 | 2 | CE+MMD vs CE（对齐项值多少） | 12k_FE/DE | -2.86 | -3.54 |
| 1 | 0 | CE+MMD vs CE（对齐项值多少） | 12k_FE/DE | +2.86 | +2.65 |
| 1 | 1 | CE+MMD vs CE（对齐项值多少） | 12k_FE/DE | -2.86 | -3.54 |
| 1 | 2 | CE+MMD vs CE（对齐项值多少） | 12k_FE/DE | +5.71 | +6.47 |
| 2 | 0 | CE+MMD vs CE（对齐项值多少） | 12k_FE/DE | +0.00 | -0.16 |
| 2 | 1 | CE+MMD vs CE（对齐项值多少） | 12k_FE/DE | -2.86 | -3.54 |
| 2 | 2 | CE+MMD vs CE（对齐项值多少） | 12k_FE/DE | +0.00 | +0.00 |
| 3 | 0 | CE+MMD vs CE（对齐项值多少） | 12k_FE/DE | +5.71 | +5.83 |
| 3 | 1 | CE+MMD vs CE（对齐项值多少） | 12k_FE/DE | -2.86 | -2.91 |
| 3 | 2 | CE+MMD vs CE（对齐项值多少） | 12k_FE/DE | +0.00 | +0.00 |
| 0 | 0 | adabn vs joint（无标注 BN 重估） | 12k_FE/DE | +11.43 | +8.08 |
| 0 | 1 | adabn vs joint（无标注 BN 重估） | 12k_FE/DE | +25.71 | +23.23 |
| 0 | 2 | adabn vs joint（无标注 BN 重估） | 12k_FE/DE | +14.29 | +8.38 |
| 1 | 0 | adabn vs joint（无标注 BN 重估） | 12k_FE/DE | +20.00 | +18.33 |
| 1 | 1 | adabn vs joint（无标注 BN 重估） | 12k_FE/DE | +22.86 | +18.36 |
| 1 | 2 | adabn vs joint（无标注 BN 重估） | 12k_FE/DE | +5.71 | +2.44 |
| 2 | 0 | adabn vs joint（无标注 BN 重估） | 12k_FE/DE | +2.86 | +5.33 |
| 2 | 1 | adabn vs joint（无标注 BN 重估） | 12k_FE/DE | +14.29 | +10.10 |
| 2 | 2 | adabn vs joint（无标注 BN 重估） | 12k_FE/DE | -14.29 | -0.56 |
| 3 | 0 | adabn vs joint（无标注 BN 重估） | 12k_FE/DE | +20.00 | +13.81 |
| 3 | 1 | adabn vs joint（无标注 BN 重估） | 12k_FE/DE | +11.43 | +8.60 |
| 3 | 2 | adabn vs joint（无标注 BN 重估） | 12k_FE/DE | +11.43 | +9.89 |
| 0 | 0 | 源续训 vs joint（额外更新步数本身） | 12k_FE/DE | +0.00 | +0.00 |
| 0 | 0 | 源续训 vs joint（额外更新步数本身） | 12k_FE/DE | +0.00 | +0.00 |
| 0 | 0 | 源续训 vs joint（额外更新步数本身） | 12k_FE/DE | +0.00 | +0.00 |
| 0 | 1 | 源续训 vs joint（额外更新步数本身） | 12k_FE/DE | +0.00 | -1.40 |
| 0 | 1 | 源续训 vs joint（额外更新步数本身） | 12k_FE/DE | +0.00 | -1.40 |
| 0 | 1 | 源续训 vs joint（额外更新步数本身） | 12k_FE/DE | +0.00 | -1.40 |
| 0 | 2 | 源续训 vs joint（额外更新步数本身） | 12k_FE/DE | -5.71 | -4.23 |
| 0 | 2 | 源续训 vs joint（额外更新步数本身） | 12k_FE/DE | -5.71 | -4.23 |
| 0 | 2 | 源续训 vs joint（额外更新步数本身） | 12k_FE/DE | -5.71 | -4.23 |
| 1 | 0 | 源续训 vs joint（额外更新步数本身） | 12k_FE/DE | +2.86 | +3.51 |
| 1 | 0 | 源续训 vs joint（额外更新步数本身） | 12k_FE/DE | +2.86 | +3.51 |
| 1 | 0 | 源续训 vs joint（额外更新步数本身） | 12k_FE/DE | +2.86 | +3.51 |
| 1 | 1 | 源续训 vs joint（额外更新步数本身） | 12k_FE/DE | -2.86 | -3.51 |
| 1 | 1 | 源续训 vs joint（额外更新步数本身） | 12k_FE/DE | -2.86 | -3.51 |
| 1 | 1 | 源续训 vs joint（额外更新步数本身） | 12k_FE/DE | -2.86 | -3.51 |
| 1 | 2 | 源续训 vs joint（额外更新步数本身） | 12k_FE/DE | +0.00 | +0.53 |
| 1 | 2 | 源续训 vs joint（额外更新步数本身） | 12k_FE/DE | +0.00 | +0.53 |
| 1 | 2 | 源续训 vs joint（额外更新步数本身） | 12k_FE/DE | +0.00 | +0.53 |
| 2 | 0 | 源续训 vs joint（额外更新步数本身） | 12k_FE/DE | -11.43 | -7.21 |
| 2 | 0 | 源续训 vs joint（额外更新步数本身） | 12k_FE/DE | -11.43 | -7.21 |
| 2 | 0 | 源续训 vs joint（额外更新步数本身） | 12k_FE/DE | -11.43 | -7.21 |
| 2 | 1 | 源续训 vs joint（额外更新步数本身） | 12k_FE/DE | +0.00 | +0.00 |
| 2 | 1 | 源续训 vs joint（额外更新步数本身） | 12k_FE/DE | +0.00 | +0.00 |
| 2 | 1 | 源续训 vs joint（额外更新步数本身） | 12k_FE/DE | +0.00 | +0.00 |
| 2 | 2 | 源续训 vs joint（额外更新步数本身） | 12k_FE/DE | -31.43 | -16.95 |
| 2 | 2 | 源续训 vs joint（额外更新步数本身） | 12k_FE/DE | -31.43 | -16.95 |
| 2 | 2 | 源续训 vs joint（额外更新步数本身） | 12k_FE/DE | -31.43 | -16.95 |
| 3 | 0 | 源续训 vs joint（额外更新步数本身） | 12k_FE/DE | -2.86 | -3.51 |
| 3 | 0 | 源续训 vs joint（额外更新步数本身） | 12k_FE/DE | -2.86 | -3.51 |
| 3 | 0 | 源续训 vs joint（额外更新步数本身） | 12k_FE/DE | -2.86 | -3.51 |
| 3 | 1 | 源续训 vs joint（额外更新步数本身） | 12k_FE/DE | +0.00 | +0.00 |
| 3 | 1 | 源续训 vs joint（额外更新步数本身） | 12k_FE/DE | +0.00 | +0.00 |
| 3 | 1 | 源续训 vs joint（额外更新步数本身） | 12k_FE/DE | +0.00 | +0.00 |
| 3 | 2 | 源续训 vs joint（额外更新步数本身） | 12k_FE/DE | +0.00 | -0.61 |
| 3 | 2 | 源续训 vs joint（额外更新步数本身） | 12k_FE/DE | +0.00 | -0.61 |
| 3 | 2 | 源续训 vs joint（额外更新步数本身） | 12k_FE/DE | +0.00 | -0.61 |
| 0 | 0 | CE vs adabn（标签 vs 无监督 BN） | 12k_FE/FE | +62.86 | +65.75 |
| 0 | 1 | CE vs adabn（标签 vs 无监督 BN） | 12k_FE/FE | +74.29 | +79.47 |
| 0 | 2 | CE vs adabn（标签 vs 无监督 BN） | 12k_FE/FE | +68.57 | +67.92 |
| 1 | 0 | CE vs adabn（标签 vs 无监督 BN） | 12k_FE/FE | +54.29 | +58.35 |
| 1 | 1 | CE vs adabn（标签 vs 无监督 BN） | 12k_FE/FE | +51.43 | +53.53 |
| 1 | 2 | CE vs adabn（标签 vs 无监督 BN） | 12k_FE/FE | +65.71 | +66.48 |
| 2 | 0 | CE vs adabn（标签 vs 无监督 BN） | 12k_FE/FE | +54.29 | +60.31 |
| 2 | 1 | CE vs adabn（标签 vs 无监督 BN） | 12k_FE/FE | +68.57 | +65.52 |
| 2 | 2 | CE vs adabn（标签 vs 无监督 BN） | 12k_FE/FE | +65.71 | +68.31 |
| 3 | 0 | CE vs adabn（标签 vs 无监督 BN） | 12k_FE/FE | +60.00 | +61.49 |
| 3 | 1 | CE vs adabn（标签 vs 无监督 BN） | 12k_FE/FE | +80.00 | +76.59 |
| 3 | 2 | CE vs adabn（标签 vs 无监督 BN） | 12k_FE/FE | +54.29 | +56.24 |
| 0 | 0 | CE vs joint（有监督适配总量） | 12k_FE/FE | +74.29 | +77.92 |
| 0 | 1 | CE vs joint（有监督适配总量） | 12k_FE/FE | +91.43 | +91.74 |
| 0 | 2 | CE vs joint（有监督适配总量） | 12k_FE/FE | +91.43 | +87.92 |
| 1 | 0 | CE vs joint（有监督适配总量） | 12k_FE/FE | +91.43 | +91.69 |
| 1 | 1 | CE vs joint（有监督适配总量） | 12k_FE/FE | +85.71 | +81.51 |
| 1 | 2 | CE vs joint（有监督适配总量） | 12k_FE/FE | +74.29 | +73.88 |
| 2 | 0 | CE vs joint（有监督适配总量） | 12k_FE/FE | +71.43 | +67.52 |
| 2 | 1 | CE vs joint（有监督适配总量） | 12k_FE/FE | +88.57 | +85.99 |
| 2 | 2 | CE vs joint（有监督适配总量） | 12k_FE/FE | +54.29 | +69.97 |
| 3 | 0 | CE vs joint（有监督适配总量） | 12k_FE/FE | +88.57 | +85.11 |
| 3 | 1 | CE vs joint（有监督适配总量） | 12k_FE/FE | +94.29 | +94.58 |
| 3 | 2 | CE vs joint（有监督适配总量） | 12k_FE/FE | +71.43 | +68.58 |
| 0 | 0 | CE vs 源续训（标签本身值多少） | 12k_FE/FE | +74.29 | +77.30 |
| 0 | 1 | CE vs 源续训（标签本身值多少） | 12k_FE/FE | +94.29 | +94.41 |
| 0 | 2 | CE vs 源续训（标签本身值多少） | 12k_FE/FE | +88.57 | +88.79 |
| 1 | 0 | CE vs 源续训（标签本身值多少） | 12k_FE/FE | +82.86 | +77.40 |
| 1 | 1 | CE vs 源续训（标签本身值多少） | 12k_FE/FE | +91.43 | +92.62 |
| 1 | 2 | CE vs 源续训（标签本身值多少） | 12k_FE/FE | +74.29 | +75.72 |
| 2 | 0 | CE vs 源续训（标签本身值多少） | 12k_FE/FE | +88.57 | +84.15 |
| 2 | 1 | CE vs 源续训（标签本身值多少） | 12k_FE/FE | +82.86 | +77.02 |
| 2 | 2 | CE vs 源续训（标签本身值多少） | 12k_FE/FE | +65.71 | +68.90 |
| 3 | 0 | CE vs 源续训（标签本身值多少） | 12k_FE/FE | +91.43 | +86.12 |
| 3 | 1 | CE vs 源续训（标签本身值多少） | 12k_FE/FE | +91.43 | +87.92 |
| 3 | 2 | CE vs 源续训（标签本身值多少） | 12k_FE/FE | +85.71 | +79.25 |
| 0 | 0 | CE+MMD vs CE（对齐项值多少） | 12k_FE/FE | +0.00 | +0.00 |
| 0 | 1 | CE+MMD vs CE（对齐项值多少） | 12k_FE/FE | +0.00 | +0.18 |
| 0 | 2 | CE+MMD vs CE（对齐项值多少） | 12k_FE/FE | +0.00 | +0.00 |
| 1 | 0 | CE+MMD vs CE（对齐项值多少） | 12k_FE/FE | +2.86 | +2.90 |
| 1 | 1 | CE+MMD vs CE（对齐项值多少） | 12k_FE/FE | +0.00 | +0.00 |
| 1 | 2 | CE+MMD vs CE（对齐项值多少） | 12k_FE/FE | -2.86 | -2.65 |
| 2 | 0 | CE+MMD vs CE（对齐项值多少） | 12k_FE/FE | +0.00 | +0.00 |
| 2 | 1 | CE+MMD vs CE（对齐项值多少） | 12k_FE/FE | +2.86 | +1.93 |
| 2 | 2 | CE+MMD vs CE（对齐项值多少） | 12k_FE/FE | +0.00 | +0.00 |
| 3 | 0 | CE+MMD vs CE（对齐项值多少） | 12k_FE/FE | -2.86 | -2.65 |
| 3 | 1 | CE+MMD vs CE（对齐项值多少） | 12k_FE/FE | +0.00 | +0.00 |
| 3 | 2 | CE+MMD vs CE（对齐项值多少） | 12k_FE/FE | +0.00 | +0.00 |
| 0 | 0 | adabn vs joint（无标注 BN 重估） | 12k_FE/FE | +11.43 | +12.17 |
| 0 | 1 | adabn vs joint（无标注 BN 重估） | 12k_FE/FE | +17.14 | +12.27 |
| 0 | 2 | adabn vs joint（无标注 BN 重估） | 12k_FE/FE | +22.86 | +20.00 |
| 1 | 0 | adabn vs joint（无标注 BN 重估） | 12k_FE/FE | +37.14 | +33.33 |
| 1 | 1 | adabn vs joint（无标注 BN 重估） | 12k_FE/FE | +34.29 | +27.98 |
| 1 | 2 | adabn vs joint（无标注 BN 重估） | 12k_FE/FE | +8.57 | +7.40 |
| 2 | 0 | adabn vs joint（无标注 BN 重估） | 12k_FE/FE | +17.14 | +7.22 |
| 2 | 1 | adabn vs joint（无标注 BN 重估） | 12k_FE/FE | +20.00 | +20.46 |
| 2 | 2 | adabn vs joint（无标注 BN 重估） | 12k_FE/FE | -11.43 | +1.67 |
| 3 | 0 | adabn vs joint（无标注 BN 重估） | 12k_FE/FE | +28.57 | +23.62 |
| 3 | 1 | adabn vs joint（无标注 BN 重估） | 12k_FE/FE | +14.29 | +17.99 |
| 3 | 2 | adabn vs joint（无标注 BN 重估） | 12k_FE/FE | +17.14 | +12.33 |
| 0 | 0 | 源续训 vs joint（额外更新步数本身） | 12k_FE/FE | +0.00 | +0.62 |
| 0 | 0 | 源续训 vs joint（额外更新步数本身） | 12k_FE/FE | +0.00 | +0.62 |
| 0 | 0 | 源续训 vs joint（额外更新步数本身） | 12k_FE/FE | +0.00 | +0.62 |
| 0 | 1 | 源续训 vs joint（额外更新步数本身） | 12k_FE/FE | -2.86 | -2.67 |
| 0 | 1 | 源续训 vs joint（额外更新步数本身） | 12k_FE/FE | -2.86 | -2.67 |
| 0 | 1 | 源续训 vs joint（额外更新步数本身） | 12k_FE/FE | -2.86 | -2.67 |
| 0 | 2 | 源续训 vs joint（额外更新步数本身） | 12k_FE/FE | +2.86 | -0.87 |
| 0 | 2 | 源续训 vs joint（额外更新步数本身） | 12k_FE/FE | +2.86 | -0.87 |
| 0 | 2 | 源续训 vs joint（额外更新步数本身） | 12k_FE/FE | +2.86 | -0.87 |
| 1 | 0 | 源续训 vs joint（额外更新步数本身） | 12k_FE/FE | +8.57 | +14.29 |
| 1 | 0 | 源续训 vs joint（额外更新步数本身） | 12k_FE/FE | +8.57 | +14.29 |
| 1 | 0 | 源续训 vs joint（额外更新步数本身） | 12k_FE/FE | +8.57 | +14.29 |
| 1 | 1 | 源续训 vs joint（额外更新步数本身） | 12k_FE/FE | -5.71 | -11.11 |
| 1 | 1 | 源续训 vs joint（额外更新步数本身） | 12k_FE/FE | -5.71 | -11.11 |
| 1 | 1 | 源续训 vs joint（额外更新步数本身） | 12k_FE/FE | -5.71 | -11.11 |
| 1 | 2 | 源续训 vs joint（额外更新步数本身） | 12k_FE/FE | +0.00 | -1.84 |
| 1 | 2 | 源续训 vs joint（额外更新步数本身） | 12k_FE/FE | +0.00 | -1.84 |
| 1 | 2 | 源续训 vs joint（额外更新步数本身） | 12k_FE/FE | +0.00 | -1.84 |
| 2 | 0 | 源续训 vs joint（额外更新步数本身） | 12k_FE/FE | -17.14 | -16.63 |
| 2 | 0 | 源续训 vs joint（额外更新步数本身） | 12k_FE/FE | -17.14 | -16.63 |
| 2 | 0 | 源续训 vs joint（额外更新步数本身） | 12k_FE/FE | -17.14 | -16.63 |
| 2 | 1 | 源续训 vs joint（额外更新步数本身） | 12k_FE/FE | +5.71 | +8.96 |
| 2 | 1 | 源续训 vs joint（额外更新步数本身） | 12k_FE/FE | +5.71 | +8.96 |
| 2 | 1 | 源续训 vs joint（额外更新步数本身） | 12k_FE/FE | +5.71 | +8.96 |
| 2 | 2 | 源续训 vs joint（额外更新步数本身） | 12k_FE/FE | -11.43 | +1.07 |
| 2 | 2 | 源续训 vs joint（额外更新步数本身） | 12k_FE/FE | -11.43 | +1.07 |
| 2 | 2 | 源续训 vs joint（额外更新步数本身） | 12k_FE/FE | -11.43 | +1.07 |
| 3 | 0 | 源续训 vs joint（额外更新步数本身） | 12k_FE/FE | -2.86 | -1.01 |
| 3 | 0 | 源续训 vs joint（额外更新步数本身） | 12k_FE/FE | -2.86 | -1.01 |
| 3 | 0 | 源续训 vs joint（额外更新步数本身） | 12k_FE/FE | -2.86 | -1.01 |
| 3 | 1 | 源续训 vs joint（额外更新步数本身） | 12k_FE/FE | +2.86 | +6.67 |
| 3 | 1 | 源续训 vs joint（额外更新步数本身） | 12k_FE/FE | +2.86 | +6.67 |
| 3 | 1 | 源续训 vs joint（额外更新步数本身） | 12k_FE/FE | +2.86 | +6.67 |
| 3 | 2 | 源续训 vs joint（额外更新步数本身） | 12k_FE/FE | -14.29 | -10.67 |
| 3 | 2 | 源续训 vs joint（额外更新步数本身） | 12k_FE/FE | -14.29 | -10.67 |
| 3 | 2 | 源续训 vs joint（额外更新步数本身） | 12k_FE/FE | -14.29 | -10.67 |
| 0 | 0 | CE vs adabn（标签 vs 无监督 BN） | 48k_DE/DE | +7.50 | +8.44 |
| 0 | 1 | CE vs adabn（标签 vs 无监督 BN） | 48k_DE/DE | +20.00 | +23.36 |
| 0 | 2 | CE vs adabn（标签 vs 无监督 BN） | 48k_DE/DE | +7.50 | +7.12 |
| 1 | 0 | CE vs adabn（标签 vs 无监督 BN） | 48k_DE/DE | +5.00 | +4.95 |
| 1 | 1 | CE vs adabn（标签 vs 无监督 BN） | 48k_DE/DE | +12.50 | +14.98 |
| 1 | 2 | CE vs adabn（标签 vs 无监督 BN） | 48k_DE/DE | +25.00 | +28.33 |
| 2 | 0 | CE vs adabn（标签 vs 无监督 BN） | 48k_DE/DE | +20.00 | +15.65 |
| 2 | 1 | CE vs adabn（标签 vs 无监督 BN） | 48k_DE/DE | +20.00 | +21.43 |
| 2 | 2 | CE vs adabn（标签 vs 无监督 BN） | 48k_DE/DE | +17.50 | +19.89 |
| 3 | 0 | CE vs adabn（标签 vs 无监督 BN） | 48k_DE/DE | +15.00 | +16.64 |
| 3 | 1 | CE vs adabn（标签 vs 无监督 BN） | 48k_DE/DE | +25.00 | +26.36 |
| 3 | 2 | CE vs adabn（标签 vs 无监督 BN） | 48k_DE/DE | +15.00 | +15.84 |
| 0 | 0 | CE vs joint（有监督适配总量） | 48k_DE/DE | +22.50 | +23.93 |
| 0 | 1 | CE vs joint（有监督适配总量） | 48k_DE/DE | +17.50 | +19.40 |
| 0 | 2 | CE vs joint（有监督适配总量） | 48k_DE/DE | +17.50 | +16.86 |
| 1 | 0 | CE vs joint（有监督适配总量） | 48k_DE/DE | +35.00 | +38.24 |
| 1 | 1 | CE vs joint（有监督适配总量） | 48k_DE/DE | +20.00 | +24.18 |
| 1 | 2 | CE vs joint（有监督适配总量） | 48k_DE/DE | +15.00 | +16.25 |
| 2 | 0 | CE vs joint（有监督适配总量） | 48k_DE/DE | +10.00 | +10.21 |
| 2 | 1 | CE vs joint（有监督适配总量） | 48k_DE/DE | +40.00 | +38.93 |
| 2 | 2 | CE vs joint（有监督适配总量） | 48k_DE/DE | +25.00 | +37.47 |
| 3 | 0 | CE vs joint（有监督适配总量） | 48k_DE/DE | +27.50 | +31.19 |
| 3 | 1 | CE vs joint（有监督适配总量） | 48k_DE/DE | +30.00 | +28.83 |
| 3 | 2 | CE vs joint（有监督适配总量） | 48k_DE/DE | +10.00 | +11.91 |
| 0 | 0 | CE vs 源续训（标签本身值多少） | 48k_DE/DE | +12.50 | +11.48 |
| 0 | 1 | CE vs 源续训（标签本身值多少） | 48k_DE/DE | +15.00 | +15.39 |
| 0 | 2 | CE vs 源续训（标签本身值多少） | 48k_DE/DE | +12.50 | +11.49 |
| 1 | 0 | CE vs 源续训（标签本身值多少） | 48k_DE/DE | +5.00 | +4.95 |
| 1 | 1 | CE vs 源续训（标签本身值多少） | 48k_DE/DE | +22.50 | +25.68 |
| 1 | 2 | CE vs 源续训（标签本身值多少） | 48k_DE/DE | +7.50 | +7.57 |
| 2 | 0 | CE vs 源续训（标签本身值多少） | 48k_DE/DE | +27.50 | +24.42 |
| 2 | 1 | CE vs 源续训（标签本身值多少） | 48k_DE/DE | +30.00 | +31.20 |
| 2 | 2 | CE vs 源续训（标签本身值多少） | 48k_DE/DE | +22.50 | +25.05 |
| 3 | 0 | CE vs 源续训（标签本身值多少） | 48k_DE/DE | +25.00 | +26.76 |
| 3 | 1 | CE vs 源续训（标签本身值多少） | 48k_DE/DE | +20.00 | +19.27 |
| 3 | 2 | CE vs 源续训（标签本身值多少） | 48k_DE/DE | +10.00 | +10.79 |
| 0 | 0 | CE+MMD vs CE（对齐项值多少） | 48k_DE/DE | +0.00 | +0.00 |
| 0 | 1 | CE+MMD vs CE（对齐项值多少） | 48k_DE/DE | +2.50 | +2.61 |
| 0 | 2 | CE+MMD vs CE（对齐项值多少） | 48k_DE/DE | +0.00 | +0.00 |
| 1 | 0 | CE+MMD vs CE（对齐项值多少） | 48k_DE/DE | +2.50 | +2.64 |
| 1 | 1 | CE+MMD vs CE（对齐项值多少） | 48k_DE/DE | +0.00 | +0.00 |
| 1 | 2 | CE+MMD vs CE（对齐项值多少） | 48k_DE/DE | -2.50 | -2.81 |
| 2 | 0 | CE+MMD vs CE（对齐项值多少） | 48k_DE/DE | +0.00 | +0.00 |
| 2 | 1 | CE+MMD vs CE（对齐项值多少） | 48k_DE/DE | +0.00 | +0.00 |
| 2 | 2 | CE+MMD vs CE（对齐项值多少） | 48k_DE/DE | +2.50 | +2.28 |
| 3 | 0 | CE+MMD vs CE（对齐项值多少） | 48k_DE/DE | -2.50 | -2.64 |
| 3 | 1 | CE+MMD vs CE（对齐项值多少） | 48k_DE/DE | -2.50 | -2.61 |
| 3 | 2 | CE+MMD vs CE（对齐项值多少） | 48k_DE/DE | +0.00 | +0.00 |
| 0 | 0 | adabn vs joint（无标注 BN 重估） | 48k_DE/DE | +15.00 | +15.50 |
| 0 | 1 | adabn vs joint（无标注 BN 重估） | 48k_DE/DE | -2.50 | -3.96 |
| 0 | 2 | adabn vs joint（无标注 BN 重估） | 48k_DE/DE | +10.00 | +9.74 |
| 1 | 0 | adabn vs joint（无标注 BN 重估） | 48k_DE/DE | +30.00 | +33.29 |
| 1 | 1 | adabn vs joint（无标注 BN 重估） | 48k_DE/DE | +7.50 | +9.20 |
| 1 | 2 | adabn vs joint（无标注 BN 重估） | 48k_DE/DE | -10.00 | -12.08 |
| 2 | 0 | adabn vs joint（无标注 BN 重估） | 48k_DE/DE | -10.00 | -5.43 |
| 2 | 1 | adabn vs joint（无标注 BN 重估） | 48k_DE/DE | +20.00 | +17.50 |
| 2 | 2 | adabn vs joint（无标注 BN 重估） | 48k_DE/DE | +7.50 | +17.57 |
| 3 | 0 | adabn vs joint（无标注 BN 重估） | 48k_DE/DE | +12.50 | +14.55 |
| 3 | 1 | adabn vs joint（无标注 BN 重估） | 48k_DE/DE | +5.00 | +2.47 |
| 3 | 2 | adabn vs joint（无标注 BN 重估） | 48k_DE/DE | -5.00 | -3.94 |
| 0 | 0 | 源续训 vs joint（额外更新步数本身） | 48k_DE/DE | +10.00 | +12.45 |
| 0 | 0 | 源续训 vs joint（额外更新步数本身） | 48k_DE/DE | +10.00 | +12.45 |
| 0 | 0 | 源续训 vs joint（额外更新步数本身） | 48k_DE/DE | +10.00 | +12.45 |
| 0 | 1 | 源续训 vs joint（额外更新步数本身） | 48k_DE/DE | +2.50 | +4.01 |
| 0 | 1 | 源续训 vs joint（额外更新步数本身） | 48k_DE/DE | +2.50 | +4.01 |
| 0 | 1 | 源续训 vs joint（额外更新步数本身） | 48k_DE/DE | +2.50 | +4.01 |
| 0 | 2 | 源续训 vs joint（额外更新步数本身） | 48k_DE/DE | +5.00 | +5.38 |
| 0 | 2 | 源续训 vs joint（额外更新步数本身） | 48k_DE/DE | +5.00 | +5.38 |
| 0 | 2 | 源续训 vs joint（额外更新步数本身） | 48k_DE/DE | +5.00 | +5.38 |
| 1 | 0 | 源续训 vs joint（额外更新步数本身） | 48k_DE/DE | +30.00 | +33.29 |
| 1 | 0 | 源续训 vs joint（额外更新步数本身） | 48k_DE/DE | +30.00 | +33.29 |
| 1 | 0 | 源续训 vs joint（额外更新步数本身） | 48k_DE/DE | +30.00 | +33.29 |
| 1 | 1 | 源续训 vs joint（额外更新步数本身） | 48k_DE/DE | -2.50 | -1.50 |
| 1 | 1 | 源续训 vs joint（额外更新步数本身） | 48k_DE/DE | -2.50 | -1.50 |
| 1 | 1 | 源续训 vs joint（额外更新步数本身） | 48k_DE/DE | -2.50 | -1.50 |
| 1 | 2 | 源续训 vs joint（额外更新步数本身） | 48k_DE/DE | +7.50 | +8.68 |
| 1 | 2 | 源续训 vs joint（额外更新步数本身） | 48k_DE/DE | +7.50 | +8.68 |
| 1 | 2 | 源续训 vs joint（额外更新步数本身） | 48k_DE/DE | +7.50 | +8.68 |
| 2 | 0 | 源续训 vs joint（额外更新步数本身） | 48k_DE/DE | -17.50 | -14.21 |
| 2 | 0 | 源续训 vs joint（额外更新步数本身） | 48k_DE/DE | -17.50 | -14.21 |
| 2 | 0 | 源续训 vs joint（额外更新步数本身） | 48k_DE/DE | -17.50 | -14.21 |
| 2 | 1 | 源续训 vs joint（额外更新步数本身） | 48k_DE/DE | +10.00 | +7.73 |
| 2 | 1 | 源续训 vs joint（额外更新步数本身） | 48k_DE/DE | +10.00 | +7.73 |
| 2 | 1 | 源续训 vs joint（额外更新步数本身） | 48k_DE/DE | +10.00 | +7.73 |
| 2 | 2 | 源续训 vs joint（额外更新步数本身） | 48k_DE/DE | +2.50 | +12.41 |
| 2 | 2 | 源续训 vs joint（额外更新步数本身） | 48k_DE/DE | +2.50 | +12.41 |
| 2 | 2 | 源续训 vs joint（额外更新步数本身） | 48k_DE/DE | +2.50 | +12.41 |
| 3 | 0 | 源续训 vs joint（额外更新步数本身） | 48k_DE/DE | +2.50 | +4.42 |
| 3 | 0 | 源续训 vs joint（额外更新步数本身） | 48k_DE/DE | +2.50 | +4.42 |
| 3 | 0 | 源续训 vs joint（额外更新步数本身） | 48k_DE/DE | +2.50 | +4.42 |
| 3 | 1 | 源续训 vs joint（额外更新步数本身） | 48k_DE/DE | +10.00 | +9.56 |
| 3 | 1 | 源续训 vs joint（额外更新步数本身） | 48k_DE/DE | +10.00 | +9.56 |
| 3 | 1 | 源续训 vs joint（额外更新步数本身） | 48k_DE/DE | +10.00 | +9.56 |
| 3 | 2 | 源续训 vs joint（额外更新步数本身） | 48k_DE/DE | +0.00 | +1.12 |
| 3 | 2 | 源续训 vs joint（额外更新步数本身） | 48k_DE/DE | +0.00 | +1.12 |
| 3 | 2 | 源续训 vs joint（额外更新步数本身） | 48k_DE/DE | +0.00 | +1.12 |
| 0 | 0 | CE vs adabn（标签 vs 无监督 BN） | 48k_DE/FE | +27.50 | +26.97 |
| 0 | 1 | CE vs adabn（标签 vs 无监督 BN） | 48k_DE/FE | +12.50 | +15.27 |
| 0 | 2 | CE vs adabn（标签 vs 无监督 BN） | 48k_DE/FE | +22.50 | +20.42 |
| 1 | 0 | CE vs adabn（标签 vs 无监督 BN） | 48k_DE/FE | +37.50 | +34.66 |
| 1 | 1 | CE vs adabn（标签 vs 无监督 BN） | 48k_DE/FE | +25.00 | +22.49 |
| 1 | 2 | CE vs adabn（标签 vs 无监督 BN） | 48k_DE/FE | +15.00 | +16.74 |
| 2 | 0 | CE vs adabn（标签 vs 无监督 BN） | 48k_DE/FE | +15.00 | +16.67 |
| 2 | 1 | CE vs adabn（标签 vs 无监督 BN） | 48k_DE/FE | +12.50 | +7.51 |
| 2 | 2 | CE vs adabn（标签 vs 无监督 BN） | 48k_DE/FE | +25.00 | +29.99 |
| 3 | 0 | CE vs adabn（标签 vs 无监督 BN） | 48k_DE/FE | +12.50 | +14.68 |
| 3 | 1 | CE vs adabn（标签 vs 无监督 BN） | 48k_DE/FE | +40.00 | +23.25 |
| 3 | 2 | CE vs adabn（标签 vs 无监督 BN） | 48k_DE/FE | +7.50 | +7.84 |
| 0 | 0 | CE vs joint（有监督适配总量） | 48k_DE/FE | +37.50 | +46.03 |
| 0 | 1 | CE vs joint（有监督适配总量） | 48k_DE/FE | +17.50 | +23.66 |
| 0 | 2 | CE vs joint（有监督适配总量） | 48k_DE/FE | +30.00 | +28.37 |
| 1 | 0 | CE vs joint（有监督适配总量） | 48k_DE/FE | +92.50 | +87.73 |
| 1 | 1 | CE vs joint（有监督适配总量） | 48k_DE/FE | +20.00 | +21.89 |
| 1 | 2 | CE vs joint（有监督适配总量） | 48k_DE/FE | +7.50 | +8.58 |
| 2 | 0 | CE vs joint（有监督适配总量） | 48k_DE/FE | +2.50 | +1.96 |
| 2 | 1 | CE vs joint（有监督适配总量） | 48k_DE/FE | +52.50 | +50.86 |
| 2 | 2 | CE vs joint（有监督适配总量） | 48k_DE/FE | +35.00 | +51.62 |
| 3 | 0 | CE vs joint（有监督适配总量） | 48k_DE/FE | +32.50 | +45.19 |
| 3 | 1 | CE vs joint（有监督适配总量） | 48k_DE/FE | +47.50 | +26.77 |
| 3 | 2 | CE vs joint（有监督适配总量） | 48k_DE/FE | +2.50 | +3.72 |
| 0 | 0 | CE vs 源续训（标签本身值多少） | 48k_DE/FE | +30.00 | +31.51 |
| 0 | 1 | CE vs 源续训（标签本身值多少） | 48k_DE/FE | +37.50 | +39.14 |
| 0 | 2 | CE vs 源续训（标签本身值多少） | 48k_DE/FE | +10.00 | +7.60 |
| 1 | 0 | CE vs 源续训（标签本身值多少） | 48k_DE/FE | +40.00 | +38.51 |
| 1 | 1 | CE vs 源续训（标签本身值多少） | 48k_DE/FE | +27.50 | +36.68 |
| 1 | 2 | CE vs 源续训（标签本身值多少） | 48k_DE/FE | +0.00 | +0.00 |
| 2 | 0 | CE vs 源续训（标签本身值多少） | 48k_DE/FE | +17.50 | +21.21 |
| 2 | 1 | CE vs 源续训（标签本身值多少） | 48k_DE/FE | +32.50 | +43.45 |
| 2 | 2 | CE vs 源续训（标签本身值多少） | 48k_DE/FE | +22.50 | +32.50 |
| 3 | 0 | CE vs 源续训（标签本身值多少） | 48k_DE/FE | +22.50 | +19.84 |
| 3 | 1 | CE vs 源续训（标签本身值多少） | 48k_DE/FE | +40.00 | +26.47 |
| 3 | 2 | CE vs 源续训（标签本身值多少） | 48k_DE/FE | +7.50 | +8.42 |
| 0 | 0 | CE+MMD vs CE（对齐项值多少） | 48k_DE/FE | +0.00 | +0.00 |
| 0 | 1 | CE+MMD vs CE（对齐项值多少） | 48k_DE/FE | +2.50 | +2.70 |
| 0 | 2 | CE+MMD vs CE（对齐项值多少） | 48k_DE/FE | +0.00 | +0.00 |
| 1 | 0 | CE+MMD vs CE（对齐项值多少） | 48k_DE/FE | +2.50 | +2.70 |
| 1 | 1 | CE+MMD vs CE（对齐项值多少） | 48k_DE/FE | +0.00 | +0.00 |
| 1 | 2 | CE+MMD vs CE（对齐项值多少） | 48k_DE/FE | -2.50 | -2.91 |
| 2 | 0 | CE+MMD vs CE（对齐项值多少） | 48k_DE/FE | +0.00 | +0.00 |
| 2 | 1 | CE+MMD vs CE（对齐项值多少） | 48k_DE/FE | +0.00 | +0.00 |
| 2 | 2 | CE+MMD vs CE（对齐项值多少） | 48k_DE/FE | +0.00 | +0.00 |
| 3 | 0 | CE+MMD vs CE（对齐项值多少） | 48k_DE/FE | -2.50 | -2.70 |
| 3 | 1 | CE+MMD vs CE（对齐项值多少） | 48k_DE/FE | -2.50 | -2.70 |
| 3 | 2 | CE+MMD vs CE（对齐项值多少） | 48k_DE/FE | +0.00 | +0.00 |
| 0 | 0 | adabn vs joint（无标注 BN 重估） | 48k_DE/FE | +10.00 | +19.07 |
| 0 | 1 | adabn vs joint（无标注 BN 重估） | 48k_DE/FE | +5.00 | +8.39 |
| 0 | 2 | adabn vs joint（无标注 BN 重估） | 48k_DE/FE | +7.50 | +7.95 |
| 1 | 0 | adabn vs joint（无标注 BN 重估） | 48k_DE/FE | +55.00 | +53.07 |
| 1 | 1 | adabn vs joint（无标注 BN 重估） | 48k_DE/FE | -5.00 | -0.60 |
| 1 | 2 | adabn vs joint（无标注 BN 重估） | 48k_DE/FE | -7.50 | -8.17 |
| 2 | 0 | adabn vs joint（无标注 BN 重估） | 48k_DE/FE | -12.50 | -14.71 |
| 2 | 1 | adabn vs joint（无标注 BN 重估） | 48k_DE/FE | +40.00 | +43.35 |
| 2 | 2 | adabn vs joint（无标注 BN 重估） | 48k_DE/FE | +10.00 | +21.63 |
| 3 | 0 | adabn vs joint（无标注 BN 重估） | 48k_DE/FE | +20.00 | +30.51 |
| 3 | 1 | adabn vs joint（无标注 BN 重估） | 48k_DE/FE | +7.50 | +3.52 |
| 3 | 2 | adabn vs joint（无标注 BN 重估） | 48k_DE/FE | -5.00 | -4.13 |
| 0 | 0 | 源续训 vs joint（额外更新步数本身） | 48k_DE/FE | +7.50 | +14.52 |
| 0 | 0 | 源续训 vs joint（额外更新步数本身） | 48k_DE/FE | +7.50 | +14.52 |
| 0 | 0 | 源续训 vs joint（额外更新步数本身） | 48k_DE/FE | +7.50 | +14.52 |
| 0 | 1 | 源续训 vs joint（额外更新步数本身） | 48k_DE/FE | -20.00 | -15.48 |
| 0 | 1 | 源续训 vs joint（额外更新步数本身） | 48k_DE/FE | -20.00 | -15.48 |
| 0 | 1 | 源续训 vs joint（额外更新步数本身） | 48k_DE/FE | -20.00 | -15.48 |
| 0 | 2 | 源续训 vs joint（额外更新步数本身） | 48k_DE/FE | +20.00 | +20.78 |
| 0 | 2 | 源续训 vs joint（额外更新步数本身） | 48k_DE/FE | +20.00 | +20.78 |
| 0 | 2 | 源续训 vs joint（额外更新步数本身） | 48k_DE/FE | +20.00 | +20.78 |
| 1 | 0 | 源续训 vs joint（额外更新步数本身） | 48k_DE/FE | +52.50 | +49.22 |
| 1 | 0 | 源续训 vs joint（额外更新步数本身） | 48k_DE/FE | +52.50 | +49.22 |
| 1 | 0 | 源续训 vs joint（额外更新步数本身） | 48k_DE/FE | +52.50 | +49.22 |
| 1 | 1 | 源续训 vs joint（额外更新步数本身） | 48k_DE/FE | -7.50 | -14.79 |
| 1 | 1 | 源续训 vs joint（额外更新步数本身） | 48k_DE/FE | -7.50 | -14.79 |
| 1 | 1 | 源续训 vs joint（额外更新步数本身） | 48k_DE/FE | -7.50 | -14.79 |
| 1 | 2 | 源续训 vs joint（额外更新步数本身） | 48k_DE/FE | +7.50 | +8.58 |
| 1 | 2 | 源续训 vs joint（额外更新步数本身） | 48k_DE/FE | +7.50 | +8.58 |
| 1 | 2 | 源续训 vs joint（额外更新步数本身） | 48k_DE/FE | +7.50 | +8.58 |
| 2 | 0 | 源续训 vs joint（额外更新步数本身） | 48k_DE/FE | -15.00 | -19.25 |
| 2 | 0 | 源续训 vs joint（额外更新步数本身） | 48k_DE/FE | -15.00 | -19.25 |
| 2 | 0 | 源续训 vs joint（额外更新步数本身） | 48k_DE/FE | -15.00 | -19.25 |
| 2 | 1 | 源续训 vs joint（额外更新步数本身） | 48k_DE/FE | +20.00 | +7.41 |
| 2 | 1 | 源续训 vs joint（额外更新步数本身） | 48k_DE/FE | +20.00 | +7.41 |
| 2 | 1 | 源续训 vs joint（额外更新步数本身） | 48k_DE/FE | +20.00 | +7.41 |
| 2 | 2 | 源续训 vs joint（额外更新步数本身） | 48k_DE/FE | +12.50 | +19.12 |
| 2 | 2 | 源续训 vs joint（额外更新步数本身） | 48k_DE/FE | +12.50 | +19.12 |
| 2 | 2 | 源续训 vs joint（额外更新步数本身） | 48k_DE/FE | +12.50 | +19.12 |
| 3 | 0 | 源续训 vs joint（额外更新步数本身） | 48k_DE/FE | +10.00 | +25.34 |
| 3 | 0 | 源续训 vs joint（额外更新步数本身） | 48k_DE/FE | +10.00 | +25.34 |
| 3 | 0 | 源续训 vs joint（额外更新步数本身） | 48k_DE/FE | +10.00 | +25.34 |
| 3 | 1 | 源续训 vs joint（额外更新步数本身） | 48k_DE/FE | +7.50 | +0.29 |
| 3 | 1 | 源续训 vs joint（额外更新步数本身） | 48k_DE/FE | +7.50 | +0.29 |
| 3 | 1 | 源续训 vs joint（额外更新步数本身） | 48k_DE/FE | +7.50 | +0.29 |
| 3 | 2 | 源续训 vs joint（额外更新步数本身） | 48k_DE/FE | -5.00 | -4.71 |
| 3 | 2 | 源续训 vs joint（额外更新步数本身） | 48k_DE/FE | -5.00 | -4.71 |
| 3 | 2 | 源续训 vs joint（额外更新步数本身） | 48k_DE/FE | -5.00 | -4.71 |

## 负增量

以下负差值全部保留，不择优选报告。
- fold=0 seed=2 12k_FE/DE，CE+MMD vs CE（对齐项值多少）：Accuracy -2.86、Macro-F1 -3.54 个百分点。
- fold=1 seed=1 12k_FE/DE，CE+MMD vs CE（对齐项值多少）：Accuracy -2.86、Macro-F1 -3.54 个百分点。
- fold=2 seed=0 12k_FE/DE，CE+MMD vs CE（对齐项值多少）：Accuracy +0.00、Macro-F1 -0.16 个百分点。
- fold=2 seed=1 12k_FE/DE，CE+MMD vs CE（对齐项值多少）：Accuracy -2.86、Macro-F1 -3.54 个百分点。
- fold=3 seed=1 12k_FE/DE，CE+MMD vs CE（对齐项值多少）：Accuracy -2.86、Macro-F1 -2.91 个百分点。
- fold=2 seed=2 12k_FE/DE，adabn vs joint（无标注 BN 重估）：Accuracy -14.29、Macro-F1 -0.56 个百分点。
- fold=0 seed=1 12k_FE/DE，源续训 vs joint（额外更新步数本身）：Accuracy +0.00、Macro-F1 -1.40 个百分点。
- fold=0 seed=1 12k_FE/DE，源续训 vs joint（额外更新步数本身）：Accuracy +0.00、Macro-F1 -1.40 个百分点。
- fold=0 seed=1 12k_FE/DE，源续训 vs joint（额外更新步数本身）：Accuracy +0.00、Macro-F1 -1.40 个百分点。
- fold=0 seed=2 12k_FE/DE，源续训 vs joint（额外更新步数本身）：Accuracy -5.71、Macro-F1 -4.23 个百分点。
- fold=0 seed=2 12k_FE/DE，源续训 vs joint（额外更新步数本身）：Accuracy -5.71、Macro-F1 -4.23 个百分点。
- fold=0 seed=2 12k_FE/DE，源续训 vs joint（额外更新步数本身）：Accuracy -5.71、Macro-F1 -4.23 个百分点。
- fold=1 seed=1 12k_FE/DE，源续训 vs joint（额外更新步数本身）：Accuracy -2.86、Macro-F1 -3.51 个百分点。
- fold=1 seed=1 12k_FE/DE，源续训 vs joint（额外更新步数本身）：Accuracy -2.86、Macro-F1 -3.51 个百分点。
- fold=1 seed=1 12k_FE/DE，源续训 vs joint（额外更新步数本身）：Accuracy -2.86、Macro-F1 -3.51 个百分点。
- fold=2 seed=0 12k_FE/DE，源续训 vs joint（额外更新步数本身）：Accuracy -11.43、Macro-F1 -7.21 个百分点。
- fold=2 seed=0 12k_FE/DE，源续训 vs joint（额外更新步数本身）：Accuracy -11.43、Macro-F1 -7.21 个百分点。
- fold=2 seed=0 12k_FE/DE，源续训 vs joint（额外更新步数本身）：Accuracy -11.43、Macro-F1 -7.21 个百分点。
- fold=2 seed=2 12k_FE/DE，源续训 vs joint（额外更新步数本身）：Accuracy -31.43、Macro-F1 -16.95 个百分点。
- fold=2 seed=2 12k_FE/DE，源续训 vs joint（额外更新步数本身）：Accuracy -31.43、Macro-F1 -16.95 个百分点。
- fold=2 seed=2 12k_FE/DE，源续训 vs joint（额外更新步数本身）：Accuracy -31.43、Macro-F1 -16.95 个百分点。
- fold=3 seed=0 12k_FE/DE，源续训 vs joint（额外更新步数本身）：Accuracy -2.86、Macro-F1 -3.51 个百分点。
- fold=3 seed=0 12k_FE/DE，源续训 vs joint（额外更新步数本身）：Accuracy -2.86、Macro-F1 -3.51 个百分点。
- fold=3 seed=0 12k_FE/DE，源续训 vs joint（额外更新步数本身）：Accuracy -2.86、Macro-F1 -3.51 个百分点。
- fold=3 seed=2 12k_FE/DE，源续训 vs joint（额外更新步数本身）：Accuracy +0.00、Macro-F1 -0.61 个百分点。
- fold=3 seed=2 12k_FE/DE，源续训 vs joint（额外更新步数本身）：Accuracy +0.00、Macro-F1 -0.61 个百分点。
- fold=3 seed=2 12k_FE/DE，源续训 vs joint（额外更新步数本身）：Accuracy +0.00、Macro-F1 -0.61 个百分点。
- fold=1 seed=2 12k_FE/FE，CE+MMD vs CE（对齐项值多少）：Accuracy -2.86、Macro-F1 -2.65 个百分点。
- fold=3 seed=0 12k_FE/FE，CE+MMD vs CE（对齐项值多少）：Accuracy -2.86、Macro-F1 -2.65 个百分点。
- fold=2 seed=2 12k_FE/FE，adabn vs joint（无标注 BN 重估）：Accuracy -11.43、Macro-F1 +1.67 个百分点。
- fold=0 seed=1 12k_FE/FE，源续训 vs joint（额外更新步数本身）：Accuracy -2.86、Macro-F1 -2.67 个百分点。
- fold=0 seed=1 12k_FE/FE，源续训 vs joint（额外更新步数本身）：Accuracy -2.86、Macro-F1 -2.67 个百分点。
- fold=0 seed=1 12k_FE/FE，源续训 vs joint（额外更新步数本身）：Accuracy -2.86、Macro-F1 -2.67 个百分点。
- fold=0 seed=2 12k_FE/FE，源续训 vs joint（额外更新步数本身）：Accuracy +2.86、Macro-F1 -0.87 个百分点。
- fold=0 seed=2 12k_FE/FE，源续训 vs joint（额外更新步数本身）：Accuracy +2.86、Macro-F1 -0.87 个百分点。
- fold=0 seed=2 12k_FE/FE，源续训 vs joint（额外更新步数本身）：Accuracy +2.86、Macro-F1 -0.87 个百分点。
- fold=1 seed=1 12k_FE/FE，源续训 vs joint（额外更新步数本身）：Accuracy -5.71、Macro-F1 -11.11 个百分点。
- fold=1 seed=1 12k_FE/FE，源续训 vs joint（额外更新步数本身）：Accuracy -5.71、Macro-F1 -11.11 个百分点。
- fold=1 seed=1 12k_FE/FE，源续训 vs joint（额外更新步数本身）：Accuracy -5.71、Macro-F1 -11.11 个百分点。
- fold=1 seed=2 12k_FE/FE，源续训 vs joint（额外更新步数本身）：Accuracy +0.00、Macro-F1 -1.84 个百分点。
- fold=1 seed=2 12k_FE/FE，源续训 vs joint（额外更新步数本身）：Accuracy +0.00、Macro-F1 -1.84 个百分点。
- fold=1 seed=2 12k_FE/FE，源续训 vs joint（额外更新步数本身）：Accuracy +0.00、Macro-F1 -1.84 个百分点。
- fold=2 seed=0 12k_FE/FE，源续训 vs joint（额外更新步数本身）：Accuracy -17.14、Macro-F1 -16.63 个百分点。
- fold=2 seed=0 12k_FE/FE，源续训 vs joint（额外更新步数本身）：Accuracy -17.14、Macro-F1 -16.63 个百分点。
- fold=2 seed=0 12k_FE/FE，源续训 vs joint（额外更新步数本身）：Accuracy -17.14、Macro-F1 -16.63 个百分点。
- fold=2 seed=2 12k_FE/FE，源续训 vs joint（额外更新步数本身）：Accuracy -11.43、Macro-F1 +1.07 个百分点。
- fold=2 seed=2 12k_FE/FE，源续训 vs joint（额外更新步数本身）：Accuracy -11.43、Macro-F1 +1.07 个百分点。
- fold=2 seed=2 12k_FE/FE，源续训 vs joint（额外更新步数本身）：Accuracy -11.43、Macro-F1 +1.07 个百分点。
- fold=3 seed=0 12k_FE/FE，源续训 vs joint（额外更新步数本身）：Accuracy -2.86、Macro-F1 -1.01 个百分点。
- fold=3 seed=0 12k_FE/FE，源续训 vs joint（额外更新步数本身）：Accuracy -2.86、Macro-F1 -1.01 个百分点。
- fold=3 seed=0 12k_FE/FE，源续训 vs joint（额外更新步数本身）：Accuracy -2.86、Macro-F1 -1.01 个百分点。
- fold=3 seed=2 12k_FE/FE，源续训 vs joint（额外更新步数本身）：Accuracy -14.29、Macro-F1 -10.67 个百分点。
- fold=3 seed=2 12k_FE/FE，源续训 vs joint（额外更新步数本身）：Accuracy -14.29、Macro-F1 -10.67 个百分点。
- fold=3 seed=2 12k_FE/FE，源续训 vs joint（额外更新步数本身）：Accuracy -14.29、Macro-F1 -10.67 个百分点。
- fold=1 seed=2 48k_DE/DE，CE+MMD vs CE（对齐项值多少）：Accuracy -2.50、Macro-F1 -2.81 个百分点。
- fold=3 seed=0 48k_DE/DE，CE+MMD vs CE（对齐项值多少）：Accuracy -2.50、Macro-F1 -2.64 个百分点。
- fold=3 seed=1 48k_DE/DE，CE+MMD vs CE（对齐项值多少）：Accuracy -2.50、Macro-F1 -2.61 个百分点。
- fold=0 seed=1 48k_DE/DE，adabn vs joint（无标注 BN 重估）：Accuracy -2.50、Macro-F1 -3.96 个百分点。
- fold=1 seed=2 48k_DE/DE，adabn vs joint（无标注 BN 重估）：Accuracy -10.00、Macro-F1 -12.08 个百分点。
- fold=2 seed=0 48k_DE/DE，adabn vs joint（无标注 BN 重估）：Accuracy -10.00、Macro-F1 -5.43 个百分点。
- fold=3 seed=2 48k_DE/DE，adabn vs joint（无标注 BN 重估）：Accuracy -5.00、Macro-F1 -3.94 个百分点。
- fold=1 seed=1 48k_DE/DE，源续训 vs joint（额外更新步数本身）：Accuracy -2.50、Macro-F1 -1.50 个百分点。
- fold=1 seed=1 48k_DE/DE，源续训 vs joint（额外更新步数本身）：Accuracy -2.50、Macro-F1 -1.50 个百分点。
- fold=1 seed=1 48k_DE/DE，源续训 vs joint（额外更新步数本身）：Accuracy -2.50、Macro-F1 -1.50 个百分点。
- fold=2 seed=0 48k_DE/DE，源续训 vs joint（额外更新步数本身）：Accuracy -17.50、Macro-F1 -14.21 个百分点。
- fold=2 seed=0 48k_DE/DE，源续训 vs joint（额外更新步数本身）：Accuracy -17.50、Macro-F1 -14.21 个百分点。
- fold=2 seed=0 48k_DE/DE，源续训 vs joint（额外更新步数本身）：Accuracy -17.50、Macro-F1 -14.21 个百分点。
- fold=1 seed=2 48k_DE/FE，CE+MMD vs CE（对齐项值多少）：Accuracy -2.50、Macro-F1 -2.91 个百分点。
- fold=3 seed=0 48k_DE/FE，CE+MMD vs CE（对齐项值多少）：Accuracy -2.50、Macro-F1 -2.70 个百分点。
- fold=3 seed=1 48k_DE/FE，CE+MMD vs CE（对齐项值多少）：Accuracy -2.50、Macro-F1 -2.70 个百分点。
- fold=1 seed=1 48k_DE/FE，adabn vs joint（无标注 BN 重估）：Accuracy -5.00、Macro-F1 -0.60 个百分点。
- fold=1 seed=2 48k_DE/FE，adabn vs joint（无标注 BN 重估）：Accuracy -7.50、Macro-F1 -8.17 个百分点。
- fold=2 seed=0 48k_DE/FE，adabn vs joint（无标注 BN 重估）：Accuracy -12.50、Macro-F1 -14.71 个百分点。
- fold=3 seed=2 48k_DE/FE，adabn vs joint（无标注 BN 重估）：Accuracy -5.00、Macro-F1 -4.13 个百分点。
- fold=0 seed=1 48k_DE/FE，源续训 vs joint（额外更新步数本身）：Accuracy -20.00、Macro-F1 -15.48 个百分点。
- fold=0 seed=1 48k_DE/FE，源续训 vs joint（额外更新步数本身）：Accuracy -20.00、Macro-F1 -15.48 个百分点。
- fold=0 seed=1 48k_DE/FE，源续训 vs joint（额外更新步数本身）：Accuracy -20.00、Macro-F1 -15.48 个百分点。
- fold=1 seed=1 48k_DE/FE，源续训 vs joint（额外更新步数本身）：Accuracy -7.50、Macro-F1 -14.79 个百分点。
- fold=1 seed=1 48k_DE/FE，源续训 vs joint（额外更新步数本身）：Accuracy -7.50、Macro-F1 -14.79 个百分点。
- fold=1 seed=1 48k_DE/FE，源续训 vs joint（额外更新步数本身）：Accuracy -7.50、Macro-F1 -14.79 个百分点。
- fold=2 seed=0 48k_DE/FE，源续训 vs joint（额外更新步数本身）：Accuracy -15.00、Macro-F1 -19.25 个百分点。
- fold=2 seed=0 48k_DE/FE，源续训 vs joint（额外更新步数本身）：Accuracy -15.00、Macro-F1 -19.25 个百分点。
- fold=2 seed=0 48k_DE/FE，源续训 vs joint（额外更新步数本身）：Accuracy -15.00、Macro-F1 -19.25 个百分点。
- fold=3 seed=2 48k_DE/FE，源续训 vs joint（额外更新步数本身）：Accuracy -5.00、Macro-F1 -4.71 个百分点。
- fold=3 seed=2 48k_DE/FE，源续训 vs joint（额外更新步数本身）：Accuracy -5.00、Macro-F1 -4.71 个百分点。
- fold=3 seed=2 48k_DE/FE，源续训 vs joint（额外更新步数本身）：Accuracy -5.00、Macro-F1 -4.71 个百分点。

## 结论边界

- 标注池与 fold/seed 无关（选标只用无标签信号），所以本轮没有划分轴可分层报告；看不到 tsplit 维度的方向反转，是协议 §4 塌缩的既定代价。
- 目标域无 Normal 类，正常召回为 null，Macro-F1 只平均有真值的三类，不能验证正常误报率。
- 同文件的 DE/FE 是两条记录但不是独立采集，逐格值不是独立样本。
- 本轮 `source_continue` 只对齐额外更新步数，不声称匹配全部 FLOPs。
- 跨版本纪律：v6 测试列表与 v5 不同，禁止与 v5 任何数字相减。
