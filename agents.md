# v6 项目规则

## 原则（唯一一条）

**实验完整，工程简单。** 代码只服务"跑完一个固定的实验矩阵并把结果如实落盘"，不为通用性、可扩展性、健壮性表演。明确禁止：数据指纹、复杂身份校验、协议 `schema`、大型实验框架、调度器／编排器、按概念另建的模块与抽象层（不建 `Arm`/`Protocol`/`Registry`/`Base` 那类东西，臂就是字符串分支加一个 dict）。必要函数留在既有数据、模型、训练、评估模块里；能 20 行写清的不封装。

允许且必须保留的少数几项检查，它们是数据完整性而不是防御：信号为空／非有限／不足一窗报错；缺通道或候选键不唯一报错；48k 只允许 `resample_poly(x,1,4)` 降采样；池与测试列表不得重叠；被当作标注训练的文件必须来自 `selection.json` 的池；测试前向无梯度且不改 BN。除此之外不加校验，不写"以防万一"的分支。

## 协议（已定稿，训练前冻结，见 `docs/v6协议草案.md`）

- 选标是**协议组件，不是研究问题**：不与 v5 的 `random13`/`stratified13` 作任何配对或引用其数字，禁止跨版本相减。
- 五步规则：目标信号 → 12 kHz → 512 点不重叠切窗 → `logspec512` → 文件内窗口均值 → DE+FE 拼接 512 维 → 短记录过滤（窗口数 < 0.5 × 中位，只约束进池不约束评估）→ 候选集上按维 z-score → k-means 搜 `k∈[6,20]` 取**最小的满足 `silhouette(k) ≥ 0.95 × max silhouette`** 的 `k`（`n_init=25`、`random_state=0`）→ 每簇取 medoid，单例簇排在名额序列最后。预算写成 `K = Σ_j max(1, round(p·n_j))`，本轮 `p=0` 即 `K=k`。
- 复算（`selection.py` 已于 2026-10-11 落盘跑通，数字即为冻结依据）：`12k_FE` k=10、K=10；`48k_DE` k=12、K=12（会话内原记 13，差在短记录阈值取整少排了一个 118 窗文件，按原文 `< 0.5 × 中位` 复算得 12）。两来源 `类×尺寸` 覆盖 9/9，无单例簇。`test = 全部目标文件 − 池` = 35/40 文件、70/80 记录，无空层。详见 `docs/v6运行冻结记录.md`。
- 本轮唯一一批：`logspec512` × {`ce`, `ce_mmd`} × 2 来源 × 4 折 × 3 seed = 48，加控制臂 `source_continue` × 4 折 × 3 seed = 12，共 **60 个完整训练流程**一次跑完；`adabn` 走无梯度分支（24 个前向运行），`joint` 复用 v5 源权重只评估。`envelope4096`、`logspec4096` 不跑；不加候选集 bootstrap；不为稳健性追加 seed 或折。
- 配对链六对，训练前写死不增删：`source_continue↔joint`、`adabn↔joint`、`ce↔joint`、`ce↔source_continue`、`ce↔adabn`、`ce_mmd↔ce`。`source_continue` 的 `lr`/`epochs`/`steps_per_epoch`/`updates` 必须与 `ce` 逐字段相同（1e-4／30／按折取该折源 config：f0、f2、f3 为 155／4650，f1 为 156／4680），它才成立为"额外更新步数相同"的控制。
- 目标划分与 `fold`/`seed` 无关，选标只用无标签信号，因此 `tsplit` 塌缩成 1 档；报告按 `fold`/`seed` 给均值、样本标准差（`ddof=1`）与逐格值，负增量完整保留。
- 训练铁律沿用 v5：源 `split_seed=0`、适配从同 `fold/seed` 的 `joint` `last.pt` 起、新建 Adam `1e-4`、`bs=64`、`adapt_epochs=30`、`steps_per_epoch=ceil(S_ref/64)` 且 `S_ref` 恒按 512 点口径从源 config 取、正式取末轮 `last.pt`（`selection_rule=last`，源验证只记录不择优）、无目标验证集、`CE` 为源重放＋目标、MMD 用 `lambda=0.3`／固定核 `sigma=(1,5,10)`／`biased`／只对共有三类逐类等权平均／不足 2 样本跳类／不回退全局、记录级评估先按同文件同通道做窗口概率平均。
- 目标文件不建缓存系统，直接算；不用 BA；三个特殊文件的通道前缀按 v5 写死（X099／X175／X217）。

## 落盘与产物（保持普通结构）

- `selection.py` 一次性生成 `outputs/v6_round1/selection/selection.json`：池清单、簇标签、每簇 medoid 到簇心距离、silhouette 曲线、孤立度（最近邻距离）、窗口数过滤名单、`k`/`p`/`random_state`/`tol`；元数据（`类/尺寸/位置`、覆盖计数）只作为 `diagnostics` 字段附后，代码路径上不得回流到选标。
- 训练目录普通文件即可：`config.json`、`last.pt`、`best.pt`、`history.csv`、`split.json`；v6 新目录补写 `checkpoint_kind` 与 `checkpoint_epoch`（补字段即可，不造校验体系）。读 v5 权重时按 `last.pt` 文件名 + `selection_rule=last` 认定，这条依赖记进 `frozen_plan_v6.json`。
- 评估产物：`metrics.json`（按"来源/通道"分组）、`predictions.csv`（一行一个文件×通道）、`eval_info.json`。
- 不写调度器与断点续训框架：用现成的顺序 shell 脚本跑，失败就单独重跑那一个目录并留 `_retry1` 后缀，不引入进度状态机。
- 输出根 `outputs/v6_round1/`：`selection/`、`train/`、`eval/`、`summary_<date>/`、`frozen_plan_v6.json`。v5 目录只读，不写入、不覆盖。

## 结果出来后

- 达标判据是记录级准确率按来源均值是否 ≥95%，事先写死不得改。
- 若未到 95%，按 `docs/v6协议草案.md` §8 预先约定的退路走：先用 `predictions.csv` 做**零算力错误结构分析**（错分集中在哪一类／哪一层、是否整档塌成一类、源模型系统性错分残留多少），再决定下一批动源域构造、分类头还是对齐。不在看到分数后临时加臂、临时改 `k` 或容差。

## 语言

文档、注释、提示用简体中文，标识符保持英文。
