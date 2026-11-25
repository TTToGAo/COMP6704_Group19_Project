# Group 19 共享接口约定（A → B/C/D）

适用范围：四个自写求解器（Platt SMO、二阶 SMO、CSMO、PGD）的共同输入、输出和解质量复核；作者版 `libsvm_cd -a 0/-a 1` 的结果也换算到同一指标后比较。本文件定义**报告口径**，各算法内部可以使用自己的更新和停止逻辑。

## 1. 唯一的优化问题与符号

使用二分类、软间隔 **C-SVC 对偶**。标签 `y_i ∈ {-1,+1}`；样本顺序一经保存，不得在求解器内部改变。令

```text
K_ij = K(x_i, x_j)
Q_ij = y_i y_j K_ij
D(α) = 1ᵀα − ½ αᵀQα                    # 报告中统一为最大化目标
约束：0 ≤ α_i ≤ C；yᵀα = 0
α₀ = 0                                  # 所有正式对照从零开始
```

`C > 0` 对所有样本相同。主实验使用 RBF 核 `K(x,z)=exp(-γ‖x−z‖²)`，其中 `γ` 是 **RBF 核系数**，不是 SMO 更新步长。初始配置沿用项目方案：`C=1`、`γ=1/22`（命令行 `-g 0.045454545454545456`）、`ε=0.001`、缓存 100 MB、关闭 shrinking（作者代码 `-m 100 -h 0`）。这些是起始配置；实验负责人锁定最终配置后，四个自写实现和作者版均使用同一数值。特征缩放与样本顺序由共享数据文件决定，不在求解器内部各自处理。

作者代码通常最小化 `F(α)=½αᵀQα−1ᵀα`；其日志目标值转换为 `D(α)=−F(α)` 后再进入报告。不要把原问题的目标值或分类准确率放进对偶目标列。

## 2. A 提供的共用解质量函数

输入为 `alpha: float64[n]`、`y: int8[n]`、`Q: float64[n,n]`、`C: float`。如内存不足，可用同一核计算器实现 `Q @ alpha`；数值含义必须一致。函数返回以下字段：

| 字段 | 定义 | 说明 |
|---|---|---|
| `dual_objective` | `sum(alpha) - 0.5 * alpha @ (Q @ alpha)` | 越大越好；只在可行且精度相近时比较。 |
| `equality_residual` | `abs(y @ alpha)` | 等式约束残差。 |
| `box_violation` | `max_i max(0, -alpha_i, alpha_i-C)` | 最大盒约束违反量。 |
| `kkt_violation` | 下文定义 | 与 LIBSVM 的 `Gmax + Gmax2` 对齐，取其与零的较大者。 |
| `gmax`, `gmax2` | 下文定义 | 保留原值，便于同作者日志核对。 |

令最小化目标的梯度 `G=Q @ alpha − 1`。以浮点边界容差 `τ_box=1e-10 * max(1,C)` 判定边界：`α_i < C−τ_box` 视为可上调，`α_i > τ_box` 视为可下调。定义

```text
I_up  = {i: (y_i=+1 且 α_i<C) 或 (y_i=−1 且 α_i>0)}
I_low = {i: (y_i=+1 且 α_i>0) 或 (y_i=−1 且 α_i<C)}
Gmax  = max_{i∈I_up}  (−y_i G_i)
Gmax2 = max_{j∈I_low} (+y_j G_j)
kkt_violation = max(0, Gmax + Gmax2)
```

实际实现中，上述集合的边界判断使用 `τ_box`。若某集合为空，返回 `NaN` 并标记解无效，不要把空集合的值写成零。**停止判据**为 `kkt_violation ≤ ε`，同时另行检查可行性；达到迭代上限须记录 `max_iter`，不能写成 `converged`。`τ_box` 仅用于诊断边界，不能用来替代求解器的步长截断或修改返回的 `alpha`。比较作者日志时，保留作者最终全工作集检查的原始 `Gmax + Gmax2` 和共用函数复算值；两者若有差异，先核对导出的 `alpha`、样本顺序与核参数。

建议把实现放在共享模块 `common/metrics.py`，对外函数名为 `evaluate_dual(alpha, y, Q, C) -> dict`。A 负责实现并提供至少一个小数组示例，供 B、C、D 对照调用。

## 3. 求解器接口（与 D 共用）

建议 Python 接口如下；其他语言或作者版程序用适配脚本输出相同字段：

```python
solve(X, y, *, C, gamma, epsilon, alpha0=None,
      max_iter=None, log_every=None) -> SolverResult

SolverResult:
    alpha              # float64[n]，顺序与输入样本一致
    iterations         # 实际完成的更新次数；PGD 为梯度更新次数
    stop_reason        # converged | max_iter | numerical_failure
    history            # 按第 4 节的字段记录
```

`alpha0=None` 表示零向量。不得在 `solve` 内随机重排样本、单独缩放特征或静默改变 `C`/`gamma`。`epsilon` 指第 2 节报告用的 KKT 目标容差；内部判断若不同，应在 README 写明。A 的原始 SMO、B 的二阶 SMO 和 C 的 CSMO 共用相同的 `X/y/K/Q`，B、C 另行约定工作集选择函数的调用接口。D 负责共享数据加载与最终绘图接口，A 负责指标函数。

## 4. 统一日志与输出文件

每次运行保存一份配置、一份终点摘要及所需的收敛记录。推荐文件格式：

```text
config.json: dataset_id, dataset_hash, sample_indices_file, n_train,
             kernel, C, gamma, epsilon, alpha0, max_iter, log_every,
             algorithm, implementation_commit, machine, threads

summary.json: algorithm, iterations, stop_reason, train_seconds,
              dual_objective, equality_residual, box_violation,
              kkt_violation, gmax, gmax2, test_accuracy（如计算）

history.csv: iteration, dual_objective, kkt_violation,
             equality_residual, box_violation

alpha.npy: 与输入训练样本完全同序的最终对偶变量
```

`history.csv` 必须包含 `iteration=0` 和终点；B、C 在发动机子集上的记录间隔一致。`iterations` 是各实现定义的更新次数，PGD 与 SMO 的迭代成本不同，报告中不能把四者的迭代次数直接当作等价工作量。正式计时以 `svm_train()` 调用前后单调时钟为准；数据读取、预处理、写盘、预测和额外指标复核在计时区间外。作者版全量实验由 B 接入同一计时与终点日志，D 汇总。

## 5. 数据与交接

| 场景 | 固定数据 | 负责交接 |
|---|---|---|
| 小规模正确性 | `heart_scale`，270 条；四个自写求解器同一文件、顺序、参数和初始点 | D 发布共享输入；A 发布 `evaluate_dual`，四人交回 `alpha` 与 `summary`。 |
| 发动机子集收敛 | `ijcnn1` 官方训练集按标签分层抽取 1,500 条，种子 19；保存行号和顺序 | B 发布抽样索引与数据校验值；B、C 交自写收敛日志和作者版终点；A 用共用函数复核导出的 `alpha`。 |
| 全量发动机计时 | `ijcnn1` 官方训练集 49,990 条、测试集 91,701 条 | B 在同一作者版构建中运行 `-a 0` 与 `-a 1`，五轮配对、相邻轮次交换先后；D 汇总。 |

若导入 LIBSVM 格式数据，**文件已有的特征索引与数值就是输入**；要额外缩放，须由 D 生成唯一的共享缩放文件和规则，所有算法均使用它。每次数据交接记录文件哈希，避免仅凭文件名认定输入一致。

## 6. A 的首批交付顺序

1. 把本文件发给 B、C、D，确认字段名称及他们的代码能读取同一数据顺序。
2. 实现 `common/metrics.py` 与小数组示例；先让 B、C、D 接入指标。
3. 再完成原始 Platt SMO 可运行入口、复现命令和 `heart_scale` 终点摘要。
4. 用共用函数检查 B、C、D 交回的 `alpha`；将数值差异及原因写入实验记录。

参考：[Lecture 6](../ppt/Lecture20261006.pdf)、[Platt 原始 SMO](https://www.microsoft.com/en-us/research/wp-content/uploads/1998/04/sequential-minimal-optimization.pdf)、[Fan 等人的二阶工作集选择](https://www.jmlr.org/papers/v6/fan05a.html)、[CSMO 论文](https://arxiv.org/abs/2003.08719)、[LIBSVM 源码](https://github.com/cjlin1/libsvm/blob/master/svm.cpp)。
