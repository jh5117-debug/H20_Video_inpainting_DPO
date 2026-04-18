# DPO 训练指标说明表

> 对应代码：
> [train_dpo_stage1.py](/home/hj/Reg_DPO_Inpainting/DPO_finetune/train_dpo_stage1.py)
> [train_dpo_stage2.py](/home/hj/Reg_DPO_Inpainting/DPO_finetune/train_dpo_stage2.py)

## 1. 最核心的公式

| 项目 | 公式/定义 | 直觉解释 |
|---|---|---|
| `model_diff` | `mse_w - mse_l` | policy 认为 winner 和 loser 差多少 |
| `ref_diff` | `ref_mse_w - ref_mse_l` | ref 认为 winner 和 loser 差多少 |
| `inside_term` | `-0.5 * beta_dpo * (model_diff - ref_diff)` | policy 相对 ref 在这个偏好对上“赢了多少” |
| `dpo_loss` | `-log(sigmoid(inside_term))` | DPO 主损失，inside_term 越大通常越小 |

## 2. 一个统一例子

| 场景 | `ref_mse_w` | `ref_mse_l` | `mse_w` | `mse_l` | 结论 |
|---|---:|---:|---:|---:|---|
| 健康例子 | `0.10` | `0.16` | `0.095` | `0.170` | winner 更好，loser 更差，属于比较理想的 DPO 改进 |
| 假性成功例子 | `0.10` | `0.16` | `0.110` | `0.220` | GT 反而更差，只是 loser 更差得更多，排序表面赢了 |

### 用健康例子计算

| 项目 | 数值 |
|---|---:|
| `model_diff` | `0.095 - 0.170 = -0.075` |
| `ref_diff` | `0.100 - 0.160 = -0.060` |
| `model_diff - ref_diff` | `-0.015` |
| `inside_term (beta=500)` | `-0.5 * 500 * (-0.015) = 3.75` |
| `sigmoid(inside_term)` | `0.977` |
| `dpo_loss` | `0.023` |

### 用假性成功例子计算

| 项目 | 数值 |
|---|---:|
| `model_diff` | `0.110 - 0.220 = -0.110` |
| `ref_diff` | `0.100 - 0.160 = -0.060` |
| `model_diff - ref_diff` | `-0.050` |
| `inside_term (beta=500)` | `12.5` |
| `sigmoid(inside_term)` | `0.999996` |
| `dpo_loss` | `接近 0` |

## 3. 训练时每个指标怎么读

| 指标 | 代码定义 | 简单理解 | 正常训练时大概应该是什么样 | 危险信号 | 小例子 |
|---|---|---|---|---|---|
| `dpo_loss` | `-log(sigmoid(inside_term))` | 当前偏好损失 | 应逐步下降，但不该几百步内直接掉到 `0` | 很快接近 `0`，同时 `implicit_acc≈1`、`sigma_term≈1` | `inside_term=0 -> loss≈0.693`；`inside_term=5 -> loss≈0.0067` |
| `implicit_acc` | `P(inside_term > 0)` | policy 相对 ref，把 GT 判成更好的比例 | 前期在 `0.55~0.85` 更健康 | 很快到 `1.0`，往往是假性成功 | `10/16` 个样本 `inside_term>0`，则 `implicit_acc=0.625` |
| `win_gap` | `mse_w - ref_mse_w` | policy 在 GT 上比 ref 更好还是更差 | 当前 `winner=GT`，理想上应尽量回到 `<=0` | 长期 `>0`，说明 GT 侧其实更差 | `0.095-0.100=-0.005` 是好现象 |
| `lose_gap` | `mse_l - ref_mse_l` | policy 在负样本上比 ref 更差多少 | 可以为正，但不能只靠它变大来“赢” | `lose_gap` 很大但 `win_gap` 也为正 | `0.180-0.160=0.020` |
| `reward_margin` | `ref_mse_w - ref_mse_l` | ref 自己能不能分出 winner 和 loser | 一般应明显为负，说明偏好对清晰 | 过于接近 `0`，说明这对样本本身不好分 | `0.10-0.16=-0.06` |
| `sigma_term` | `sigmoid(inside_term)` | sigmoid 有没有太快饱和 | 大致在 `0.5~0.9` 更健康 | 太快贴近 `1.0`，梯度会迅速变小 | `inside_term=2.5 -> sigma≈0.924` |
| `inside_term_mean/min/max` | inside_term 的统计量 | 整批样本的偏好打分分布 | 应适度为正，但不要整体过大 | `mean/min/max` 都很大，说明整批一起饱和 | `mean=0.4,min=-0.2,max=1.1` 比较健康 |
| `mse_w` | policy winner MSE | policy 在 GT 上的绝对误差 | 应下降或至少不坏于 `ref_mse_w` | 持续升高 | `0.102 -> 0.099 -> 0.095` |
| `mse_l` | policy loser MSE | policy 在负样本上的误差 | 可上升，但不能把“loser 变更差”误当进步 | `mse_l` 上升同时 `mse_w` 也变差 | `0.16 -> 0.19` 不一定是好事 |
| `ref_mse_w` | ref winner MSE | ref 在 GT 上的 baseline | 用来做对照 | 不是优化目标，但不能忽略 | 如果 `mse_w > ref_mse_w`，说明 GT 侧没提升 |
| `ref_mse_l` | ref loser MSE | ref 在 loser 上的 baseline | 用来做对照 | 同上 | 如果 `mse_l` 只是远大于它，不代表模型真的更好 |
| `kl_divergence` | `0.5*(all_model_losses-all_ref_losses)` | policy 离 ref 漂了多远 | 小幅上升可接受 | 暴涨通常说明训练发散 | `0.002 -> 0.006` 还行，`0.05+` 要警惕 |
| `dgr_grad_norm` | DPO loss 梯度范数 | DPO 信号是否还在推动更新 | 不能长期接近 `0` | 很早就接近 `0` | `0.12 -> 0.08 -> 0.05` 还活着 |
| `grad_norm_ratio` | 当前 DPO 梯度 / 初始 DPO 梯度 | 当前总梯度里 DPO 还剩多少 | 应明显大于 `0.01` | 长期低于 `0.01` | `0.60 -> 0.35 -> 0.18` 健康 |
| `lr` | 当前学习率 | 用来排查 scheduler 是否异常 | 按预期 schedule 走 | 异常跳变或过大/过小 | 这次失败主因不是 lr，而是 beta 太大 |

## 4. 额外很重要的诊断：`loser_dominant_ratio`

| 项目 | 定义 | 简单理解 | 怎么读 | 例子 |
|---|---|---|---|---|
| `loser_dominant_ratio` | 在所有 `inside_term>0` 的样本里，`loser_degradation > winner_improvement` 的比例 | 有多少“正确样本”其实主要是靠 loser 变差赢的 | 越高越要小心“假性成功” | `implicit_acc=1.0` 但 `loser_dominant_ratio=0.85`，说明大多数胜利都是靠 loser 变差 |

## 5. `inside_term` 到底是什么

| 问题 | 解释 |
|---|---|
| 它是什么 | DPO 里最核心的“偏好打分” |
| 它大于 0 表示什么 | policy 比 ref 更满足这一个偏好对 |
| 它小于 0 表示什么 | policy 还不如 ref |
| 为什么它重要 | `implicit_acc`、`sigma_term`、`dpo_loss` 都直接由它决定 |
| 为什么要看它的均值/最值 | 用来看是不是整批样本一起过早饱和 |

## 6. `beta_dpo` 大和小分别意味着什么

| 维度 | `beta` 大，比如 `1000 / 2500` | `beta` 小，比如 `500` |
|---|---|---|
| 直觉 | 把“赢了多少”的分数放大 | 把“赢了多少”的分数缩小 |
| 优点 1 | 偏好信号更强 | 更稳定 |
| 优点 2 | 排序学得更快 | 不容易几百步就饱和 |
| 优点 3 | 难分样本可能更容易拉开差距 | DPO 梯度能活更久 |
| 优点 4 |  | 更容易看清模型是在改善 GT 还是只是在恶化 loser |
| 缺点 1 | `inside_term` 很快变大 | 学得更慢 |
| 缺点 2 | `sigma_term` 很快接近 `1` | 偏好信号更弱 |
| 缺点 3 | `dpo_loss` 很快接近 `0` | 如果太小，排序提升可能不明显 |
| 缺点 4 | 梯度很快变小，容易早饱和 | 训练可能更像“贴着 ref 慢慢动” |
| 缺点 5 | 容易出现“看起来排序赢了，其实只是 loser 更差” |  |

## 7. 同样的样本，`beta` 大小会差多少

假设：

```text
model_diff - ref_diff = -0.002
```

| `beta_dpo` | `inside_term` | `sigma_term` | `dpo_loss` | 解释 |
|---:|---:|---:|---:|---|
| `500` | `0.5` | `0.62` | `0.47` | 还有明显训练信号 |
| `2500` | `2.5` | `0.924` | `0.079` | 看起来已经快赢完了，更容易过早饱和 |

## 8. 为什么第二次训练主要先减小 `beta`

| 第一次训练观察到的现象 | 说明什么 |
|---|---|
| `implicit_acc` 很快到 `1.0` | 排序很快全赢了 |
| `sigma_term` 很快到 `1.0` | sigmoid 快速进入饱和区 |
| `dpo_loss` 很快接近 `0` | DPO 梯度很快会变小 |
| `win_gap` 长期还是正的 | GT 侧没有真的变好 |

| 因此第二次训练先做什么 | 目的 |
|---|---|
| `beta_dpo: 2500 -> 500` | 让 `inside_term` 不要涨太猛 |
| 降低 `beta` | 让 `sigma_term` 不要几百步就贴到 `1` |
| 降低 `beta` | 让 DPO loss 保持更长时间的有效梯度 |
| 降低 `beta` | 更容易区分“真实改进”和“假性成功” |

## 9. 最后怎么用一张表快速判断训练是不是健康

| 如果你看到 | 更像是健康训练 | 更像是出问题 |
|---|---|---|
| `dpo_loss` | 缓慢下降 | 几百步内直接接近 `0` |
| `implicit_acc` | 稳步上升，先到 `0.6~0.8` | 很快到 `1.0` |
| `win_gap` | 往 `<=0` 回 | 长期 `>0` |
| `sigma_term` | 不要太快贴近 `1` | 很快接近 `1.0` |
| `dgr_grad_norm` | 仍明显大于 `0` | 很快接近 `0` |
| `loser_dominant_ratio` | 不要长期太高 | 长期很高，说明主要靠 loser 变差 |

一句话结论：

| 真正想要的状态 |
|---|
| 不只是排序赢了，而是 GT 侧也真的更好了。 |
