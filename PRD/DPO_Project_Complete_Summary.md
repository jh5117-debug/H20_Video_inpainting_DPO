# DiffuEraser DPO Finetune 项目完整总结

> **项目目标**：在已完成的 SFT 全量微调基础上，引入 VideoDPO 的 Direct Preference Optimization (DPO) 机制，进一步优化 DiffuEraser 视频修复质量。
> **日期**：2026-03-23 ~ 2026-03-31
> **说明**：本文件已经合并原 `DPO_Finetune_PRD.md` 的有效内容，后续请以本文件作为唯一主文档。

---

## 一、项目构想与设计

### 1.1 背景

DiffuEraser 是一个基于 Stable Diffusion 1.5 + BrushNet 的视频修复模型，采用两阶段训练策略：
- **Stage 1**：训练 UNet2D + BrushNet（空间质量）
- **Stage 2**：训练 MotionModule（时序一致性）

我们已在 YouTube-VOS + DAVIS 数据集上完成了 SFT 全量微调（Stage 1: 30000 步 + Stage 2: 34000 步），权重保存在：
```
/sc-projects/sc-proj-cc09-repair/hongyou/dev/Reg_DPO_Inpainting/finetune-stage2/converted_weights_step34000
```

### 1.2 DPO 核心思路（源自 VideoDPO）

参考 `/home/hj/VideoDPO` 的开源实现，将 DPO 引入视频修复领域：

1. **偏好对构建**：GT（正样本） vs 退化修复结果（负样本/home/hj/All_Repo/VideoInpainting_PDF/PRD/video_inpainting_papers_summary.md），无需人工标注
2. **损失函数**：Diffusion-DPO Loss + Reg-DPO 诊断指标
3. **双模型架构**：Policy（可训练） + Reference（冻结），共享 SFT 权重初始化
4. **两阶段延续**：DPO Stage 1 → 2 对齐 SFT 的训练范式

> **说明**：当前实际落地的是 **vanilla Diffusion-DPO loss + Reg-DPO 风格诊断指标**，尚未引入 Reg-DPO 论文中的 SFT regularization 项。

### 1.3 关键设计决策

| 决策 | 选择 | 理由 |
|------|------|------|
| BrushNet 条件 | pos/neg 共享 GT masked image | 防止信息泄漏 |
| DAVIS 过采样 | 10x | 平衡 DAVIS (~30) vs YouTube-VOS (~3400) |
| nframes | 16 | 对齐 DPO 数据集 chunk 大小 |
| beta_dpo | 初版 2500，现默认 500 | 首次实跑出现早饱和后下调 |
| 权重保存 | 仅 best + last | 节省 WandB 存储 |

### 1.4 DPO 数据集

```
/sc-projects/.../data/DPO_Finetune_data/   (HF: JiaHuang01/DPO_Finetune_Data, 69.9GB)
├── manifest.json              (2066 entries)
├── davis_bear/                (~30 DAVIS 视频)
│   ├── gt_frames/             原始 GT 帧
│   ├── masks/                 二值 mask
│   ├── neg_frames_1/          退化负样本 1 (chimera_chunked)
│   ├── neg_frames_2/          退化负样本 2
│   └── meta.json              chunk 边界
└── ytbv_*/                    (~2000 YouTube-VOS 视频, 同结构)
```

### 1.5 训练参数

| 参数 | Stage 1 | Stage 2 |
|------|---------|---------|
| 可训练模块 | UNet2D + BrushNet | MotionModule only |
| LR | 1e-6 | 1e-6 |
| beta_dpo | 500 | 500 |
| nframes | 16 | 16 |
| max_steps | 20000 | 30000 |
| batch_size | 1 (per GPU) | 1 |
| GPU 数 | 8 (DDP) | 8 |
| val_steps | 2000 | 2000 |
| val 指标 | PSNR + SSIM | PSNR + SSIM + Ewarp + TC |

### 1.6 监控指标（按 scope 分组）

每步记录到 WandB：
- `rank0/dpo_loss`, `rank0/mse_w`, `rank0/mse_l`
- `rank0/win_gap`, `rank0/lose_gap`, `rank0/reward_margin`
- `rank0/sigma_term`, `rank0/kl_divergence`
- `rank0/dgr_grad_norm`, `rank0/grad_norm_ratio`
- `global/implicit_acc`, `global/inside_term_mean`, `global/inside_term_min`, `global/inside_term_max`, `global/loser_dominant_ratio`
- gather 失败时自动 fallback 到 `rank0/implicit_acc`, `rank0/inside_term_*`, `rank0/loser_dominant_ratio`
- 终端诊断表额外标注 `[R0]` / `[G]`，避免把本卡指标和全局指标混淆

### 1.7 训练时到底会看到哪些表，以及“正常训练”大概长什么样

训练时我们主要盯的是两类表：

1. **偏好排序类**
   - `dpo_loss`
   - `implicit_acc`
   - `inside_term_mean/min/max`
   - `sigma_term`
   - `win_gap / lose_gap`
   - `reward_margin`

2. **质量与稳定性类**
   - `mse_w / mse_l`
   - `ref_mse_w / ref_mse_l`
   - `kl_divergence`
   - `dgr_grad_norm`
   - `grad_norm_ratio`
   - `lr`

如果用更直白的话去理解：

| 指标 | 简单理解 | 正常训练时大概应该是什么样 |
|------|----------|----------------------------|
| `dpo_loss` | 当前偏好损失 | 应该逐步下降，但不该在几百步内直接掉到 0 |
| `implicit_acc` | policy 相对 ref，把 GT 判成更好的比例 | 前期在 `0.55~0.85` 更健康；太快到 `1.0` 往往是假性成功 |
| `win_gap` | policy 在 GT 上比 ref 更好还是更差 | 当前 winner=GT，理想上应尽量回到 `<= 0` |
| `lose_gap` | policy 在负样本上比 ref 更差多少 | 可以为正，但不能只靠它变大来“赢” |
| `sigma_term` | sigmoid 有没有太快饱和 | 不要太快贴近 `1.0`，否则梯度会迅速变小 |
| `inside_term_mean/min/max` | 整批样本的偏好打分分布 | 用来看是不是整批样本一起过早饱和 |
| `mse_w` | policy 在 GT 上的重建误差 | 应下降或至少不坏于 ref |
| `mse_l` | policy 在负样本上的误差 | 可上升，但不能把“loser 变更差”误当作真正进步 |
| `kl_divergence` | policy 离 ref 漂了多远 | 小幅上升可接受，暴涨通常说明训练发散 |
| `dgr_grad_norm` | DPO 信号是否还在推动更新 | 不能长期接近 0 |
| `grad_norm_ratio` | 当前总梯度里有多少来自 DPO | 太小表示 DPO 基本不工作了 |

一句话概括“正常训练的样子”：
- `dpo_loss` 缓慢下降；
- `implicit_acc` 上升但不要几百步就到 `1.0`；
- `win_gap` 尽量往 `<= 0` 回；
- `sigma_term` 不要太快贴到 `1.0`；
- `dgr_grad_norm` 不能太早掉到接近 0。

---

## 二、代码架构

所有 DPO 代码严格隔离在 `DPO_finetune/` 目录，不修改任何 SFT 代码：

```
DPO_finetune/
├── dataset/
│   └── dpo_dataset.py              DPO 偏好对数据集
├── train_dpo_stage1.py             Stage 1 训练（UNet2D + BrushNet）
├── train_dpo_stage2.py             Stage 2 训练（MotionModule）
└── scripts/
    ├── run_dpo_stage1.py           Stage 1 Python 启动入口
    ├── run_dpo_stage2.py           Stage 2 Python 启动入口
    ├── 03_dpo_stage1.sbatch        Stage 1 SLURM 脚本
    └── 03_dpo_stage2.sbatch        Stage 2 SLURM 脚本
```

### 路径规范

- **集群**：`${PROJECT_HOME}/dev/Reg_DPO_Inpainting/`
- **本地**：`/home/hj/Reg_DPO_Inpainting/`
- **所有路径通过命令行参数传入，默认值指向集群路径**
- 通过 GitHub push/pull 同步

---

## 三、从初始代码到部署：逐次 Debug 记录

### Bug #1: UNetMotionModel 未 import

**时间**：首次提交
**现象**：`NameError: name 'UNetMotionModel' is not defined`
**原因**：`train_dpo_stage1.py` 在 `_extract_2d_from_motion` 中使用 `UNetMotionModel`，但未在顶部 import。
**修复**：
```diff
+from libs.unet_motion_model import UNetMotionModel
```

---

### Bug #2: DDP 多进程函数属性不安全

**时间**：首次提交
**现象**：`initial_grad_norm` 在多 GPU 下可能不一致
**原因**：使用 `main._initial_grad_norm`（函数属性）在 DDP 多进程间不可靠。
**修复**：
```diff
-main._initial_grad_norm = grad_norm
+initial_grad_norm = grad_norm  # 普通局部变量
```

---

### Bug #3: run_dpo 脚本冗余 mixed_precision 参数

**时间**：首次提交
**现象**：`accelerate launch` 参数冲突
**原因**：`run_dpo_stage1.py` 同时在 accelerate launch 和训练脚本参数中传了 `--mixed_precision`。
**修复**：删除训练脚本参数中的冗余 `--mixed_precision`。

---

### Bug #4: WandB 初始化过晚

**时间**：第一次集群运行
**现象**：WandB 上看不到任何报错信息，只有 SLURM stdout 有 traceback。
**原因**：WandB `init_trackers` 在模型加载之后调用，模型加载阶段崩溃时 WandB 还没启动。
**修复**：将 WandB 初始化移到 `main()` 函数最前面（权重加载之前）。

---

### Bug #5: 权重加载 ValueError (num_attention_heads)

**时间**：首次集群运行
**现象**：`ValueError: At the moment it is not possible to define the number of attention heads via num_attention_heads`
**原因**：DPO 代码用 `UNet2DConditionModel.from_pretrained()` 加载 SFT 权重，但权重目录的 `config.json` 声明为 `UNetMotionModel`，config 格式不兼容。
**修复**：自动检测 config `_class_name`，若为 `UNetMotionModel` 则先用 `UNetMotionModel.from_pretrained()` 加载，再提取 2D 权重。

---

### Bug #6: Stage 2 权重拷贝缺少 hasattr 保护

**时间**：外部 Code Review + 审计确认
**现象**：Stage 2 启动时 `AttributeError`（未触发因为还没跑到 Stage 2）
**原因**：
- `down_block.attentions` 在 `DownBlock2D` 不存在，仅检查了目标侧 `hasattr`
- `stage1_unet.conv_act` 未用 `hasattr` 保护

**修复**：
```diff
-if hasattr(unet_main.down_blocks[i], "attentions"):
+if hasattr(unet_main.down_blocks[i], "attentions") and hasattr(down_block, "attentions"):
```
```diff
-if stage1_unet.conv_act is not None:
+if hasattr(stage1_unet, 'conv_act') and stage1_unet.conv_act is not None:
```

---

### Bug #7: Stage 2 encoder_hidden_states 未翻倍

**时间**：外部 Code Review + 审计确认
**现象**：Shape mismatch（未触发因为还没跑到 Stage 2）
**原因**：DPO concat 后 `noisy_all=(2*bsz*nframes,...)`，但 `encoder_hidden_states=(bsz,seq,dim)`。`UNetMotionModel` 内部 `repeat_interleave(num_frames)` 只展到 `(bsz*nframes,...)`，不匹配。
**修复**：
```diff
+encoder_hidden_states_motion = encoder_hidden_states.repeat(2, 1, 1)
```

---

### Bug #8: manifest key ≠ 目录名

**时间**：第二次集群运行 (commit `16eb51c`)
**现象**：`DPODataset: 0 entries` → `ValueError: num_samples=0`
**原因**：HF 上传的 `manifest.json` 的 key 为 `davis_bear_part1`，但实际目录名为 `davis_bear`（不含 `_part1`）。代码用 key 拼路径 → 目录不存在 → 跳过所有 entry。
**修复**：`dpo_dataset.py` 的 `_load_manifest` 新增 fallback，当 key 对应目录不存在时，从 manifest 的 `gt_frames` 路径字段提取实际目录名。

---

### Bug #9: timesteps 维度不匹配 (BrushNet expand 崩溃)

**时间**：第三次集群运行 (commit `18f76c4`)
**现象**：`RuntimeError: The expanded size of the tensor (32) must match the existing size (2)`
**原因**：
```
timesteps = (bsz=1,)
timesteps_all = timesteps.repeat(2) = (2,)
noisy_all = (2*1*16, ...) = (32, ...)
BrushNet.forward: timesteps.expand(sample.shape[0]=32) → expand(32) from (2,) → 💥
```

**修复**（Stage 1）：
```diff
+timesteps_expanded = timesteps.repeat_interleave(args.nframes, dim=0)
-noisy_pos = noise_scheduler.add_noise(pos_latents, noise, timesteps)
+noisy_pos = noise_scheduler.add_noise(pos_latents, noise, timesteps_expanded)
-timesteps_all = timesteps.repeat(2)
+timesteps_all = timesteps_expanded.repeat(2)  # (2*bsz*nframes,)
```

---

### Bug #10: Stage 2 BrushNet vs UNetMotionModel 需要不同维度的 timesteps

**时间**：深度审计发现 (commit `28853f9`)
**现象**：（预防性修复，Stage 2 尚未运行）
**原因**：
```
BrushNet.forward:       timesteps.expand(sample.shape[0])          → 需要 (2*bsz*nframes,)
UNetMotionModel.forward: timesteps.expand(sample.shape[0]//nframes) → 需要 (2*bsz,)
```

两个模型需要不同维度的 timesteps，不能共用同一个变量。

**修复**：
```diff
-timesteps_all = timesteps_expanded.repeat(2)
+timesteps_all_2d = timesteps_expanded.repeat(2)      # (32,) → BrushNet
+timesteps_all_motion = timesteps.repeat(2)            # (2,)  → UNetMotionModel
```

---

### 全局改进: WandB 异常捕获

**问题**：Python traceback 只出现在 SLURM stdout，WandB 上看不到任何报错。
**修复**：在 `__main__` 入口添加全局 `try-except`：
```python
try:
    main(args)
except Exception as e:
    tb = traceback.format_exc()
    logger.error(f"Training crashed!\n{tb}")
    if wandb.run is not None:
        wandb.alert(title="DPO Crashed", text=tb, level=wandb.AlertLevel.ERROR)
        wandb.finish(exit_code=1)
    raise
```

---

## 四、Git 提交历史

| Commit | 描述 |
|--------|------|
| 初始 | DPO 全套代码首次提交 |
| `cb4cecb` | 修复 UNetMotionModel import + DDP 变量 + 冗余 mixed_precision |
| `7c7b7a7` | Stage 2 hasattr 保护 + encoder_hidden_states 翻倍 |
| `16eb51c` | manifest key ≠ 目录名 fallback |
| `18f76c4` | timesteps `repeat_interleave(nframes)` + 全局异常 WandB |
| `28853f9` | Stage 2 拆分 `timesteps_all_2d` / `timesteps_all_motion` |
| 2026-03-30/31 本地修订 | `beta_dpo` 默认 500、`implicit_acc` 跨卡 gather、`inside_term` 统计、`loser_dominant_ratio`、WandB scope 化 |

---

## 五、首次 Stage 1 集群实跑复盘（2026-03-30）

### 5.1 首次实跑现象

首次 Stage 1 集群实跑使用 `beta_dpo=2500`。日志呈现出非常典型的 vanilla DPO 早饱和特征：

- Step 1: `implicit_acc=0.25`，对应本卡 `4/16` 帧判断正确
- Step 300 起：`implicit_acc=1.0`、`sigma_term=1.0`、`dpo_loss≈0`
- `win_gap` 长期为正，说明 policy 在 winner（GT）上的误差并未优于 ref
- `lose_gap` 同时更大为正，说明模型主要通过“让 loser 更差”来拉开相对偏好差距

这与 Reg-DPO 论文中对 vanilla DPO 不稳定性的分析高度一致：DPO 只约束正负样本的**相对差值**，不直接约束每个样本自己的输出分布，因此会出现 loss 很快下降、梯度快速衰减、`Win Gap` 与 `Lose Gap` 一起变差的现象。

### 5.2 从首次实跑得到的关键结论

1. **`implicit_acc` 不能直接按“8 卡全局 batch”理解**  
   最初实现中的 `implicit_acc` 是本卡局部指标，分母是 `B * F = 1 * 16`，因此会出现 `0.25, 0.375, 0.4375` 这类 `k/16` 的离散值。

2. **8 卡影响的是每个 global step 看到的数据量，不是 `implicit_acc` 的原始分母**  
   8 GPU 会让训练按 step 看起来更快进入某种状态，但早期离散跳变本身来自本卡 16 帧统计，而非 8 卡汇总。

3. **`winner = GT` 让 `win_gap` 的符号尤为重要**  
   当 `win_gap > 0` 时，含义不是“GT 更好”，而是“policy 在 GT 上比 ref 更差”。如果这时 `implicit_acc` 仍然是 1.0，说明模型赢主要靠的是 loser 退化更严重，而不是 winner 拟合得更好。

4. **`beta_dpo=2500` 对当前任务过激进**  
   对当前 DiffuEraser + GT-pair 设定，`2500` 很快把 `inside_term` 推到 sigmoid 饱和区，导致训练几百步后就几乎不再有有效 DPO 梯度。

### 5.3 `beta_dpo` 到底是什么，为什么第二次训练主要就是先减小它

`beta_dpo` 可以理解成：**把“winner 比 loser 好多少”这个相对差值放大的系数**。

- `beta` 越大，DPO 越激进
- `beta` 越大，`inside_term` 越容易快速变大
- `inside_term` 一旦太大，`sigma_term = sigmoid(inside_term)` 就会很快贴近 `1`
- `sigma_term` 贴近 `1` 后，`dpo_loss` 会很快接近 `0`
- `dpo_loss` 接近 `0` 后，DPO 梯度也会快速变小

所以第一次训练的问题不是“模型已经学好了”，而是：
- `implicit_acc` 很快到 `1.0`
- `sigma_term` 很快到 `1.0`
- `dpo_loss` 很快接近 `0`
- 但 `win_gap` 仍然长期为正

这说明模型主要靠“让 loser 更差”满足排序，而不是让 GT 侧真正变好。

因此第二次训练最重要的修改，不是先去改一堆别的超参数，而是**先把 `beta_dpo` 从 `2500` 降到 `500`**：
- 让 `inside_term` 增长没那么猛
- 让 `sigma_term` 不要几百步就饱和
- 让 DPO loss 在更长时间里保持有效梯度
- 让我们能更清楚地区分“真实进步”和“假性成功”

### 5.4 针对首次实跑的代码与监控修订

基于上述复盘，代码已进行以下修订：

- `beta_dpo` 不再写死，改为 CLI/sbatch 可配置，默认值从 `2500` 下调为 `500`
- `implicit_acc` 改为跨卡 gather 后的全局指标
- 新增 `inside_term_mean/min/max`，直接监控 sigmoid 输入是否进入饱和区
- 新增 `loser_dominant_ratio`，用于区分“靠 loser 退化获胜”与“靠 winner 改善获胜”
- WandB 与终端日志统一增加 scope 标识：`rank0/` vs `global/`，终端表格使用 `[R0]` / `[G]`

### 5.5 Stage 1 重新提交前的推荐监控口径

重新使用 `beta_dpo=500` 提交后，建议重点看：

- `global/implicit_acc`：前期希望处于 `0.6 ~ 0.85`，而不是几百步内迅速到 1
- `rank0/sigma_term` 与 `global/inside_term_mean/max`：避免过快进入饱和
- `rank0/win_gap`：理想上应回到 `<= 0` 附近，因为 winner 是 GT
- `global/loser_dominant_ratio`：若长期偏高，说明模型仍主要靠恶化 loser 获胜
- validation 指标：Stage 1 以 `PSNR + SSIM` 为主，不应低于 SFT/ref baseline

---

## 六、H20 实验复盘（2026-04-21 ~ 2026-04-22）

### 6.1 H20 运行环境与日志路径

本轮实验迁移到 H20 机器运行，代码目录为：

```bash
/home/nvme01/Video_inpainting_DPO
```

为了避免把大量日志写在代码/权重盘，H20 启动脚本已改为默认将训练日志镜像到：

```bash
/home/nvme03/workspace/world_model_phys/Diffueraser_DPO_Log
```

每次运行会创建类似如下目录：

```bash
/home/nvme03/workspace/world_model_phys/Diffueraser_DPO_Log/<RUN_VERSION>_<RUN_NAME>/
├── train_stdout.log
├── experiment/
│   ├── run_manifest.json
│   ├── wandb_run_info.json
│   └── console_logs/rank*.log
└── wandb/
```

模型权重仍保存在原实验目录：

```bash
/home/nvme01/Video_inpainting_DPO/experiments/dpo/stage1/<RUN_VERSION>_<RUN_NAME>/
```

### 6.2 原始 2-GPU vanilla DPO 长训尝试

运行配置概要：

| 项 | 值 |
|---|---|
| GPU | `0,1` 或 `2,3` 的 2 卡实验 |
| `MAX_STEPS` | `20000` |
| `BATCH_SIZE` | `1` |
| `GRAD_ACCUM` | `2` |
| `MIXED_PRECISION` | `bf16` |
| `GRADIENT_CHECKPOINTING` | `1` |
| 目标 | 原始 DPO 数据集 Stage 1 长训 |

本地已保留的一份关键日志：

```bash
/home/hj/Video_inpainting_DPO/Diffueraser_DPO_Log/h20_dpo_stage1_p77ehlz2_20260422_000137/h20_dpo_stage1_original_data_2gpu_gpu01_split_20260421_150838.log
```

该日志显示，训练从第 300 步开始已经进入非常明显的不健康状态：

| Step | `implicit_acc` | `win_gap` | `lose_gap` | `mse_win` | `ref_mse_win` | `DGR` | `loser_dominant_ratio` |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.843750 | -0.000004 | 0.000096 | 0.037193 | 0.037197 | 75.248701 | 1.000000 |
| 300 | 1.000000 | 0.058843 | 0.120061 | 0.064548 | 0.005705 | 0.000071 | 1.000000 |
| 600 | 1.000000 | 0.098237 | 0.123641 | 0.099192 | 0.000954 | 0.408249 | 1.000000 |
| 1200 | 1.000000 | 0.112232 | 0.131490 | 0.112617 | 0.000385 | 7.607672 | 1.000000 |
| 1800 | 1.000000 | 0.120868 | 0.303063 | 0.176040 | 0.055172 | 0.000000 | 1.000000 |
| 2100 | 1.000000 | 0.144727 | 0.319537 | 0.176502 | 0.031776 | 0.000000 | 1.000000 |

关键结论：

- `implicit_acc` 长期为 `1.0`，不是健康成功，而是偏好排序过早饱和。
- `loser_dominant_ratio` 从头到尾几乎为 `1.0`，说明正确排序主要来自 loser 被进一步恶化。
- `win_gap` 长期为正，且 `mse_win` 显著高于 `ref_mse_win`，说明 policy 在 GT/winner 上比 reference 更差。
- `DGR` 多次降到接近 `0`，DPO 梯度在中后期基本失活。

这说明当前 vanilla DPO 的主要问题不是“训练步数不够”，而是**目标函数和偏好对组合后存在明确捷径**：模型通过扩大 loser 误差完成排序，而不是提高 winner/GT 侧质量。

### 6.3 训练停止与 OOM 现象

H20 上出现过两类停止：

1. **SIGHUP 终止**
   - 早期 `nohup bash -lc '...'` 运行在当前 shell/session 生命周期下，断开或关闭终端后触发 `SIGHUP`，导致 `torch.distributed.elastic` 关闭 workers。
   - 后续改为 `setsid nohup bash -lc '...' </dev/null >/dev/null 2>&1 &`，避免随终端关闭而退出。

2. **CUDA OOM**
   - 2-GPU Stage 1 训练曾报：
     ```text
     torch.cuda.OutOfMemoryError: CUDA out of memory. Tried to allocate 320.00 MiB.
     ```
   - 日志 grep 未出现 `Running validation` 或 `Saved state`，因此该次 OOM 更像是训练前向/反向过程的显存峰值，而不是 validation 本身撑爆。

为降低显存，后续引入过：

- `GRADIENT_CHECKPOINTING=1`
- `bf16`
- policy/ref forward 内存峰值优化
- pos/neg split forward 选项
- H20 外部日志根目录
- `MAIN_PROCESS_PORT=0` 自动解析为真实空闲端口

但本轮实验显示：显存与启动稳定性只是工程问题，真正的科学问题仍是 vanilla DPO 指标退化。

### 6.4 `MAIN_PROCESS_PORT=0` 与 split forward 的工程结论

H20 上曾使用：

```bash
MAIN_PROCESS_PORT=0
```

直接传给 `accelerate` 后，出现：

```text
DistNetworkError: timed out while trying to connect to (127.0.0.1, 0)
```

因此启动脚本已修复：当 `MAIN_PROCESS_PORT=0` 或 `auto` 时，先在 localhost 上解析一个真实空闲 TCP 端口，再传给 `accelerate`。

另外，`SPLIT_POS_NEG_FORWARD=1` 在 H20 DDP 下不稳定：

| GPU 配置 | `SPLIT_POS_NEG_FORWARD` | 结果 |
|---|---:|---|
| `0,1,4,5,6,7` | `1` | Step 0 `SIGFPE` |
| `4,5,6,7` | `1` | Step 0 `SIGFPE` |
| `4,5,6,7` | `0` | 成功跑完 600 step |

当前建议：**正式实验暂时使用 `SPLIT_POS_NEG_FORWARD=0`**。split forward 后续可单独在 1 卡/无 gradient checkpointing/无 bf16 的条件下排查。

### 6.5 β=50 的 600-step 探针实验

成功完成的一次关键探针：

| 项 | 值 |
|---|---|
| Run name | `h20-dpo-stage1-probe-beta50-600step-gpu4567-nosplit` |
| GPU | `4,5,6,7` |
| `MAX_STEPS` | `600` |
| `BATCH_SIZE` | `1` |
| `GRAD_ACCUM` | `1` |
| `BETA_DPO` | `50` |
| `SPLIT_POS_NEG_FORWARD` | `0` |
| W&B run | `https://wandb.ai/WorldModel_11/DPO_Diffueraser/runs/zh4qk7px` |
| H20 log | `/home/nvme03/workspace/world_model_phys/Diffueraser_DPO_Log/20260421_224309_h20-dpo-stage1-probe-beta50-600step-gpu4567-nosplit/train_stdout.log` |
| Last weights | `/home/nvme01/Video_inpainting_DPO/experiments/dpo/stage1/20260421_224309_h20-dpo-stage1-probe-beta50-600step-gpu4567-nosplit/last_weights` |

Step 600 诊断表：

| 指标 | 值 | 判断 |
|---|---:|---|
| `L_dpo` | 0.007126 | 已很低 |
| `implicit_acc` | 1.000000 | 过度饱和 |
| `win_gap` | 0.124823 | winner/GT 变差 |
| `lose_gap` | 0.359212 | loser 变差 |
| `reward_margin` | -0.015021 | 排序方向表面正确 |
| `sigma_term` | 0.992983 | sigmoid 接近饱和 |
| `kl_divergence` | 0.121009 | policy 已明显偏离 ref |
| `mse_win` | 0.134755 | 高于 ref |
| `mse_lose` | 0.384165 | 高于 ref |
| `ref_mse_win` | 0.009932 | ref 在 GT 上好得多 |
| `ref_mse_lose` | 0.024953 | ref 在 loser 上也好得多 |
| `DGR` | 3.175381 | 梯度仍未完全死亡 |
| `inside_term_mean` | 5.840755 | 仍偏大 |
| `loser_dominant_ratio` | 1.000000 | 仍完全由 loser degradation 主导 |

结论：

- 将 `beta_dpo` 从 `500/2500` 降到 `50` 可以降低一部分饱和速度，但**不能解决根问题**。
- 即使 β=50，训练仍然主要通过恶化 loser 建立偏好差距。
- `win_gap > 0` 且 `mse_win >> ref_mse_win` 说明 GT/winner 侧没有被保护，policy 对 winner 的噪声预测变差。
- 600 step 结束时 DGR 仍大于 0，说明梯度尚未完全死亡，但方向已经不健康。

### 6.6 对当前偏好对构建的判断

当前数据构建方式的核心假设是：

```text
win = GT
lose = 人工制造的退化补全结果
```

这个方向本身适合 BR（Background Restoration），但本轮实验表明：

1. 负样本可能过于 catastrophic  
   `blur / hallucination / flicker` 这类人工退化和 GT 差距太大，导致 DPO 排序任务过于容易，`implicit_acc` 很快到 `1.0`。

2. 16 帧 chunk 评分偏短  
   16 帧与训练 `nframes=16` 对齐，显存和实现更简单，但对长程时序漂移、背景慢变、身份不一致等问题敏感度不足。

3. 选 absolute worst 会放大坏信号  
   每个 chunk 直接选最差负样本，容易让训练学到“识别/惩罚极端坏样本”，而不是学习真实修复质量排序。

下一版数据更适合改为：

```text
win = GT
lose candidates = DiffuEraser / ProPainter / CoCoCo / MiniMax / 旧 checkpoint / 弱采样配置
hard_neg = 从 candidate pool 中筛选 hard but plausible loser
```

也就是说，候选 loser 应来自真实 inpainting 模型输出，而不是只依赖人工制造的极端退化；筛选时不再选 absolute worst，而是保留“GT 明显更好，但 loser 仍然像一个合理修复结果”的中等难度样本。

### 6.7 当前科学结论

本轮 H20 实验给出的结论非常明确：

> **不建议继续对当前偏好对运行 vanilla DPO 长训。**

原因不是训练还不够长，也不是 β 还没调好，而是：

- `implicit_acc=1.0` 与 `loser_dominant_ratio=1.0` 同时出现；
- `win_gap` 长期为正；
- `mse_win` 明显高于 `ref_mse_win`；
- 降到 `beta=50` 后仍然出现相同模式；
- 因此模型的“成功排序”主要来自 loser degradation，而不是 winner improvement。

这与 `/home/hj/DPO如何融入/Region-Reg-DPO_完整数学推导_终版.md` 中对 vanilla DPO 的风险分析一致：纯 DPO 只有相对约束，没有持续的 winner/SFT 锚定，容易走向破坏 winner 的捷径。

## 七、当前状态与下一步

### 当前状态

- ✅ Stage 1 / Stage 2 DPO 基础代码已跑通并完成多轮工程修复
- ✅ H20 日志已统一写入 `/home/nvme03/workspace/world_model_phys/Diffueraser_DPO_Log`
- ✅ `MAIN_PROCESS_PORT=0/auto` 已修复为自动解析真实端口
- ✅ 原始 2-GPU vanilla DPO 长训尝试已完成复盘，指标不健康
- ✅ β=50 / 600-step / 4-GPU / no-split 探针已成功跑完，仍显示 vanilla DPO 退化
- ⚠️ `SPLIT_POS_NEG_FORWARD=1` 在 H20 DDP 下会触发 `SIGFPE`，暂不作为默认路径
- ⚠️ 当前偏好对与纯 DPO objective 不适合继续直接长训

### 下一步建议

1. **先实现 winner/SFT anchor，而不是继续 vanilla DPO 长训**

   最小版本：

   ```text
   loss = dpo_loss + λ * mse_win
   ```

   或者：

   ```text
   loss = dpo_loss + λ * relu(win_gap)
   ```

   目标是强制 `win_gap` 回到 `<= 0` 附近，避免模型靠破坏 winner 获胜。

2. **随后升级到 Region-Reg-DPO**

   对应文档中的分区正则项：

   ```text
   loss = region_dpo_loss
        + rho_h * mse_win_hole
        + rho_b * mse_win_boundary
        + rho_c * mse_win_context
   ```

   初始可先用统一 winner anchor 验证方向，再引入洞内/边界/上下文分区权重。

3. **重构偏好对数据**

   下一版数据建议：

   - `win = GT`
   - `lose candidates = DiffuEraser + ProPainter + CoCoCo + MiniMax + old checkpoints`
   - mask 外区域全部强制 composite 回 GT，确保 pos/neg 收到同一道考题
   - 评分窗口从单一 16 帧扩展到 32/48 帧，增强对时序一致性的敏感度
   - 不再直接选 absolute worst，而是筛选 hard-but-plausible loser

4. **VideoDPO 复现降级为参考校准，不作为当前主线 blocker**

   VideoDPO 的开源实现可以用于观察成功 DPO 工作中的曲线形态，但当前实验已经足够说明：本项目的主要矛盾在自己的 pair 构造和缺少 winner regularization，而不是缺少完整 VideoDPO 复现。

5. **下一轮探针建议**

   ```text
   BETA_DPO=5 或 10
   WINNER_ANCHOR_WEIGHT=0.05 / 0.1 / 0.2
   SPLIT_POS_NEG_FORWARD=0
   MAX_STEPS=600
   ```

   健康目标：

   - `implicit_acc` 不要长期贴 `1.0`
   - `loser_dominant_ratio` 不要长期贴 `1.0`
   - `win_gap` 接近 `0` 或 `< 0`
   - `lose_gap > 0`
   - `mse_win` 不应显著高于 `ref_mse_win`
   - `sigma_term` 不要长期接近 `1.0`
