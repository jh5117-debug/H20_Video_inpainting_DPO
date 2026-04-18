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

## 六、当前状态与下一步

### 当前状态
- ✅ Stage 1 代码审计完成，所有已知 bug 修复
- ✅ Stage 2 代码审计完成，所有已知 bug 修复
- ✅ 数据集路径兼容性修复
- ✅ WandB 异常捕获已添加
- ✅ 已完成首次 Stage 1 集群试跑与日志复盘
- ✅ 已完成 `beta_dpo=500` + 全局监控 + scope 化的代码修订
- ⏳ 待重新提交 Stage 1（beta=500）

### 下一步
1. 集群 `git pull` + 重新提交 Stage 1 训练（`beta_dpo=500`）
2. 重点监控前 500~1000 步的 `global/implicit_acc`、`global/inside_term_*`、`rank0/win_gap`、`global/loser_dominant_ratio`
3. Stage 1 完成后提交 Stage 2
4. 若 vanilla DPO 仍快速饱和，下一阶段再考虑引入论文 Reg-DPO 的 SFT regularization 项
5. 后续：Region-Reg 融合（不在本阶段范围内）

### 集群运行命令
```bash
cd ${PROJECT_HOME}/dev/Reg_DPO_Inpainting
git pull origin main
BETA_DPO=500.0 NUM_GPUS=8 MAX_STEPS=20000 VAL_STEPS=2000 sbatch DPO_finetune/scripts/03_dpo_stage1.sbatch
```
