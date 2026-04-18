# DiffuEraser 首次全量微调需求文档

> **微调方案**：基于 YouTube-VOS + DAVIS 数据集，对 DiffuEraser 进行两阶段 (Stage 1 + Stage 2) 全量微调。

---

## 1. 训练监控与日志体系

### 1.1 日志方案选型

审查训练脚本的日志输出机制，确认当前采用的是纯文本日志、TensorBoard 还是 Weights & Biases (W&B) Dashboard。

### 1.2 W&B 集成方案

| 步骤 | 说明 |
|------|------|
| 创建 Organization | 在 W&B 平台创建 Organization (OR) |
| 创建 Project | 在 OR 下创建训练项目 |
| 添加合作者 | 将合作者账号加入 OR，共享 Dashboard 访问权限 |
| 训练脚本配置 | 在训练代码中集成 W&B，写入 API Key 等配置 |

**W&B Dashboard 应监控的指标**：
- Training Loss Curve
- Learning Rate
- 验证指标：PSNR、SSIM（每次 validation 步输出所有验证视频的**平均值**）

### 1.3 当前代码中 W&B 的实际行为（已落地）

当前 Stage 1 和 Stage 2 训练脚本都已经完成了较完整的 W&B 接入，不再只是“能画曲线”，而是把训练监控、文本日志和 crash 回溯都串了起来。

**1. 最早初始化，GPU 任务一启动就尽快创建 run**
- `accelerator.init_trackers()` 被放在 `Accelerator()` 初始化之后、任何 tokenizer / model / dataset 加载之前执行。
- 这意味着从实际使用体验上看，只要集群任务开始真正进入训练主进程，W&B 页面就会很快出现新的 run，而不是等模型加载完半天之后才建 run。
- 现在这一步还是 **fail-fast** 的：如果 W&B 初始化失败，训练不会继续偷偷空跑，而是直接报错退出，避免出现“GPU 已经在跑，但网页上一个 run 都没有”的黑箱情况。

**2. W&B 不只出图表，也会同步训练期文本日志**
- 图表层面：`accelerator.log(...)` 和 `wandb.log(...)` 发送的标量会进入 Charts，包括 `dpo_loss`、`implicit_acc`、`mse_w`、`mse_l`、`win_gap`、`lose_gap`、`reward_margin`、`sigma_term`、`kl_divergence`、学习率，以及 validation 的均值指标（Stage 1 的 PSNR/SSIM，Stage 2 的 PSNR/SSIM/Ewarp/TC）。
- 文本日志层面：训练脚本在 `main()` 一开始就把当前进程的 `stdout/stderr` tee 到 `output_dir/console_logs/rank*.log`，并把这些文件以 `policy="live"` 同步到 W&B run Files。
- 因此，`logger.info(...)`、`logger.warning(...)`、`print(...)`、大部分 traceback，以及多卡下各 rank 的文本输出，都会在 W&B 中保留，不再只存在于集群终端或 SLURM `.out` 文件里。
- 训练结束或异常退出时，脚本还会再执行一次 `policy="now"` 的补传，尽量把最后一段日志刷进 W&B。

**3. “W&B 输出所有 log” 这句话更准确的理解**
- 从训练脚本进入 `main()` 并完成 console tee 之后，训练期的绝大多数文本输出都能在 W&B 中看到。
- 但 W&B `Logs` 页面并不保证和集群终端 100% 逐字符一致，尤其是 `tqdm` 这类动态刷新内容。
- 最完整的文本记录实际上在 W&B `Files` 里的 `console_logs/rank*.log`。
- 仍有少量边界情况不能夸大：如果程序在进入 `main()` 之前就崩溃，或者是节点被系统硬杀（如极端 OOM / SIGKILL）导致来不及 flush，那么最原始的证据仍然是 SLURM `.out` 文件。

**4. Crash 时会把完整 SLURM 日志补回同一个 run**
- 训练脚本在 W&B 初始化成功后，会把当前 run 的关键信息落盘到输出目录。
- 如果 launcher 进程检测到训练子进程非零退出，`run_dpo_stage1.py` 和 `run_dpo_stage2.py` 会恢复到同一个 W&B run，并把对应的完整 SLURM 日志上传为 `slurm_full_crash_output.out`。
- 同时还会生成 `launcher_crash_summary.txt`，并发送 `wandb.alert(...)`，这样即使 live logs 不完整，也能直接在 W&B 页面下载最完整的 crash 日志。
- 训练进程自己的 Python traceback 至少会通过 `logger.error(...)`、console log 文件和 W&B alert 保留下来；其中 Stage 1 还会额外写入 `train_process_traceback.txt` 并上传到 run Files。

**5. 认证与缓存路径已经踩过坑并完成修复**
- SLURM 脚本现在统一把 `WANDB_DIR`、`WANDB_CACHE_DIR`、`WANDB_DATA_DIR`、`WANDB_CONFIG_DIR` 指到项目目录内，避免把缓存和 staging 文件写爆 home 配额。
- 同时，如果环境里没有现成的 `WANDB_API_KEY`，脚本会从 `~/.netrc` 解析并导出，避免只改了 `WANDB_CONFIG_DIR` 之后导致节点端拿不到认证信息。
- 这套配置已经在 Stage 1 和 Stage 2 的 sbatch 脚本里对齐，不再存在“某一阶段能登录、另一阶段 silent fail”的不一致问题。

---

## 2. 脚本架构与职责分离

### 2.1 SLURM 脚本定位

SLURM 脚本**仅作为 Python 脚本的启动器**，不承载任何业务逻辑（不进行权重转换、路径拼接等操作）。所有训练、转换、评估逻辑均封装在 Python 文件中。

### 2.2 两组启动脚本

| 模式 | 说明 |
|------|------|
| **联合训练** | Stage 1 → Stage 2 一键串行执行（Stage 1 训练完成后仍保存中间权重） |
| **独立训练** | Stage 1 和 Stage 2 可分别独立提交运行 |

---

## 3. 路径与环境变量规范

### 3.1 项目根目录

所有代码文件直接平铺部署于：

```
/sc-projects/sc-proj-cc09-repair/hongyou/dev/Reg_DPO_Inpainting/
```

合作者通过 `git clone` 至 `/sc-projects/sc-proj-cc09-repair/hongyou/dev/` 目录下，自动生成上述路径。

项目根目录下不允许嵌套多余的中间文件夹。

### 3.2 环境变量

| 变量名 | 用途 |
|--------|------|
| `$HF_TOKEN` | Hugging Face API Token |
| `$WANDB_API_KEY` | Weights & Biases API Key |
| `$PROJECT_HOME` | 项目根目录路径 |

所有脚本应通过上述环境变量加载路径，**禁止硬编码绝对路径**。

---

## 4. 权重保存与转换

### 4.1 训练结束后的权重转换

每个 Stage 训练结束后，在训练脚本内部**自动衔接**权重转换流程：

1. 完成 `accelerator.save_state()` 保存 Accelerator 格式 checkpoint；
2. 紧接着将 checkpoint 转换为标准 Diffusers 格式（`unet_main/config.json` + `model.safetensors`）。

转换逻辑内聚于训练脚本自身，**不拆分为独立的转换脚本文件**。

### 4.2 定期 Checkpoint 转换与 W&B 上传

- 每次 `checkpointing_steps` 保存 checkpoint 时，若权重转换耗时在可接受范围内，**同步执行转换**（而非仅在最后一步转换）。
- 转换后的权重上传至 W&B Artifact，便于从 W&B 网站直接下载任意 checkpoint 的可用权重。

---

## 5. Bug 修复

### 5.1 Stage 1 保存逻辑错误

**问题**：Stage 1 脚本末尾调用了 `unet_main.save_motion_modules()`，但 Stage 1 的 `unet_main` 是 `UNet2DConditionModel`，不包含 Motion Modules。

**修复**：Stage 1 仅保存 `UNet2DConditionModel` 和 `BrushNet` 的权重，`save_motion_modules()` 调用仅适用于 Stage 2。

---

## 6. 多卡训练与步数配置

### 6.1 max_train_steps 语义

`max_train_steps` 表示**每张 GPU 的最大优化步数**，而非多卡的累计总步数。

> 例：`max_train_steps=26000`，8 卡并行时，每张卡各训练 26000 步。

### 6.2 当前配置

| 参数 | 值 |
|------|-----|
| `max_train_steps` | 26000（每卡） |
| `validation_steps` | 2000（每 2000 步执行一次 validation，用于监控过拟合） |

---

## 7. 验证数据集规范

### 7.1 数据源

验证集采用 **ProPainter 预处理的 432×240 分辨率版本**，配合特定的 mask 数据。

验证集**独立于训练数据集**，严禁使用训练数据进行验证。

### 7.2 数据目录结构

```
data_val/
├── JPEGImages_432_240/   # 视频帧 (432×240)
│   ├── bear/
│   ├── boat/
│   └── ...               # 共 50 个视频
└── test_masks/           # 对应的分割 mask
    ├── bear/
    ├── boat/
    └── ...
```

### 7.3 验证流程

训练脚本**自动扫描** `data_val/JPEGImages_432_240/` 下的所有视频子目录，对每个视频执行推理并计算 PSNR/SSIM，最终在 W&B 中仅输出所有视频的**平均 PSNR** 和**平均 SSIM** 两条指标曲线，不生成逐视频的可视化。

---

## 8. 首次全量微调“踩坑”与经验总结（含反面教材）

作为第一次进行 SFT (Supervised Fine-Tuning) 的经验沉淀，以下几点工程规范至关重要。如果不遵守这些规范，极易导致训练失败、显卡资源浪费或无法进行 Debug 回溯。

### 8.1 路径与协作规范 (防“路径依赖”)
- **正确的做法**：严格使用代码仓的相对路径或环境变量定义基础路径，保证在合作者集群（如 `/sc-projects/.../Reg_DPO_Inpainting`）和本地环境机上代码都能直接运行。
- **如果不这么做的后果**：哪怕把一个本地绝对路径（如 `/home/hj/...`）硬编码写死在代码里，合作者 `git pull` 后立刻就会触发 `FileNotFoundError` 或 `[Errno 13] Permission denied` 报错。轻则导致评测模块崩溃缺失，重则数据集加载瘫痪，训练从一开始就无法启动。

### 8.2 模型加载与参数防呆 (防“盲人摸象”式炼丹)
- **正确的做法**：在训练最开始（就在进度条刚出来之前），必须打印一份清晰的模型参数报表。标明每个核心网络组件的总参数量（total）、可训参数量（trainable）和冻结参数量（frozen）。
- **如果不这么做的后果**：深度学习的代码只要网络跑起来往往就不会报错。如果不打印这些信息，你很容易遇到“跑了 3 天代码，最终发现整个网络的梯度其实被意外截断全冻结了（损失没下降）”，或者是“只想训 Stage 2 的 Motion Module，结果不小心连 8 亿参数的整个 UNet 主体一块改变了”。在最开始打出参数量，能一秒判断是不是“训歪了层”。

### 8.3 阶段性评测指标对齐 (防“南辕北辙”)
- **正确的做法**：
  - **Stage 1 (2D + BrushNet)**：只关注空间画质指标（PSNR、SSIM）。
  - **Stage 2 (Motion Module)**：除了 PSNR/SSIM，**必须加入时序一致性指标**（直接复用代码库中现成的 `metrics.py`，调用诸如 VFID、Ewarp 等函数，不再重新造相似的计算轮子）。
- **如果不这么做的后果**：如果你在 Stage 2 依然只死只看 PSNR/SSIM，往往会选出一个“虽然每一帧都抠得很细致，但一旦连贯播放由于缺失时序约束，帧与帧之间剧烈跳闪”的废弃权重。没有明确的时序打分指标约束，专注于处理时间维度的 Stage 2 训练就会变成一盘散沙。

### 8.4 日志整洁可用度 (防“大海捞针”)
- **正确的做法**：
  - **降噪清理**：使用系统函数屏蔽那些无关痛痒的第三方库 `deprecation` 警告；把每次 Forward 都刷屏没用的 log（如 `Forward upsample size...`）从 `INFO` 降级为 `DEBUG` 或剔除。
  - **表格统计**：不输出单视频杂碎分，而是在 Validation 计算完 50 个视频后仅输出一张美观规整的一目了然平均分表格（比如跳过无关紧要的 step 1，从 step 2000 以后有价值的时候开始打表输出）。
- **如果不这么做的后果**：如果对此前输出近一万次的 `Forward upsample size` 置之不理，训练哪怕只跑几个 epoch，日志就会迅速膨胀成一座没法查看的几十兆的“垃圾山”。当后续训练遇到显存溢出、NaN Loss 最致命的问题报错时，由于刷屏过快，在超长滚日志中很难定位报错最初始的一行到底是什么。同时，如果逐个乱糟糟输出 50 个验证视频的分数，由于极其杂乱毫无对比性可言，直接拉低了整体训练架构的工程感与可控性。

### 8.5 WandB 初始化必须放在最前面 (防"黑箱崩溃")
- **正确的做法**：在 `main()` 函数最前面（仅在 `Accelerator()` 初始化之后、任何模型/数据加载之前）就完成 `accelerator.init_trackers()`。同时，这一步必须是 **fail-fast** 的，初始化失败要直接退出，不能继续训练。`__main__` 入口再用全局 `try-except` 包裹整个 `main()`，把异常通过 `logger.error()`、`wandb.alert()`、traceback 文件和 launcher 补传日志一起写回同一个 W&B run。
- **如果不这么做的后果**：如果 WandB 初始化放在模型加载、数据集创建等步骤之后，一旦这些步骤崩溃（路径错误、权重格式不兼容、数据集为空等），Dashboard 上会一片空白，你只能去集群的 SLURM stdout 文件里大海捞针。更糟的是，如果初始化失败却继续训练，就会出现“GPU 已经开始烧卡，但 W&B 上没有任何 run、也没有任何曲线”的典型黑箱问题。

### 8.6 生成的文件绝不逃出项目目录 (防"磁盘配额"炸弹)
- **正确的做法**：在 SLURM 脚本中显式设置以下环境变量，将所有第三方库的缓存/日志/临时文件全部重定向到项目存储目录内：
  ```bash
  export WANDB_DIR="${PROJECT_ROOT}/.wandb_cache"
  export WANDB_CACHE_DIR="${PROJECT_ROOT}/.wandb_cache"
  export WANDB_DATA_DIR="${PROJECT_ROOT}/.wandb_cache"
  export WANDB_CONFIG_DIR="${PROJECT_ROOT}/.wandb_cache/config"
  export HF_HOME="${PROJECT_ROOT}/.hf_cache"
  export TRANSFORMERS_CACHE="${PROJECT_ROOT}/.hf_cache"
  ```
- **补充一条真实踩坑经验**：如果显式改了 `WANDB_CONFIG_DIR`，还必须确保节点端能拿到 `WANDB_API_KEY`。最稳妥的做法是在 sbatch 中检测环境变量缺失时，从 `~/.netrc` 解析并导出。否则极易出现“训练进程已经启动，但 W&B 因认证失败没有创建 run”的问题。
- **如果不这么做的后果**：集群的 `/home/` 目录通常只有 10~50 GB 配额，而项目存储 `/sc-projects/` 有几百 TB。WandB 和 HuggingFace 默认将缓存写入 `~/.local/share/wandb/` 和 `~/.cache/huggingface/`。一次 artifact 上传就能产出数 GB 的 staging 文件，瞬间撑爆 home 目录配额，导致 `[Errno 28] No space left on device`，不仅当前训练中断，还会连带影响合作者的其他进程。

### 8.7 Tensor 维度一致性检查 (防"静默维度广播"陷阱)
- **正确的做法**：在编写涉及 `timesteps`、`encoder_hidden_states` 等需要与 `noisy_latents` 对齐的张量时，必须手动推算每个张量在 concat / repeat 后的 shape，确保与模型 `forward` 内部的 `expand` / `repeat_interleave` 行为一致。不同模型对 timesteps 维度的期望可能不同：
  - `UNet2DConditionModel` / `BrushNet`：`timesteps.expand(sample.shape[0])` → 期望 per-frame `(bsz*nframes,)`
  - `UNetMotionModel`：`timesteps.expand(sample.shape[0] // num_frames)` → 期望 per-batch `(bsz,)`
- **如果不这么做的后果**：PyTorch 的 `expand` 在 `bsz=1` 时从 `(1,)` → `(N,)` 碰巧能跑通，但 DPO（batch concat 翻倍）或多卡（DDP）场景下立刻 `RuntimeError: The expanded size of the tensor (32) must match the existing size (2)`。建议在训练循环第一步加 `assert timesteps_all.shape[0] == noisy_all.shape[0]` 做运行时校验。

### 8.8 DPO Finetune 的 High-Level Tricks
- **偏好对条件统一**：pos（GT）和 neg（退化修复结果）必须共享完全相同的 BrushNet 条件（GT masked image + mask），绝对不能让 neg 样本用自己的修复结果作为条件，否则会导致信息泄漏——模型学到的是"条件图长什么样"而非"修复质量的好坏"。
- **DAVIS 过采样**：DAVIS 只有 ~30 个视频但质量极高，YouTube-VOS 有 ~3400 个。若不对 DAVIS 进行 10x 过采样，模型几乎只在 YouTube-VOS 上训练，丢失高质量数据的学习机会。
- **Chunk-Aligned 采样**：DPO 数据集的负样本是分 chunk 生成的（每 16 帧一个 chunk），采样时必须对齐 chunk 边界，否则跨缝合线采样会引入人工 artifact，污染 loss 信号。
- **Sigma Term 监控**：`sigmoid(inside_term)` 的均值如果接近 1.0，说明 DPO loss 已饱和，梯度消失，训练实质上停滞。此时应降低 `beta_dpo` 或减小学习率。
- **权重保存策略**：Stage 1/2 各只保存 best + last 两个权重。不要每隔 2000 步就上传到 WandB，存储配额很快会用光。
- **DGR (Degenerate Gradient Ratio)**：记录 `grad_norm / initial_grad_norm` 的比值曲线。如果比值降到 0.01 以下，说明梯度消散严重，训练已退化。
