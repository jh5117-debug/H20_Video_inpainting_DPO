# H20 Multimodel DPO Data Generation

目标是不改训练侧 DPO Dataset 逻辑，只重新生成更可信的负样本：

- `win = GT`
- `lose = ProPainter / COCOCO / MiniMax-Remover` 等完整模型输出
- 每个候选输出先做硬合成：`candidate = mask * model_output + (1 - mask) * GT`
- 保存随机 hard mask，不再用完删除
- 评分使用现有 `inference.metrics` 的 PSNR/SSIM/LPIPS，VBench 作为可选项
- 最终仍输出训练代码已经支持的 `neg_frames_1/neg_frames_2`

## 1. H20 上准备第三方代码、环境、权重目录

```bash
cd /home/nvme01/Video_inpainting_DPO
git pull --ff-only origin main

bash DPO_finetune/scripts/setup_multimodel_h20.sh
```

这个脚本会：

- clone/update:
  - `https://github.com/sczhou/ProPainter.git`
  - `https://github.com/zibojia/COCOCO.git`
  - `https://github.com/zibojia/MiniMax-Remover.git`
  - `https://github.com/lixiaowen-xw/DiffuEraser.git`
  - `https://github.com/Vchitect/VBench.git`
- 在 `/home/nvme01/Video_inpainting_DPO/third_party_video_inpainting` 下建立：
  - `repos/`
  - `weights/`
  - `envs/`
  - `manifests/`
- 导出现有 DiffuEraser 环境信息。
- 搜索 H20 上的 DAVIS / YouTube-VOS full-resolution 数据集路径。

第三方权重优先用统一脚本下载：

```bash
bash DPO_finetune/scripts/download_multimodel_weights_h20.sh
```

该脚本会把权重整理到同一个根目录：

```text
/home/nvme01/Video_inpainting_DPO/third_party_video_inpainting/weights/
├── COCOCO_weight/
│   ├── cococo/model_0.pth ... model_3.pth
│   └── stable-diffusion-v1-5-inpainting/
├── minimax/vae,transformer,scheduler
└── propainter/
```

脚本还会生成：

```bash
/home/nvme01/Video_inpainting_DPO/third_party_video_inpainting/WEIGHTS_TODO.md
```

## 2. 准备 adapter 配置

```bash
cd /home/nvme01/Video_inpainting_DPO

cp DPO_finetune/configs/multimodel_adapters_h20.example.json \
   DPO_finetune/configs/multimodel_adapters_h20.json

vim DPO_finetune/configs/multimodel_adapters_h20.json
```

默认三个模型都已经接到了本 repo 的 wrapper：

```bash
python DPO_finetune/infer_propainter_candidate.py ...
python DPO_finetune/infer_cococo_candidate.py ...
python DPO_finetune/infer_minimax_candidate.py ...
```

- COCOCO 需要 prompt。
- ProPainter / MiniMax-Remover 不需要 prompt。
- DiffuEraser SFT 不作为默认负样本来源，只保留代码和权重用于对照实验。

## 3. Prompt 来源

COCOCO 是 text-guided video inpainting，因此推荐先用 Qwen/VLM 对每个视频生成一句场景 prompt，保存成 JSON：

```json
{
  "davis_bear": "a bear walking in a natural outdoor scene",
  "ytbv_xxxxx": "a handheld outdoor video with people and background objects"
}
```

生成器用法：

```bash
CAPTION_JSON=/path/to/captions.json \
bash DPO_finetune/scripts/run_multimodel_dpo_generation_h20.sh
```

如果没有 `CAPTION_JSON`，脚本会用视频目录名生成一个保守 fallback prompt，但 COCOCO 的质量会受影响。

## 4. 一键生成 DPO 数据

默认使用 GPU `0,1,2,3`，输出到 `/home/nvme03/workspace/world_model_phys/DPO_Finetune_Data_Multimodel_v1`：

```bash
cd /home/nvme01/Video_inpainting_DPO

CUDA_VISIBLE_DEVICES=0,1,2,3 \
GPUS=0,1,2,3 \
NUM_VIDEOS=0 \
MAX_FRAMES=48 \
HEIGHT=512 \
WIDTH=512 \
ENABLE_LPIPS=1 \
ENABLE_VBENCH=0 \
bash DPO_finetune/scripts/run_multimodel_dpo_generation_h20.sh
```

先小样本 smoke test：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
GPUS=0,1,2,3 \
NUM_VIDEOS=4 \
MAX_FRAMES=32 \
ENABLE_LPIPS=0 \
ENABLE_VBENCH=0 \
bash DPO_finetune/scripts/run_multimodel_dpo_generation_h20.sh
```

更推荐先跑 combined smoke：每个模型都对两个视频生成预览，并尽量并行占用 GPU：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
GPUS=0,1,2,3 \
REFRESH_ADAPTER_CONFIG=1 \
METHODS=propainter,cococo,minimax \
NUM_VIDEOS=2 \
MAX_FRAMES=32 \
bash DPO_finetune/scripts/smoke_multimodel_h20.sh
```

看这些文件：

```text
<SMOKE_ROOT>/<video>/candidates/<method>/previews/gt_mask_raw_comp.mp4
```

四栏分别是 `GT / mask overlay / raw model output / composited candidate`。

如果只想先测试 ProPainter：

```bash
METHODS=propainter \
NUM_VIDEOS=4 \
ENABLE_LPIPS=0 \
bash DPO_finetune/scripts/run_multimodel_dpo_generation_h20.sh
```

## 5. 输出结构

```text
DPO_Finetune_Data_Multimodel_v1/
├── manifest.json
├── generation_summary.json
└── davis_xxx_mask2026.../
    ├── gt_frames/
    ├── masks/
    ├── candidates/
    │   ├── propainter/
    │   │   ├── raw_output/
    │   │   ├── normalized_raw/
    │   │   ├── composited/
    │   │   └── inference.log
    │   ├── cococo/
    │   └── minimax/
    ├── neg_frames_1/
    ├── neg_frames_2/
    └── meta.json
```

`meta.json` 会记录：

- mask seed / 运动方式 / dilation
- 所有候选模型的分数
- `neg_frames_1/2` 分别来自哪个模型
- 当前 source balancing 计数

## 6. 训练入口

生成完成后，训练仍然使用现有脚本，只改数据根目录：

```bash
cd /home/nvme01/Video_inpainting_DPO

DPO_DATA_ROOT=/home/nvme03/workspace/world_model_phys/DPO_Finetune_Data_Multimodel_v1 \
CUDA_VISIBLE_DEVICES=1,2,3 \
bash scripts/h20_run_dpo_stage1.sh
```

## 7. 为什么不直接全排序或全塞进去

PSNR/SSIM 会偏向 ProPainter 这种传播式方法，因为它更贴近 GT 像素，但不一定主观质量最好。VBench/LPIPS 也各有偏差。

所以当前脚本做的是：

- 保存所有模型候选；
- 计算多指标；
- 不把某一个综合分当成绝对真理；
- 以完整模型输出作为负样本；
- 按 source balancing 选 `neg_frames_1/2`，避免某一种模型缺陷支配所有 DPO pair。

这比旧的人工拼接/极端负样本更接近真实模型错误分布。
