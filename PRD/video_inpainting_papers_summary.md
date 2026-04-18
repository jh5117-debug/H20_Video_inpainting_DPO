# Video Inpainting 论文实验总结大表

> 基于12篇论文 + 开源代码仓库的综合分析  
> 评测协议参考: **STTN (ECCV 2020)** 首次定义 DAVIS/YouTube-VOS + stationary mask + 432×240 评测标准，经 **FuseFormer → E2FGVI** 固化为社区通用协议

---

## 1. 方法 × 数据集 × Mask × 分辨率 × 推理配置

| 方法 | 训练数据集 | 测试数据集 | 训练Mask类型 | 测试Mask类型 | 训练分辨率 | 推理分辨率 | 可手动设分辨率 | 每次最多输入帧数 | Clip/分段策略 | Clip重叠 | Blend (GT融合) | Blend高斯模糊 | Mask Dilation |
|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|
| **ProPainter** | YouTube-VOS (3471 videos) | DAVIS (50 clips) + YouTube-VOS (508 videos) | ⚠️ **仅随机mask**: 50% stationary(固定位置) + 50% moving(随机运动), 均为贝塞尔曲线随机形状, **非SAM/分割mask** | STTN/E2FGVI标准评测协议 Stationary Mask (**432×240**), object mask用于定性 | 432×240 | 432×240 (可resize) | ✅ `--resize_ratio` | 子序列80帧 (`subvideo_length`) | ✅ subvideo sliding | ✅ overlap (neighbor_length=10) | ✅ 用dilated mask做像素级合成 | ❌ | ✅ `flow_mask_dilates=8, mask_dilates=5` (代码CLI默认4) |
| **DiffuEraser** | Panda-70M (3.18M clips) | **无定量metric表格** (仅可视化对比) | ⚠️ **仅随机mask**: 同ProPainter, `create_random_shape_with_random_motion`, **非SAM/分割mask** | N/A (定性OR对比) | 512 | 最大1280 (自适应8对齐) | ✅ `--max_img_size` | 22帧/clip (`nframes=22`) | ✅ 先均匀采样22帧做pre-inference, 再逐clip推理 | ✅ pre-inference锚帧替换 | ✅ `blended=True` 用mask与原始GT融合 | ✅ `GaussianBlur((21,21),0)` | ✅ 先erode(1 iter) 再dilate(默认8 iter) |
| **AVID** | Shutterstock (私有) | 自建测试集 | ⚠️ **随机rectangular mask + 随机mask** (非SAM object mask, 未开源代码, 论文描述可能误导) | Object mask (SAM) + rectangular mask | 512×320~512×896 | 512分辨率 | ✅ | 约16帧 | ✅ MultiDiffusion temporal sliding window | ✅ overlap denoise averaged | ✅ 非mask区域保持原始像素 | 论文未明确 | 论文未明确 |
| **MiniMax-Remover** | Stage1: WebVid-10M (2.5M pairs); Stage2: Pexels (10K人工筛选) | DAVIS (90 videos) + Pexels (200 videos) | ⚠️ Stage1: 随机mask(**正方向**) + SAM2 mask(**CFG负方向**, 标记"生成物体"方向); Stage2: 人工筛选伪GT(**非原始GT**). 训练代码未开源. | SAM2 tracking object mask | 336×592 ~ 720×1280 (随机采样) | 480p, 81帧 | 未开源训练 | 81帧 | ❌ 一次处理 | N/A | 论文未明确 | 论文未明确 | 论文未明确 |
| **FFF-VDI** | YouTube-VOS training set | YouTube-VOS (508 videos) + DAVIS (50 clips) | ⚠️ **仅随机mask**: 50% stationary + 50% moving 随机形状 (同ProPainter, **非object mask**) | **STTN/E2FGVI标准评测协议 Stationary Mask (432×240)**, 使用ProPainter提供的同一套test mask | 448×256 (从432×240 pad) | 448×256 | ❌ 固定 | 训练14帧 | 论文未明确clip策略 | N/A | ✅ DDIM inversion保持非mask一致 | 论文未明确 | 论文未明确 |
| **CoCoCo** | WebVid-10M | 自建测试集 (300 videos) | Instance-aware region selection + random rectangular (**来自论文描述, 训练代码未开源无法验证**) | Instance-aware region selection | 256×256 (训练) | 256×256 或 512×320 | ✅ | 16帧 | 论文未明确 | N/A | 非mask区域保留 | 论文未明确 | 论文未明确 |
| **FloED** | Open-Sora (421K clips) | 自建benchmark (100 videos: 50 BR + 50 OR, 4K分辨率) | ⚠️ 论文声称: 随机mask(BR) + SAM object mask(OR), **但训练代码未开源无法验证**; 注意: SAM mask + GT做监督学到的是重建物体 | BR: 合成random mask; OR: SAM object mask | 512, 16帧 | 512, 16帧 (DDIM 25步) | ✅ | 16帧 (anchor frame策略) | ✅ anchor frame + latent interpolation | ✅ anchor帧作为keyframe | ✅ 用mask做区域合成 | 论文未明确 | 论文未明确 |
| **HomoGen** | 内部Shutterstock数据集 | YouTube-VOS + DAVIS + RORD | 随机生成stationary mask | **STTN/E2FGVI标准评测协议 Stationary Mask (432×240)** | 论文未明确 | 240×432 (评估) | 论文未明确 | GoP (Group of Pictures) 分段 | ✅ 重叠GoP | ✅ GoP之间overlap | ✅ blurred mask与GT blend | ✅ 使用blurred mask | ✅ |
| **RT-Remover** | Stage1: WebVid-10M; Stage2: Pexels + video removal system distill | DAVIS (90 videos) + Pexels (200 videos) | Stage1: 随机mask applied to videos; Stage2: SAM2 object mask + first-frame-only策略 | SAM2 tracking, 仅第一帧给mask | 480×832 | 480×832, 81帧 | 论文未明确 | 理论无上限 (auto-regressive) | ✅ auto-regressive + KV cache | ✅ KV cache滑动窗口 | 论文未明确 | 论文未明确 | 论文未明确 |
| **VideoComposer** | WebVid-10M + LAION-400M | 自建测试集 | 手绘/rectangular mask | 手绘/rectangular mask | 256×256 | 256×256, 16帧 | 论文未明确 | 16帧 | 论文未明确 | N/A | ❌ (主做视频生成) | N/A | N/A |
| **VideoRepainter** | 内部watermark-free视频 (约300K) | DAVIS (50 videos) + 20 in-the-wild editing samples | ⚠️ **代码验证**: SAM分割mask(从all_masks.npz加载) + **70%概率变形**为brush(30%)/rect(30%)/ellipse(20%)/circle(20%), ~30%保留原始SAM轮廓. 论文描述的spatial/temporal aug与代码不完全一致. | OR: object mask; editing: user-specified mask | 320×576(预训练) → 576×1024(微调) | 576×1024, 14~25帧(svd) | ✅ | 14帧(svd-base) / 25帧(svd-xt) | ✅ coarse-to-fine + MultiDiffusion | ✅ anchor帧 + overlap子序列 | ✅ symmetric mask condition | 论文未明确 | 论文未明确 |
| **LGVI** | ROVI (5650 videos) + GQA-Inpaint (images) | ROVI test (478 videos, 758 objects) | Object mask (来自referring segmentation GT, dilation d∈{0,3,5,7,10,15}) | referring segmentation mask (最佳dilation d=7) | 512×320 | 512×320, 24帧 | ❌ 固定 | 24帧 | 论文未明确 | N/A | ❌ 端到端输出 | N/A | ✅ 人工选择最佳dilation d=7 |
| **VACE** | 内部大规模视频数据 (Tongyi Lab, 未公开具体数据集) | 在MiniMax-Remover中作为baseline被评测 (DAVIS 90 + Pexels 200) | 多种mask类型经VCU输入, 训练时通过**MaskAugAnnotator增强**: original/hull/bbox各含expand变体(随机概率), 训练代码未完全开源 | SAM2 object mask (MiniMax评测中) | 论文未明确 (Wan2.1框架: 1.3B→480×832, 14B→720×1280) | 480×832 (1.3B) / 720×1280 (14B), ~81帧 | ✅ 支持任意分辨率输入 | ~81帧 (Wan2.1架构) | 论文未明确 | N/A | 通过mask条件保持非mask区域 | 论文未明确 | 论文未明确 |

---

## 2. 任务类型 (Metric 实际测量的Task)

> [!IMPORTANT]
> 此表仅标注各论文 **定量metric实际测量的任务**, 而非论文声称的能力范围。  
> 原因: OR (Object Removal) 没有像素级GT, 无法计算PSNR/SSIM。很多论文的OR只做可视化对比，metric实际测的是BR。  
> MiniMax-Remover和RT-Remover例外: 它们用PSNR/SSIM测OR结果的**背景区域**（不测mask内部），本质是**OR任务下的背景保持度**。

| 方法 | Metric测的Task | 论文声称/展示的具体Task | 备注 |
|:---|:---|:---|:---|
| **ProPainter** | BR (Background Restoration) | **BR**: 用stationary mask遮挡背景区域后恢复; **OR(仅可视化)**: 用DAVIS/YouTube-VOS object segmentation label做object mask定性展示去除前景物体 | Metric用STTN/E2FGVI协议stationary mask测BR; OR仅定性 |
| **DiffuEraser** | **无定量metric** | **BR+OR(仅可视化)**: 基于ProPainter prior做扩散去噪,展示前景物体移除(行人/车辆/文字等)和背景修复效果的可视化对比 | 论文只有可视化对比图,无任何metric表格 |
| **AVID** | Text-guided inpainting (视频编辑) | **Text-guided object replacement**: 用文本替换mask内物体(如"car"→"MINI Cooper"); **Content removal**: 用背景描述文本去除前景(如"A wheat field"); **Background editing**: 修改场景背景(如改变季节/天气); 支持任意时长视频 | 用BP/TC/TA定量评估;无PSNR/SSIM(无GT) |
| **MiniMax-Remover** | OR (背景保持度: PSNR/SSIM测背景区域) | **专注OR**: 输入视频+SAM2 object mask,用DiT直接生成去除目标物体后的视频; 支持81帧480p一次性推理; 6步即可高质量移除 | PSNR/SSIM只测非mask背景区域; VQ/Succ由GPT-O3评估mask内部 |
| **FFF-VDI** | BR (Background Restoration) | **BR**: First Frame Filling + I2V扩散模型恢复stationary mask遮挡区域; **OR(仅可视化)**: 展示去除DAVIS中的前景对象 | 使用ProPainter提供的同一套stationary mask |
| **CoCoCo** | Text-guided inpainting (视频编辑) | **Text-guided mask region filling**: 用文本描述填充mask区域内容(如在空区域生成指定物体); **Text-controlled object insertion**: 在mask区域插入文本描述的新物体; **支持适配个人化T2I模型**(如DreamBooth/LoRA风格) | 用CS/BP/TC/VQ/TA评估; 无PSNR/SSIM |
| **FloED** | BR + OR | **BR**: 用optical flow引导的扩散模型恢复synthetic random mask区域; **OR**: 用SAM object mask去除视频中的前景物体,用text prompt描述期望背景; 两个task都有定量metric | BR用random mask有GT; OR用SAM mask,评估BR指标+TC |
| **HomoGen** | BR (Background Restoration) | **BR**: 用homography propagation传播上下文像素+扩散模型生成缺失内容; **OR(仅可视化)**: 展示去除DAVIS中前景对象的定性效果 | 使用标准评测协议stationary mask; OR仅定性对比 |
| **RT-Remover** | OR (背景保持度: PSNR/SSIM测背景区域) | **实时OR**: 仅给第一帧object mask,auto-regressive DiT自动追踪+移除目标物体; 支持无限帧长度; 达到33FPS实时推理(2步采样); 集成tracking和removal为统一模型 | 与MiniMax-Remover相同策略 |
| **VideoComposer** | Compositional Video Synthesis (多任务) | **多条件视频合成**: text-to-video生成, image-to-video生成, depth/sketch引导生成, motion vector控制, **video inpainting(子功能)**: 给定mask+text用STC-encoder组合时空条件填充; 还支持style transfer和handwriting-to-video | 用Frame Consistency(CLIP cos sim) + EPE(motion) |
| **VideoRepainter** | 视频编辑 (Keyframe-guided inpainting) | **Keyframe-guided video inpainting**: 用户先用image inpainting编辑一帧keyframe,再用I2V模型传播到全视频; 具体支持: **object removal**(移除前景), **object replacement**(替换物体), **background change**(更换背景), **novel concept insertion**(插入新物体如Dreambooth概念), **virtual try-on**(虚拟试穿) | Cons(CLIP)/PSNR(Bg)/MSE(Bg)测背景; AS测美学 |
| **LGVI** | Language-driven OR (referring inpainting) | **Referring video inpainting**: 用自然语言指代(如"remove the cat on the right")定位并移除指定对象; **Interactive video inpainting**: 用对话式交互(chat-style)理解复杂用户请求并执行修复; 集成MLLM理解隐式/复杂语言指令 | PSNR/SSIM/VFID/Ewarp 在ROVI数据集(有GT)上测 |
| **VACE** | MV2V (Masked Video-to-Video) 多任务 | **All-in-one视频创作与编辑**: R2V(参考图生成视频), V2V(视频到视频编辑), MV2V(mask引导视频编辑/inpainting); 具体能力: Move-Anything, Swap-Anything, Reference-Anything, Expand-Anything(outpainting), Animate-Anything; **Inpainting为子任务之一**,非主要贡献; 通过VCU统一多任务条件输入 | 论文无专门inpainting定量metric; 在MiniMax-Remover中作为baseline, 使用SSIM/PSNR(bg)/TC/VQ/Succ评测 |

---

## 3. Evaluation Metrics 计算方式详解

| Metric | 全称 | 计算逻辑 | 各方法是否一致 |
|:---|:---|:---|:---|
| **PSNR** | Peak Signal-to-Noise Ratio | `10 * log10(MAX²/MSE)`, 像素级逐帧计算取平均。MAX=255 | ✅ 一致, 但**测量区域不同**: BR论文测全图; MiniMax/RT-Remover仅测非mask背景区域 |
| **SSIM** | Structural Similarity | 基于亮度、对比度、结构三分量, 逐帧求平均 | ✅ 一致 (同PSNR的区域差异) |
| **VFID** | Video Fréchet Inception Distance | 提取I3D特征,计算生成视频与真实视频特征分布间的Fréchet距离 | ✅ 基本一致 (使用I3D backbone) |
| **E_warp** | Warping Error | 用光流warp前一帧到当前帧,计算warp图像与实际帧在非遮挡区域的差异 | ✅ 一致, 注意单位: 有的报告 ×10⁻³, 有的报告 ×10⁻² |
| **TC** | Temporal Consistency | CLIP视觉特征在相邻帧间的余弦相似度 | ⚠️ backbone差异: MiniMax/RT用CLIP-ViT-h-b14; CoCoCo/AVID未明确指定 |
| **VQ** | Visual Quality | GPT-O3/GPT-5自动评分(1-10分) 或 人类主观评估 | ❌ **不一致**: MiniMax用GPT-O3; RT-Remover用GPT-5; CoCoCo用人类投票 |
| **Succ** | Success Rate | GPT-O3/GPT-5判定目标对象是否被成功移除(%) | 仅MiniMax-Remover和RT-Remover使用 |
| **BP** | Background Preservation | 非mask区域L1像素差距(scale 0-255, lower=better) | ⚠️ CoCoCo用L1; VideoRepainter用PSNR(Bg)/MSE(Bg) |
| **CS** | CLIP Score | CLIP计算文本与生成帧之间的对齐度 | ✅ CoCoCo使用 |
| **TA** | Text Alignment | 生成结果与输入文本描述的对齐度(CLIP-based) | ✅ 基本一致 |
| **Cons.** | CLIP Consistency | 相邻帧CLIP embedding余弦相似度 ×100 | VideoRepainter使用 |
| **AS** | Aesthetic Score | LAION aesthetic predictor逐帧评分 | 仅VideoRepainter使用 |
| **EPE** | End-Point Error | 光流预测端点误差 | VideoComposer用于motion control评估 |
| **Frame Consistency** | 帧一致性 | CLIP vision embedding相邻帧余弦相似度 | VideoComposer使用 |

---

## 4. 开源状态 × 代码仓库

| 方法 | 开源 | GitHub仓库 | `/home/hj/All_Repo`中已克隆 | 备注 |
|:---|:---:|:---|:---:|:---|
| **ProPainter** | ✅ | [sczhou/ProPainter](https://github.com/sczhou/ProPainter) | ✅ `ProPainter` | 推理代码完整 |
| **DiffuEraser** | ✅ | [modelscope/DiffuEraser](https://github.com/modelscope/DiffuEraser) | ✅ `DiffuEraser` | 推理 + 训练代码完整 |
| **AVID** | ❌ | 未开源 | ❌ | 论文close-sourced |
| **MiniMax-Remover** | ✅ | [MiniMax-AI/MiniMax-Remover](https://github.com/MiniMax-AI/MiniMax-Remover) | ✅ `MiniMax-Remover` | 仅推理,无训练代码 |
| **FFF-VDI** | ✅ | 有GitHub (未确认URL) | ❌ | 论文提到代码 |
| **CoCoCo** | ✅ | [zibojia/CoCoCo](https://github.com/zibojia/CoCoCo) | ✅ `COCOCO` | 推理代码 |
| **FloED** | ✅ | [BohaGU/FloED](https://github.com/BohaGU/FloED) | ✅ `FloED` | 推理代码 |
| **HomoGen** | ❌ | 未开源 | ❌ | 论文无代码链接 |
| **RT-Remover** | ❌ | 未开源 | ❌ | Under review at ICLR 2026 |
| **VideoComposer** | ✅ | [ali-vilab/videocomposer](https://github.com/ali-vilab/videocomposer) | ❌ | 未克隆 |
| **VideoRepainter** | ✅ | [VideoPainter](https://github.com/TencentARC/VideoPainter) | ✅ `VideoPainter` | 推理+训练完整; 目录名为VideoPainter |
| **LGVI** | ✅ | [jianzongwu/Language-Driven-Video-Inpainting](https://github.com/jianzongwu/Language-Driven-Video-Inpainting) | ❌ | 数据集+代码+模型公开 |
| **VACE** | ✅ | [ali-vilab/VACE](https://github.com/ali-vilab/VACE) | ✅ `VACE` | All-in-one视频创作与编辑 |

---

## 5. 关键代码细节 (来自开源仓库验证)

### 5.1 ProPainter (代码验证)

```python
# inference_propainter.py 关键参数
mask_dilation = 4           # CLI默认mask膨胀4次
flow_mask_dilates = 8       # 光流mask膨胀8次 (代码内默认)
mask_dilates = 5            # mask膨胀5次 (代码内默认)
subvideo_length = 80        # 子视频长度
neighbor_length = 10        # 邻近帧数
ref_stride = 10             # 参考帧步长
resize_ratio = 1.0          # 分辨率缩放比例 (可调)

# Blend方式: 用dilated mask做像素级合成 (无高斯模糊)
# comp = frame * (1-mask) + pred * mask
```

### 5.2 DiffuEraser (代码验证)

```python
# diffueraser/diffueraser.py 关键参数 (⚠️ 代码验证修正版)
mask_dilation_iter = 8      # run_diffueraser.py CLI默认8; forward()签名默认4
nframes = 22                # 每clip 22帧
max_img_size = 960          # ⚠️ run_diffueraser.py CLI默认960 (非1280); forward()签名默认1280
num_inference_steps = 2     # ⚠️ run_diffueraser.py默认ckpt="2-Step" → 2步推理
                            # 注意: L243硬编码 guidance_scale=0, 无论选哪个ckpt
                            # "Normal CFG 4-Step"虽然列出cfg=7.5, 但被硬编码覆盖

# Mask处理: 先erode(1次)去噪 → 再dilate(mask_dilation_iter次)扩展
m = cv2.erode(m, kernel_3x3, iterations=1)
m = cv2.dilate(m, kernel_3x3, iterations=mask_dilation_iter)

# Blend: GaussianBlur软边融合
mask_blurred = cv2.GaussianBlur(mask, (21, 21), 0) / 255.
binary_mask = 1 - (1 - mask/255.) * (1 - mask_blurred)
img = pred * binary_mask + original * (1 - binary_mask)

# Clip策略:
# 1. 帧数 > 44帧 → 先均匀采样22帧做pre-inference
# 2. 用pre-inference结果替换对应帧的latent和mask
# 3. 再逐clip (22帧/次) 推理全部帧
```

---

## 6. Benchmark 分组对比与现有 Benchmark 总结

### 6.1 按共享Benchmark分组

#### Group A: STTN/E2FGVI 标准评测协议 (BR Task)

| 属性 | 详情 |
|:---|:---|
| **测试集** | DAVIS (50 clips) + YouTube-VOS (508 videos) |
| **分辨率** | 432×240 |
| **Mask** | Stationary Mask (FuseFormer提供的`random_mask_stationary_w432_h240`) |
| **Metric** | PSNR, SSIM, VFID, E_warp |
| **使用此benchmark的论文** | **ProPainter**, **FFF-VDI**, **HomoGen** |
| **其他baseline** | STTN, FuseFormer, E²FGVI, FGT, FGVC, CPNet, DFVI, ISVI, TSAM |
| **特点** | 社区最通用的BR benchmark; 有完整像素级GT; mask和GT文件被广泛复用 |

> HomoGen额外增加了RORD数据集评测 (见Section 3.2)

#### Group B: DAVIS + Pexels OR评测 (OR Task, 背景区域PSNR/SSIM)

| 属性 | 详情 |
|:---|:---|
| **测试集** | DAVIS (90 videos) + Pexels (200 videos) |
| **分辨率** | 480p (MiniMax) / 480×832 (RT-Remover) |
| **Mask** | SAM2 tracking生成的object mask |
| **Metric** | SSIM, PSNR (仅背景区域), TC (CLIP-ViT-h-b14), VQ (GPT评分), Succ (GPT判定), User Preference |
| **使用此benchmark的论文** | **MiniMax-Remover**, **RT-Remover** |
| **共享baseline** | ProPainter, VideoComposer, CoCoCo, FloED, DiffuEraser, VideoPainter, VACE |
| **特点** | 专为OR设计; PSNR/SSIM只测背景保持度; 引入GPT-O3/GPT-5作为自动评估器 |

> MiniMax-Remover 和 RT-Remover 使用几乎相同的benchmark设计(DAVIS+Pexels), 但baseline有差异: MiniMax对比7个方法, RT-Remover对比3种forcing策略。两篇论文的DAVIS和Pexels测试集不完全相同(RT-Remover的VQ/Succ数值与MiniMax不直接可比,因为GPT版本不同)。

#### Group C: FloED 自建Benchmark (BR+OR)

| 属性 | 详情 |
|:---|:---|
| **测试集** | 自建100 videos (50 BR + 50 OR, 来自Pexels和Pixabay) |
| **分辨率** | 4K源 → 512推理 |
| **Mask** | BR: synthetic random mask; OR: SAM object mask |
| **Metric** | PSNR, SSIM, VFID, E_warp, TC, TA |
| **使用此benchmark的论文** | **FloED** |
| **baseline** | VideoComposer, CoCoCo, DiffuEraser |
| **特点** | 同时评估BR和OR; 4K高清源视频; 覆盖多样运动幅度 |

#### Group D: CoCoCo 自建Benchmark (Text-guided Inpainting)

| 属性 | 详情 |
|:---|:---|
| **测试集** | 自建300 videos |
| **分辨率** | 256×256 / 512×320 |
| **Mask** | Instance-aware region selection |
| **Metric** | CS (CLIP Score), BP (L1), TC (CLIP cos sim), VQ (User), TA (User) |
| **使用此benchmark的论文** | **CoCoCo** |
| **baseline** | AnimateDiffV3*, VideoCrafter2*, VideoComposer |
| **特点** | 专为text-guided editing设计; 无PSNR/SSIM; 依赖user study评VQ/TA |

#### Group E: VideoRepainter Benchmark (Keyframe-guided Editing)

| 属性 | 详情 |
|:---|:---|
| **测试集** | DAVIS (50 videos) + 20 in-the-wild editing samples |
| **分辨率** | 576×1024 |
| **Mask** | object mask / user-specified mask |
| **Metric** | Cons. (CLIP Consistency), PSNR(Bg.), MSE(Bg.), AS (Aesthetic Score) |
| **使用此benchmark的论文** | **VideoRepainter** |
| **baseline** | CoCoCo, I2VGen-XL, ModelScope, AnyV2V |
| **特点** | 分离inpainting和editing两个evaluation; 测背景一致性+美学 |

#### Group F: ROVI Benchmark (Language-Driven Inpainting)

| 属性 | 详情 |
|:---|:---|
| **测试集** | ROVI test set (478 videos, 758 objects) |
| **分辨率** | 512×320 |
| **Mask** | referring segmentation GT mask (with dilation) |
| **Metric** | PSNR, SSIM, VFID, E_warp |
| **使用此benchmark的论文** | **LGVI** |
| **baseline** | InstructPix2Pix, Inst-Inpaint, MagicBrush, Inpaint Anything* |
| **特点** | 首个language-driven video inpainting benchmark; 有完整GT; 支持referring和interactive两种模式 |

#### 独立/无Benchmark

| 方法 | 原因 |
|:---|:---|
| **DiffuEraser** | 论文无定量metric表格, 不构成独立benchmark |
| **AVID** | 使用自建测试集 + 私有Shutterstock数据, 未公开 |
| **VideoComposer** | 多任务视频合成, inpainting非主要贡献, 无统一inpainting benchmark |

---

### 6.2 目前该方向已有的 Benchmark 汇总

| Benchmark名称 | 来源 | Task | 测试集规模 | 分辨率 | Mask类型 | Metric | 公开可用 | 备注 |
|:---|:---|:---|:---|:---|:---|:---|:---:|:---|
| **STTN/E2FGVI Protocol** | STTN(ECCV2020) → FuseFormer(ICCV2021) → E2FGVI(CVPR2022) | BR | DAVIS 50 clips + YouTube-VOS 508 videos | 432×240 | Stationary Mask | PSNR, SSIM, VFID, E_warp | ✅ | 社区最广泛使用; mask文件由FuseFormer/E2FGVI提供 |
| **DAVIS+Pexels OR Protocol** | MiniMax-Remover / RT-Remover | OR | DAVIS 90 videos + Pexels 200 videos | 480p | SAM2 Object Mask | SSIM, PSNR(bg), TC, VQ(GPT), Succ(GPT) | ⚠️ Pexels测试集未公开 | 新兴OR benchmark; 引入GPT自动评估 |
| **FloED Benchmark** | FloED | BR+OR | 100 videos (50+50) from Pexels/Pixabay | 4K→512 | Random (BR) + SAM (OR) | PSNR, SSIM, VFID, E_warp, TC, TA | ⚠️ 未确认公开 | 首个4K源+双任务benchmark |
| **ROVI Dataset** | LGVI (CVPR 2024) | Language-driven OR | 5650 videos (train) + 478 videos (test) | 512×320 | Referring Segmentation GT | PSNR, SSIM, VFID, E_warp | ✅ | 首个language-driven video inpainting benchmark; 含对话式标注 |
| **CoCoCo TestSet** | CoCoCo | Text-guided editing | 300 videos | 256~512 | Instance-aware | CS, BP, TC, VQ(user), TA(user) | ⚠️ 未确认公开 | 专为text-guided设计 |
| **VideoRepainter TestSet** | VideoRepainter | Editing | DAVIS 50 + 20 in-the-wild | 576×1024 | Object/user mask | Cons, PSNR(Bg), MSE(Bg), AS | ⚠️ 部分公开 | 分离inpainting和editing评测 |

> [!IMPORTANT]
> **真正被社区广泛复用的benchmark只有两个**:  
> 1. **STTN/E2FGVI Protocol** — BR任务的事实标准, 被ProPainter/FFF-VDI/HomoGen/E2FGVI/FuseFormer等10+篇论文使用  
> 2. **ROVI Dataset** — Language-driven任务的首个也是目前唯一的公开benchmark  
>
> 其余benchmark多为各论文自建, 难以跨论文复用和对比。**OR任务至今缺少一个像STTN/E2FGVI协议那样被社区广泛接受的统一benchmark**。

---

## 7. 核心发现总结

> [!WARNING]
> **1. Metric体系不统一**: BR方法用 PSNR/SSIM/VFID/E_warp (有像素级GT); 生成式编辑方法用 TC/VQ/BP/TA (无GT), **两套体系无法直接对比**。
>
> **2. PSNR/SSIM测量区域不同**: 传统BR论文(ProPainter/FFF-VDI/HomoGen)测**全图**; MiniMax-Remover/RT-Remover在OR任务中仅测**非mask背景区域**, 这导致数值差异巨大且不可直接对比。
>
> **3. VQ指标计算方式不一致**: MiniMax-Remover用GPT-O3; RT-Remover用GPT-5; CoCoCo用人类投票。
>
> **4. 测试集不统一**: 仅DAVIS和YouTube-VOS在BR流派中被多方法共同使用; 其他测试集多为私有/自建。
>
> **5. DiffuEraser无定量metric**: 论文仅有可视化对比, 其定量数据仅出现在其他论文(FloED/MiniMax-Remover)中作为baseline测评。
>
> **6. 评测协议追溯**: STTN (ECCV 2020) 首次定义 DAVIS/YouTube-VOS + stationary mask + 432×240 标准; FuseFormer 提供具体mask文件(random_mask_stationary_w432_h240); E2FGVI 进一步固化为社区通用协议。
>
> **7. OR方向缺少统一benchmark**: MiniMax-Remover和RT-Remover各自定义DAVIS+Pexels评测，但测试集和GPT版本不同，结果不直接可比。该方向急需一个公开、标准化的OR benchmark。

---

