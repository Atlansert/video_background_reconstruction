# SVOR 本地效果检验（本地 vs 官方对照）

用官方仓库示例检验本地 SVOR（`external/svor` + `Wan2.1-VACE-1.3B` + 两阶段 remove LoRA）的效果，
与官方发布结果对照。检验日期 2026-09-17，GPU 4 号卡。

## 结论

> **2026-09-17 第二次修正（重要）**：上一版提出的"加上 `composite_source` 就是修复"**也是错的**。
> 用户目视指出：`camel_official_vs_local_composited.mp4` 的第三段（本地+修复）效果**更差**——
> 前景一闪一闪，去除帧里也有明显边缘问题。
>
> 复核确认属实，原因如下（详见"为什么 composite 在 camel 上反而更差"）：
> `maskdrop0.5` 是**缺陷掩膜**基准，50% 的帧掩膜被整帧丢弃；`composite_source` 是**逐帧 2D 融合**，
> 在掩膜被丢弃的帧会把源像素（含骆驼）贴回去 → 物体随掩膜有无而忽现忽隐 → 闪烁 + 边缘接缝。
> 正确做法是**时序并集掩膜**（与模型内部的 MUSE 策略对应），实测可消除该问题。
>
> **同时确认（回答"用的是不是官方代码/权重"）**：是。见下方"代码与权重核验"。
> 该缺陷也**不影响项目的正式产物**——真实流水线掩膜时序连贯（0 空帧、翻转率 1.2%），
> 逐帧合成对它安全（实测掩膜外相邻帧变化 3.05、边界 6.12，无闪烁）。

**三次结论的演进**（诚实记录）：

| 版本 | 结论 | 状态 |
| --- | --- | --- |
| 初版 | "本地与官方一致"，偏软是抖动假象 | ❌ 错（漏测未遮挡结构丢失） |
| 第二版 | "加 `composite_source` 即修复" | ❌ 错（逐帧合成在缺陷掩膜上引发闪烁） |
| 第三版（本版） | 见下 | ✅ 经数值 + 用户目视双重确认 |

**本版结论：**

1. **代码与权重均为官方原版**（唯二改动在未启用的代码路径上，见核验一节）。
2. **本地 SVOR 的"原始输出"（raw）行为正确**：它在掩膜存在与否的帧上都一致地移除目标，
   时序指标优于或接近官方（见下表）。
3. **问题出在后处理**：`composite_source` 的逐帧 2D 融合在**时序不连贯的掩膜**上会制造
   闪烁与边缘接缝。camel 的 `maskdrop0.5` 正是这种病态掩膜。
4. **对本项目正式产物无影响**：真实掩膜时序连贯，逐帧合成安全。

## 素材与方法要点

官方 `asset/examples/input_maskdrop0.5/camel.gif` **不是可直接推理的输入**，而是"半透明绿色掩膜
叠加在视频上"的预览图，且是 256 色调色板 GIF（R/G 各 8 级、B 仅 4 级）。因此 camel 一路的所有
像素级指标都被量化噪声主导，不能直接用于判断模型质量。

为此本检验采用两条独立证据链：

| | 素材 | 性质 |
| --- | --- | --- |
| A | `camel / maskdrop0.5` | 用户指定的示例；输入与结果均为量化 GIF |
| B | `bmx-bumps`（官方 Quick Test 素材） | 真实 raw 视频 + 真实二值掩膜，**无量化**，作为定量基准 |

## 视频对比（GitHub 可直接播放）

| 文件 | 内容 |
| --- | --- |
| `camel_maskdrop_fix_compare.mp4` | **四段对比：官方 \| 本地 raw \| 逐帧 composite（差）\| 时序并集 composite（好）** — 建议先看这个 |
| `camel_official_vs_local_composited.mp4` | 三段：官方 \| 本地 raw \| 本地+逐帧 composite（**问题版**，闪烁/边缘可见） |
| `camel_local_vs_official.mp4` | 两段：官方 \| 本地 raw（未加任何透传后处理） |
| `bmx_local_vs_official.mp4` | 官方 \| 本地 720,1280（干净素材，320×180，2x 放大） |

> 保留问题版视频作为复现记录；修复效果请看 `camel_maskdrop_fix_compare.mp4` 的第四段。

## 四联图

`figures/camel_maskdrop_fix.png`：每行 5 列 = 源输入 \| 官方 \| 本地 raw \| 逐帧 composite \| 时序并集 composite，
行为帧 0/12/24/36 —— 最能看清闪烁与边缘问题的成因和正确修法。

`figures/camel_fix_grid.png`：每行 5 列 = 源输入 \| 掩膜 \| 官方 \| 本地 raw \| 本地 + composite（逐帧）。

`figures/camel_grid.png`：每行 4 列 = 输入 \| 掩膜 \| 本地 \| 官方，行为帧 0/12/24/36。

`figures/bmx_grid.png`：每行 4 列 = raw \| 掩膜 \| 本地 720×1280 \| 官方，行为帧 0/27/54/80。

## 定量结果（关键指标）

### B) bmx-bumps 干净素材（无量化，81 帧）—— 定量基准

> **修正**：初版此表在 320×180（大幅降采样）下比较，掩盖了未遮挡结构丢失。下表为原生 854×480 重测。

移除/背景/闪烁三项在两种分辨率下都一致；但**未遮挡结构保留**必须单列，且是本次的核心发现：

| 指标（原生 854×480） | 本地 raw | 本地 + composite_source |
| --- | --- | --- |
| 未遮挡结构保留 | **83.4%** | **99.8%** |
| 掩膜内移除力度 | 90.4 | 85.6 |

（320×180 对照的移除/背景/闪烁：官方 92.2/4.98/21.6、本地 720,1280 = 91.6/4.66/23.0、本地 540,960 = 94.5/4.48/22.5。）

结论：**bmx 上也存在同样的未遮挡结构丢失**（raw 保留 83.4%），补上 `composite_source` 后恢复到 99.8%。
说明这不是 camel 独有，而是绕过透传后处理的普遍后果。

### A) camel / maskdrop0.5（480×270，48 帧）

| 项目 | 数值 |
| --- | --- |
| 官方结果 vs 量化输入：掩膜内 / 掩膜外 | 64.9 / 19.0 |
| 本地 vs 官方（同量化口径）：逐像素差 | 16.2/255 |
| 掩膜内闪烁：本地 / 官方 | **13.4** / 20.2 |

## 代码与权重核验（回答"是否用了官方仓库的代码和权重"）

**是官方原版。** 用官方 tarball（`/tmp/svor.tar.gz`，2026-05-20 版）解包后做全量递归 diff：

| 项 | 结果 |
| --- | --- |
| `diff -rq`（排除 models/__pycache__） | 仅 2 个文件有差异 |
| `predict_SVOR.py` | 新增可选参数 `--noise_file/--noise_offset`（默认 None，**未启用**） |
| `videox_fun/pipeline/pipeline_SVOR.py` | 仅 `latents is None` 分支（**未传入时不触发**） |
| 推理管线其余部分 | 与官方逐字节一致 |
| LoRA `remove_model_stage{1,2}.safetensors` | md5 与原始备份 `checkpoints/svor/` **完全一致** |
| 基座 `Wan2.1-VACE-1.3B` | 官方 HF 原始格式权重（`diffusion_pytorch_model.safetensors` 等） |

即：本次推理走的是**官方默认代码路径**（配置里 `noise_alignment: false` 且未传 `--no_file`），
权重与官方发布的一致。问题不在代码或权重，而在**后处理与素材的相互作用**。

## 为什么 composite 在 camel 上反而更差

**机制**：`maskdrop0.5` 是官方专门用来测"缺陷掩膜"的基准——**50% 的帧掩膜被整帧丢弃**：

```
掩膜有无的帧序（前 24 帧）: .MM..MMM.MM....M.M..MM.MMMM..MM.MM
（M=有掩膜  .=掩膜被丢弃）  空掩膜帧 24/48 = 50%
```

`composite_source` 是**逐帧 2D 融合**：`out = src*(1-alpha) + gen*alpha`，alpha 来自**当前帧**掩膜。
于是掩膜被丢弃的帧上 alpha≈0，源像素（含骆驼）被原样贴回；掩膜存在的帧上 alpha≈1，物体被抹掉。
→ 物体随掩膜有无而**忽现忽隐**，这正是你看到的"前景一闪一闪"；羽化 alpha 的边界则表现为"边缘明显"。

**数据（camel，480×270，48 帧）**：

| 变体 | 物体区改动@空掩膜帧 | @有掩膜帧 | 帧间闪烁 | 掩膜切换处闪烁 | 掩膜边界梯度 |
| --- | --- | --- | --- | --- | --- |
| 官方结果（参考） | 25.7 | 29.8 | 16.7 | 16.8 | 81.3 |
| 本地 raw | 37.3 | 40.4 | 15.4 | 15.6 | 63.5 |
| **本地 + 逐帧 composite（上一版"修复"）** | **16.9** | 32.9 | **24.4** | **30.9** | **104.7** |
| 本地 + 时序并集 composite | 33.1 | 36.2 | 17.1 | 17.3 | 67.9 |

读法：逐帧合成让"空掩膜帧"的物体区改动从 37.3 掉到 16.9（物体被贴回），而"有掩膜帧"仍是 32.9
（物体已消失）——两列差 16 个点，就是闪烁的量化来源；掩膜切换处的帧间变化 30.9 是 raw 的 2 倍。
**时序并集**（对该窗口内任一帧被掩膜的像素都视为要移除）使两列一致（33.1 / 36.2），闪烁回到 17.3，
接近官方 16.8。这正对应论文里的 MUSE（Mask Union for Stable Erasure）思路。

## 对本项目正式产物的影响：无

真实流水线喂给适配器的掩膜 `outputs/001_sam31_slam/masks_inpaint/`（1799 帧）时序**高度连贯**：

| 指标 | 项目真实掩膜 | camel `maskdrop0.5` |
| --- | --- | --- |
| 空掩膜帧 | **0 / 1799** | 24 / 48（50%） |
| 覆盖率 | 0.13–0.67 | 0–0.17 |
| 相邻帧掩膜翻转率 | **1.2%** | 极高（帧间整帧丢掩膜） |

因此逐帧 `composite_source` 对正式产物是安全的。实测正式产物 `background_video.mp4`（已含该后处理）：

| 区域 | 相邻帧变化 |
| --- | --- |
| 掩膜内 | 2.02 |
| 掩膜外（透传源像素） | 3.05 |
| 掩膜边界环 | 6.12 |

均为低值，**无闪烁**。结论：该项目正式交付物不受此问题影响；camel 的现象是
"逐帧合成 × 时序不连贯掩膜"的组合产物。

### 关于抖动量化（保留观察，但已降级）

GIF 抖动量化确实会污染像素级指标：把本地无损输出套用与官方相同的 Floyd-Steinberg 抖动后，
Laplacian 锐度从 1084.7 跳到 3444.7（真存 GIF 再读回 3444.7），逼近官方 5254.8。
相邻像素平均绝对差官方 18.64、输入 19.16，本地无损仅 10.26。

**但这一观察不能用来否定上述缺陷**：结构丢失与阴影在**无抖动的 bmx 素材上同样可测到**，
所以抖动只是叠加在真实缺陷之上的一层测量噪声，不是缺陷本身。
（初版报告误把它当成"全部解释"，已修正。）

### 重要方法论警告

**Laplacian 锐度指标跨分辨率不可比**：把同一张输入缩放到 480×270 / 960×544 / 1280×720，
锐度读数依次为 5750 / 1302 / 653。任何跨分辨率的锐度比较都是无效的，必须先缩放到同一尺寸。

**HF（高频能量）与抖动混叠**：评估"结构是否被删除"时，必须用**未量化**素材（如 bmx raw）或在
**同一量化口径**下比较，否则 GIF 抖动会把官方结果的高频人为抬高，得出"官方更清晰"的假象。

## 复现

camel 干净输入与掩膜已保存（`svor_camel_check/assets/`），可直接重跑推理：

```bash
cd /data/lzx/video_background_reconstruction/external/svor
CUDA_VISIBLE_DEVICES=4 conda run --no-capture-output -n svor python predict_SVOR.py \
  --input_video ../../svor_camel_check/assets/input_padded49.mp4 \
  --input_mask_video ../../svor_camel_check/assets/mask_padded49.mp4 \
  --save_dir ../../svor_camel_check/local_output/rerun \
  --model_name models/Wan2.1-VACE-1.3B \
  --lora_path models/remove_model_stage1.safetensors models/remove_model_stage2.safetensors \
  --sample_size 540,960 --video_length 49 --fps 25 \
  --num_inference_steps 20 --gpu_memory_mode model_full_load \
  --guidance_scale 6.0 --seed 43 --dilation 6 --weight_dtype bfloat16
```

bmx 干净素材复现：

```bash
cd /data/lzx/video_background_reconstruction/external/svor
CUDA_VISIBLE_DEVICES=4 conda run --no-capture-output -n svor python predict_SVOR.py \
  --input_video samples/input/bmx-bumps_raw.mp4 \
  --input_mask_video samples/input/bmx-bumps_mask.mp4 \
  --save_dir /tmp/bmx_out --sample_size 720,1280 --video_length 81 --fps 16 \
  --model_name models/Wan2.1-VACE-1.3B \
  --lora_path models/remove_model_stage1.safetensors models/remove_model_stage2.safetensors \
  --num_inference_steps 20 --gpu_memory_mode model_full_load \
  --guidance_scale 6.0 --seed 43 --dilation 6 --weight_dtype bfloat16
```

`final_report.json` 为完整指标存档；工作目录（含四组 dilation/分辨率对照与全部原始输出）
为 `svor_camel_check/`，该目录不纳入 git 跟踪，仅此处发布对照产物。