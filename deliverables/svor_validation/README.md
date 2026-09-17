# SVOR 本地效果检验（本地 vs 官方对照）

用官方仓库示例检验本地 SVOR（`external/svor` + `Wan2.1-VACE-1.3B` + 两阶段 remove LoRA）的效果，
与官方发布结果对照。检验日期 2026-09-17，GPU 4 号卡。

## 结论

> **2026-09-17 修正（重要）**：本文件初版结论"本地与官方一致"是**错误的**。
> 经用户目视反馈 + 复核重测后确认：**camel 上本地输出确实变差了**——误删了掩膜外的背景骆驼
> 并引入阴影。根因是**这次验证绕过了项目自带的 `composite_source` 透传后处理**。
> 补上该后处理后，未遮挡结构保留 **99.9%**、阴影像素归零、掩膜内移除力度不受影响。
> 详见下方"根因与修复"一节。以下保留原始记录与修正后的数据。

**修正后的结论：**

- **根因**：SVOR 会整帧重生成，未被掩膜覆盖的区域（背景骆驼、墙地纹理）会被一并重画并丢掉细节。
  项目主流程对此早有对策——`configs/vggt_slam.yaml` → `video_completion.svor.composite_source: true`，
  把掩膜外的真实像素羽化贴回。**但本次 camel 验证是直接调 `python predict_SVOR.py`，
  绕过了这一后处理**，于是暴露了未遮挡区的结构丢失与阴影。
- **证据**：本地 raw 输出在未遮挡区仅保留 77.3% 的结构能量（官方 97.2%），阴影像素占比 16.3%（官方 8.8%）；
  加上 `composite_source` 后分别为 **99.9% / 0.0%**，掩膜内移除力度保持 64.7（官方 66.6）。
- **干净素材 bmx 同样存在该现象**（本地保留 83.4% 未遮挡结构），证实这是 SVOR 的普遍行为而非 camel 特例，
  也解释了项目为何默认开启 `composite_source`。
- 此前"GIF 抖动量化造成偏软假象"的观察**部分成立但不完整**：抖动确实会污染像素指标，
  但它掩盖不了"未遮挡结构被删除"这一真实缺陷（该缺陷在无抖动的 bmx 上同样可测到）。

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
| `camel_official_vs_local_composited.mp4` | **三段对比：官方 \| 本地 raw \| 本地 + composite_source（修复）** — 建议先看这个 |
| `camel_local_vs_official.mp4` | 两段对比：官方 \| 本地 raw（**问题版**，未加透传后处理） |
| `bmx_local_vs_official.mp4` | 官方 \| 本地 720,1280（干净素材，320×180，2x 放大） |

> 注意：`camel_local_vs_official.mp4` 就是用户指出问题的那一版，保留作为"问题复现"记录；
> 修复效果请看 `camel_official_vs_local_composited.mp4` 的第三段。

## 四联图

`figures/camel_fix_grid.png`：每行 5 列 = 源输入 \| 掩膜 \| **官方** \| **本地 raw（问题版）** \|
**本地 + composite（修复）**，行为帧 0/12/24/36 —— 最能看清"背景骆驼被删 + 阴影"以及修复效果。

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

### 根因与修复（2026-09-17 复核）

**症状**（用户目视）：`camel_local_vs_official.mp4` 右侧本地结果比左侧官方**变差**——删掉了
未被掩膜覆盖的背景骆驼，并出现明显阴影。

**定位**：SVOR 是整帧重生成模型，掩膜外的像素同样被重新绘制，因此真实结构（背景骆驼、
墙地纹理）会丢失或漂移。项目对此已有对策：`configs/vggt_slam.yaml` →
`video_completion.svor.composite_source: true`（`vbr/models/svor.py::_composite_source`），
在生成后把掩膜外的**真实源像素**以羽化 alpha 贴回。

本次 camel 验证直接调 `predict_SVOR.py`，**绕过了这个后处理**，所以问题原样暴露。

**修复验证**（camel，480×270，48 帧；后处理 = 同一 `_composite_source` 羽化融合）：

| 变体 | 未遮挡结构保留 | 阴影像素占比 | 掩膜内移除力度 |
| --- | --- | --- | --- |
| 喂入的源输入 | 100.0% | 0.00% | — |
| 官方结果 | 97.2% | 8.77% | 66.60 |
| **本地 raw（即已发布的那版）** | **77.3%** | **16.27%** | 66.26 |
| **本地 + composite_source（修复）** | **99.9%** | **0.00%** | 64.70 |

结论：补上 `composite_source` 后，未遮挡结构几乎无损、阴影完全消除，且掩膜内移除力度几乎不变
（64.7 vs 66.6）。**这正是项目默认开启该选项的原因。**

**这不是 camel 特例**：用同一结构指标测官方干净素材 bmx-bumps，本地 raw 输出保留 83.4% 的
未遮挡结构（无抖动干扰时测得），同样是"掩膜外被重画"的表现。

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