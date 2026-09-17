# SVOR 本地效果检验（本地 vs 官方对照）

用官方仓库示例检验本地 SVOR（`external/svor` + `Wan2.1-VACE-1.3B` + 两阶段 remove LoRA）的效果，
与官方发布结果对照。检验日期 2026-09-17，GPU 4 号卡。

## 结论

**本地 SVOR 与官方效果一致，环境与权重配置正确。**

- 在官方**干净素材**（bmx-bumps，真实 raw + 真实掩膜）上，本地与官方逐像素差仅 **3.9~4.8/255**，
  移除力度、背景保留、时序稳定性三项对齐，锐度不低于官方。
- camel 示例上曾观察到的"本地偏软"是 **GIF 抖动量化造成的测量假象**，不是模型缺陷（见下）。

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
| `camel_local_vs_official.mp4` | 左=官方结果，右=本地输出（同量化口径，480×270） |
| `bmx_local_vs_official.mp4` | 左=官方结果，右=本地输出 720,1280（干净素材，320×180，2x 放大） |

## 四联图

`figures/camel_grid.png`：每行 4 列 = 输入 \| 掩膜 \| 本地 \| 官方，行为帧 0/12/24/36。

`figures/bmx_grid.png`：每行 4 列 = raw \| 掩膜 \| 本地 720×1280 \| 官方，行为帧 0/27/54/80。

## 定量结果（关键指标）

### B) bmx-bumps 干净素材（无量化，320×180，81 帧）—— 定量基准

| 指标 | 官方结果 | 本地 720,1280 | 本地 540,960 |
| --- | --- | --- | --- |
| 掩膜内移除力度（越高越彻底） | 92.2 | 91.6 | 94.5 |
| 掩膜外背景改动（越低保留越好） | 4.98 | 4.66 | 4.48 |
| 掩膜内逐帧闪烁（越低越稳） | 21.6 | 23.0 | 22.5 |
| 全帧锐度 | 515 | **652** | **551** |
| 与官方逐像素差 | — | **3.9/255** | **4.8/255** |

### A) camel / maskdrop0.5（480×270，48 帧）

| 项目 | 数值 |
| --- | --- |
| 官方结果 vs 量化输入：掩膜内 / 掩膜外 | 64.9 / 19.0 |
| 本地 vs 官方（同量化口径）：逐像素差 | 16.2/255 |
| 掩膜内闪烁：本地 / 官方 | **13.4** / 20.2 |

### "偏软"是抖动假象的证据

| 本地输出的存储路径 | Laplacian 锐度 |
| --- | --- |
| 无损 PNG | 1084.7 |
| + 调色板量化（无抖动） | 1776.0 |
| + 调色板量化（Floyd-Steinberg 抖动） | **3454.7** |
| 真正存成 GIF 再读回 | 3444.7 |
| 官方 result GIF（参考） | 5254.8 |

即：仅把本地输出套用与官方相同的 GIF 抖动量化，锐度就从 1084.7 跳到 3454.7。
官方 GIF 的"高锐度"主体是抖动噪声——相邻像素平均绝对差官方 18.64、输入 19.16，而本地无损仅 10.26。

### 重要方法论警告

**Laplacian 锐度指标跨分辨率不可比**：把同一张输入缩放到 480×270 / 960×544 / 1280×720，
锐度读数依次为 5750 / 1302 / 653。任何跨分辨率的锐度比较都是无效的，必须先缩放到同一尺寸。

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