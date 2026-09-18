# 官方代码零修改运行（权威结果）

用**官方代码 + 官方权重 + 官方示例**原样跑一遍，不做任何修改、不加任何后处理。

## 运行方式

代码：`external/svor` 于 2026-09-18 从 GitHub 官方仓库（`xiaomi-research/svor`）**重新 clone**，
工作区干净、无任何本地修改（此前对 `predict_SVOR.py` / `pipeline_SVOR.py` 的两处补丁已随重拉消失）。

命令：完全照 README 的 Quick Test，**未加任何额外参数**（全部走官方默认）：
`video_length=81`、`sample_size=720,1280`、`num_inference_steps=20`、`guidance_scale=6.0`、
`seed=43`、`dilation=6`、`weight_dtype=bfloat16`、`gpu_memory_mode=model_full_load`。

```bash
cd external/svor
CUDA_VISIBLE_DEVICES=7 conda run --no-capture-output -n svor python predict_SVOR.py \
  --input_video samples/input/bmx-bumps_raw.mp4 \
  --input_mask_video samples/input/bmx-bumps_mask.mp4
```

运行环境：GPU 7（用户指定优先卡）。运行时间 2026-09-18。

权重加载日志：`loaded 3D transformer's pretrained weights ... ### missing keys: 0; ### unexpected keys: 0;`
两个 LoRA 均以 weight 1.0 载入。输出 1280×720@16fps、81 帧。

## 结果：官方代码输出 ≈ 官方发布结果

输入 854×480×81，与官方发布结果（`docs/assets/videos/result/bmx-bumps.mp4`）
统一缩放到 320×180 对齐比较：

| 指标 | 输入（基准） | 官方发布结果 | 本次官方代码运行 |
| --- | --- | --- | --- |
| 掩膜内改动（越高=移除越彻底） | 0.00 | 90.85 | **90.42** |
| 掩膜外改动（越低=背景保留越好） | 0.00 | 5.96 | **5.34** |
| 掩膜外结构保留（越高越好） | 1.000 | 0.686 | **0.781** |
| 掩膜内闪烁（越低越稳） | 51.89 | 19.14 | 22.01 |
| 与官方发布结果逐像素差 | — | — | **4.42 / 255** |

结论：移除力度（90.42 vs 90.85）、背景保留（5.34 vs 5.96）与官方发布结果一致；
未遮挡结构保留还略优于官方（0.781 vs 0.686）；闪烁略高但同量级。
**本机环境、代码、权重、推理链路均正确。**

## 产物

| 文件 | 内容 |
| --- | --- |
| `official_code_output_bmx.mp4` | 官方代码原始输出（1280×720，81 帧，无任何后处理） |
| `official_code_vs_released_3panel.mp4` | 三联对比（854×480 下）：掩膜输入 \| 官方发布结果 \| 本次官方代码输出 |
| `compare_grid.png` | 静态对照图（帧 0/20/40/60/80：输入 vs 输出） |
| `run.log` | 完整运行日志（含权重加载与 missing/unexpected keys: 0 证据） |

## 说明：官方仓库只有一组可直接推理的示例

`find` 全部资产后确认：**只有 `samples/input/bmx-bumps_{raw,mask}.mp4` 这一组带独立掩膜文件**。
其余 mp4（含 `docs/assets/videos/input_maskdrop0.5/camel.mp4` 等）都是 README 展示用的
**预览素材**，没有配套的 `*_mask.mp4`，无法直接作为官方示例推理。

因此"跑官方示例"= bmx-bumps。
