# 官方代码零修改运行（权威结果）

用**官方代码 + 官方权重 + 官方示例**原样跑一遍，不做任何修改、不加任何后处理。

## 运行方式

代码：官方 tarball（`svor-main`，2026-05-20 版）**原样解包**，`predict_SVOR.py`
与 `videox_fun/pipeline/pipeline_SVOR.py` 均为官方原版，未改一字。

命令：完全照 README 的 Quick Test，未加任何额外参数：

```bash
python predict_SVOR.py \
  --input_video samples/input/bmx-bumps_raw.mp4 \
  --input_mask_video samples/input/bmx-bumps_mask.mp4
```

权重（sha256 已验证与官方 LFS 一致）：

| 文件 | sha256（前 16） | 官方 LFS |
| --- | --- | --- |
| `remove_model_stage1.safetensors` | `7846f8a188aa8890` | ✓ 一致 |
| `remove_model_stage2.safetensors` | `fd52a47c4c49f5f2` | ✓ 一致 |
| `diffusion_pytorch_model.safetensors` | `c46a6f5f7d32c453` | ✓ 一致 |
| `models_t5_umt5-xxl-enc-bf16.pth` | `7cace0da2b446bbb` | ✓ 一致 |
| `Wan2.1_VAE.pth` | `38071ab59bd94681` | ✓ 一致 |

加载日志：`loaded 3D transformer's pretrained weights ... ### missing keys: 0; ### unexpected keys: 0;`
两个 LoRA 均以 weight 1.0 载入。输出 1280×720@16fps、81 帧（即官方默认 `sample_size 720,1280`）。

## 说明：官方仓库只有一组可直接推理的示例

`find` 全部资产后确认：**只有 `samples/input/bmx-bumps_{raw,mask}.mp4` 这一组带独立掩膜文件**。
其余 44 个 mp4（含 `docs/assets/videos/input_maskdrop0.5/camel.mp4` 等）都是 README 展示用的
**预览素材**，没有配套的 `*_mask.mp4`，无法直接作为官方示例推理。

因此"跑官方示例"= bmx-bumps。

## 结果：官方代码输出 ≈ 官方发布结果

与官方发布结果（`docs/assets/videos/result/bmx-bumps.mp4`，缩放到同尺寸比较，81 帧）：

| 指标 | 数值 |
| --- | --- |
| 整帧平均绝对差 | **3.92 / 255** |
| 掩膜内 | 9.92 / 255 |
| 掩膜外 | 3.78 / 255 |
| 目标移除力度：官方代码输出 | **91.6** |
| 目标移除力度：官方发布结果 | **92.2** |

两者几乎重合（移除力度差 0.6/255）。**这证明本机环境、代码、权重全部正确，推理链路无问题。**

## 产物

| 文件 | 内容 |
| --- | --- |
| `official_code_vs_released_3panel.mp4` | 三段对比（1920×360，2x）：掩膜输入 \| 官方发布结果 \| **官方代码零修改输出** |
| `official_code_output_bmx.mp4` | 官方代码的原始输出（1280×720，未做任何后处理） |
| `run.log` | 完整运行日志（含权重加载与 `missing keys: 0` 证据） |

`official_code_output_bmx.mp4` 的 sha256：
`45999224a97140fd12ed64da799800501e30406b0e0f8697fedb2b0f24d98fb1`

## 与同目录其他结论的关系

本目录其余内容（`FINAL_*.mp4`、`final_report.json`）是在**非官方素材 camel** 上做的分析——
那份素材的输入是从绿色叠加预览**反解**出来的、掩膜是从叠加位置**提取**的，**不是官方示例**，
且其中若干步骤与后处理属于自建流程。

**若只想看"官方代码本身效果如何"，以本页结果为准**：官方示例上官方代码输出与官方发布结果一致。