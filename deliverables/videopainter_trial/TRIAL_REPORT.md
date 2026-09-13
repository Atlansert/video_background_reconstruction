# VideoPainter 试验报告（videopainter-trial 分支，2026-09-13）

## 结论（TL;DR）

VideoPainter（TencentARC, SIGGRAPH 2025）已完整接入为第三种 backend 并全片跑通（52.6 分钟生成 +
minterpolate 上采样）。与当前 SVOR 交付物 A/B 后的结论：

- **中小掩膜场景（书架/绿植/餐桌/零散家具）：VideoPainter 明显更优** —— 填充干净、无 ghost
  家具、无边缘框；`replace_gt` 在潜空间锁定掩膜外为真实像素，墙地零漂移（未掩膜 MAE 8-15）。
- **大掩膜场景（客厅整景 ~44% 覆盖、厨房台面近景）：SVOR 明显更优** —— VideoPainter 需要凭空
  生成整屋内容，输出糊状（帧 950/1250，见 `vp_full_*.jpg` vs `vp_svor_*.jpg`）；SVOR（Wan2.1-VACE
  remove-LoRA）对大掩膜的整屋重建强得多。
- 时序一致性（warping error，越低越好）：SVOR 1.549 vs VideoPainter 1.679（SVOR 略优）。
- 透传率（残留前景拷贝，越低越好）：SVOR 0.071 vs VideoPainter **0.047**（VP 去除更彻底，部分
  得益于大掩膜糊化覆盖）。
- glitch 双方均为 0；闪烁比 VP 0.58 显著低，但含 minterpolate 运动补偿插帧的时序混合效应，需打折解读。

**综合判断：作为本任务（大掩膜为主的整屋清空）的替代后端，VideoPainter 当前形态不如 SVOR；
但其潜空间透传锁死与中小掩膜填充质量值得保留在分支上，可作为"小物件精修"或融合候选。**

## 关键接入事实（后续复用必读）

1. **仓库自带 diffusers fork 必须独立环境**（`pip install -e external/videopainter/diffusers`，
   env 前缀 `.conda-vp`，torch 2.4.0+cu121 / transformers 4.42.2）；与 `svor`/`vbr` 环境互不兼容。
2. **官方 infer/inpaint.py 的无 FLUX 路径有 bug**（`first_frame_gt` 时 `gt_video_first_frame`
   未定义），FLUX.1-Fill 首帧修补是其流程的必要环节。本试验用 OpenCV TELEA 首帧修补等价替代
   （`--first_frame_fill telea`）。
3. **黑洞根因与修复（最重要的 off-paper 改动）**：官方管线把打黑视频作为 `replace_gt` 的 GT 源，
   VAE 编码时黑色扩散到掩膜周边潜向量，在填充边界保留为黑环（近黑占比 0.014+）。改为
   TELEA 预填充视频后黑环彻底消除（0.0008）；branch 条件输入经管线内部重新打黑，与官方训练
   分布完全一致（`--fill_mode telea`，已设为默认）。
4. **对齐**：单窗口 0 偏移；多窗（stride=49）有 +1 偏移；全片 stride=29 隐空间重叠平均后整体
   0 偏移（逐段校验通过）。输出 fps=source/stride（9.99）保持时长，再 minterpolate 3x 回
   29.97fps；VAE 潜变量数学使模型帧少 ~2.7%，最终 `-frames:v 1799` + tpad 克隆补齐契约帧数。
5. **成本**：单次模型加载（vs SVOR 每块重载 2-3 分钟），全片 20 窗口 41 分钟生成（30 步/H200）；
   TELEA 预填充 ~6 分钟 CPU；minterpolate+编码 ~5 分钟。

## 指标对照表

| 指标 | SVOR v5（现交付） | VideoPainter（本试验） |
| --- | --- | --- |
| warping_error_mean | **1.549** | 1.679 |
| warping_error_p95 | **2.939** | 3.357 |
| warping inside/outside | **1.612 / 1.490** | 1.796 / 1.528 |
| 透传率（copy_through overall） | 0.071 | **0.047** |
| glitch | 0 | 0 |
| flicker_ratio_mean | ~0.99 | 0.58（含插帧混合效应） |
| 生成耗时（全片） | ~2.5h（32 块逐块重载） | **41min**（单次加载 20 窗） |

## 文件

- 代码：`vbr/videopainter_driver.py`（vp env 驱动）、`vbr/models/videopainter.py`（适配器）、
  `vbr/cli.py`（backend 分发）、`configs/vggt_slam.yaml`（`videopainter:` 块）
- 权重：`external/videopainter/ckpt/`（CogVideoX-5b-I2V 21G + branch 712M + ID LoRA 528M，
  hf-mirror 下载脚本 `download_weights.py`）
- 运行产物：`outputs/001_sam31_slam/videopainter/`（raw 生成 + driver 日志/报告）、
  `background_video.mp4`（1799 帧 @29.97）、`inpainting_evaluation.json`、
  `flow_metrics_videopainter.json`
- 对比图：本目录 `vp_full_950/1250.jpg`（VP）vs `vp_svor_950/1250.jpg`（SVOR）、
  `s3_15/30.jpg`（VP 中小掩膜良好样本）
- SVOR 基线：主树 `flow_metrics_baseline.json`（warping error）

## 已知限制与后续可试

- 大掩膜糊化为模型能力边界（CogVideoX 通用修补 vs Wan-VACE remove 专用训练）；未走 FLUX
  首帧（GPT-4o 依赖），对首帧质量敏感的场景可接 FLUX.1-Fill-dev。
- 仅训练分布 8fps：`down_sample_stride: 3`（≈10fps）是质量/成本折中；stride=1（30fps）会
  显著劣化（黑块/糊）。
- 评估口径注意：minterpolate 插帧让闪烁/时序指标偏乐观；与 SVOR 对比时以视觉段落抽查为准。
