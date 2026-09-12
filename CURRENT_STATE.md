# 当前状态与常用命令（2026-09-11）

## 当前状态

项目处于**纯背景模式**定稿状态：前景 = 全部家具（含冰箱、厨房柜体、水槽等固定家具）与移动物；保留 = 建筑结构（墙、地板、天花板、门窗、楼梯）。

### SVOR 试验后端（`svor-trial` 分支，当前 `backend: svor`）

`video_completion.backend` 已切换为 `svor`（Wan2.1-VACE-1.3B + 两阶段 remove LoRA，Apache-2.0，比 ProPainter 非商用许可更友好）；ProPainter 成品备份在 `snapshots/videos/background_video_presvor.mp4`（md5 与切换前 deliverable 一致）：

- 代码：`vbr/models/svor.py`（`SVORAdapter`：帧/掩膜 → 4k+1 对齐分块（77 帧 + 20 帧重叠）→ 逐块 `conda run -n svor` 调 `predict_SVOR.py` → 重叠区线性交叉淡化 → **掩膜外透传合成**（`composite_source: true`，修复 SVOR 全帧重生成导致的墙位漂移/未遮挡物体闪现）→ 可选 RAFT 时序平滑（复用 `video_completion.temporal_smooth`，抑制填充区闪烁）→ h264+aac 封装）；`cli.py` 按 backend 分发，默认仍 `propainter`
- 资产：`external/svor/`（上游 tarball，gitignore）+ `external/svor/models/`（两 LoRA + Wan2.1-VACE-1.3B 原始格式基座，17.7GB，hf-mirror 下载）；独立环境 `svor`（python3.10 + torch2.7.0 + diffusers 0.31）；`svor权重/` 为用户下载的原始 LoRA（已 gitignore）
- **正式流水线切换运行（2026-09-12）**：`backend: svor` 全链重跑（分割指纹含 cli.py 触发重分割，掩膜逐像素一致），产物与 `svor_full/background_video_v2.mp4` 逐像素一致（生成确定性强）。同口径对比 ProPainter 快照：整体透传率 **0.071** vs 0.126，掩膜外保真 4.30 vs 3.84，填充深处闪烁 4.26/10.09 vs 4.00/7.79，glitch 双方 0。已重建 GLB（27,490 点）
- **已知暴露**：透传合成会如实呈现掩膜欠覆盖区域（帧 300 沙发脚碎片、帧 1103 衣物边缘锚定的深色幻觉）——根治靠补 box 种子或调大 `mask_expand_px`/`svor.dilation`（重跑视频段约 2.5–3 小时，分割缓存不受影响）
- 切换方式：`configs/vggt_slam.yaml` → `video_completion.backend`（当前 svor）；`svor_full/` 为试验产物留档
- 已知开销：每块单独 `conda run` 重新加载模型（约 2–3 分钟/块，全片 32 块约 2.5 小时）；尚未接时序平滑/残差反哺等后处理

### 最终产物（`outputs/001_sam31_slam/`）

| 产物 | 说明 |
| --- | --- |
| `background_video.mp4` | 最终背景视频（1799 帧，960×540@29.97，h264+aac）。链路（2026-09-12 起）：SVOR 32 块（77 帧+20 重叠，20 步）→ 交叉淡化 → 掩膜外透传合成 → RAFT 光流中值；ProPainter 版链路成品备份在 `snapshots/videos/background_video_presvor.mp4` |
| `background_scene.glb` / `background_mesh.ply` / `interactive.html` | 由纯背景视频重估深度重建：27,490 点（SVOR 版，2026-09-12） |
| `mask_overlay.mp4` / `mask_overlay_inpaint.mp4` | 掩膜叠加预览 |
| `masks/` `masks_inpaint/` `masks_keyframes_sam31(_onset)/` | 分割掩膜、修复掩膜、种子 |
| `slam/` `slam_bg/` | VGGT-SLAM 与背景视频重估深度两套重建源 |
| `masks_openings/` | 门窗/楼梯开口雕刻掩膜（manifest 缓存） |
| 报告 | `video_evaluation.json`（`vggt_slam_kitchen_20260909` 为最新）、`comparison_report.json`、`pipeline_status.json`、`miss_windows.json`、`geometry_redepth_report.json`、`inpainting_evaluation.json` |

版本节奏说明：历史上各轮版本（双锚点/门控/首现精修/n40/漏检/纯背景）的快照已于 2026-09-10 清理（`outputs/001_sam31` 纯 vggt 基线与 `snapshots/` 全部删除）；对比用历史指标仍在 `video_evaluation.json` / `comparison_report.json` 中。如需新的回退快照，工具会自动写入 `outputs/001_sam31_slam/snapshots/{videos,masks,geometry}/`。

### 关键指标（`vggt_slam_kitchen_20260909`）

- 前景 mask 均值 **0.3902**（最大 0.5795，无过度覆盖）
- 整体拷贝率 **0.126**（上一版 0.175）；残留均值 53.5（去除更多前景所致）
- 闪烁比 **0.926**；glitch 帧 0
- 剩余较高残留窗口：764–815 拷贝 0.311、0–82 0.167

### 环境

- 三个 conda 环境**不可合并**：`vbr`（主流程/几何）、`vbr-seg`（SAM3.1/SAM2/ProPainter/RAFT）、`vbr-slam`（VGGT-SLAM）
- GPU：7 号卡空闲可跑（0–3 被 vLLM 长期占用；4/5/6 可用）
- 权重全部离线：`checkpoints/`（sam3.1/sam2/vggt/salad/dinov2）+ `external/ProPainter/weights/`
- `python -m vbr.cli doctor --config configs/vggt_slam.yaml`：环境/权重/ffmpeg 自检（13 项）

### 已知遗留

1. **764–815 帧段沙发/漏检对象**：文本 prompt 对该沙发免疫，自动 box 只能部分命中。人工修复：在 `configs/vggt_slam.yaml` → `segmentation.miss_refinement.box_seeds` 填入归一化框，例如
   `[{"start": 764, "end": 815, "box": [0.2, 0.3, 0.95, 0.9], "prompt": "sofa"}]`，
   然后运行 `python -m tools.refine_misses --config configs/vggt_slam.yaml`（约 50 分钟）。
2. **子图尺度漂移**（0.23–1.0）未归一化，可能造成 GLB 地面叠影；已记录在 `geometry_redepth_report.json`。
3. ProPainter 及其权重为**非商用许可**，商用前需单独核对。

## 常用命令（默认在 `vbr` 环境、GPU7）

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate vbr
export CUDA_VISIBLE_DEVICES=7   # 子进程会按配置自行设置，此处仅为约定

# 全链重跑（改过 mask/prompt/代码后必须 --force；约 3–4 小时）
python -m vbr.cli run --config configs/vggt_slam.yaml --force

# 只重跑漏检窗口精修（缓存分割，支持 box 种子）并可继续出视频
python -m tools.refine_misses --config configs/vggt_slam.yaml            # 完整（含视频）
python -m tools.refine_misses --config configs/vggt_slam.yaml --no-video # 只到掩膜

# 光流对齐时序平滑（对已有 background_video.mp4 单独应用；--keep-version 保留平滑前快照）
python -m tools.apply_temporal_smooth --output-dir outputs/001_sam31_slam \
    --config configs/vggt_slam.yaml --keep-version

# 两遍残差反哺（把掩膜内透传区补进 mask 后重跑 ProPainter）
python -m tools.refine_inpainting --output-dir outputs/001_sam31_slam \
    --config configs/vggt_slam.yaml --max-rounds 3

# 背景视频重估深度 → 重建 GLB（缓存先清后写，可直接重跑）
python -m tools.rebuild_geometry_from_background \
    --output-dir outputs/001_sam31_slam --config configs/vggt_slam.yaml

# 评估（残留/闪烁/分窗口拷贝/glitch）
python -m tools.evaluate_inpainting --output-dir outputs/001_sam31_slam \
    --windows-json outputs/001_sam31_slam/miss_windows.json

# ProPainter 参数搜索（片段级网格，如 760–990）
python -m tools.search_propainter_params --output-dir outputs/001_sam31_slam \
    --start 760 --end 990 --config configs/vggt_slam.yaml

# 无 RAFT 的朴素 3 帧中值（兼容旧工具，新流程请用 apply_temporal_smooth）
python -m tools.post_temporal_smooth --output-dir outputs/001_sam31_slam

# 环境/权重/ffmpeg 自检
python -m vbr.cli doctor --config configs/vggt_slam.yaml

# 单元测试（51 项）
python -m pytest tests/ -q
```

### 缓存与重跑行为

- 帧缓存：`outputs/<run>/frames_all/` 由 `outputs/<run>/video_source.json` marker 校验（视频尺寸/mtime/形状变化即重抽，**marker 不可放进 frames 目录**——ProPainter 会无差别读取目录内所有文件）。
- 分割缓存：`segmentation_manifest.json` 指纹 = 视频 stat + 分割配置 + 相关源码哈希；改任何一项都会触发全量重分割（预期行为）。
- 掩膜链：`masks/`（重建用）→ `masks_inpaint/`（修复用，close 9 + 时序并集 ±2 + 扩张 1，ProPainter 内部 dilation 8）。
- `masks_onset_refinement` / `masks_miss_refinement` 等中间目录均由流程重建，可随时删除。