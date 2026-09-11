# 当前状态与常用命令（2026-09-10）

## 当前状态

项目处于**纯背景模式**定稿状态：前景 = 全部家具（含冰箱、厨房柜体、水槽等固定家具）与移动物；保留 = 建筑结构（墙、地板、天花板、门窗、楼梯）。

### 最终产物（`outputs/001_sam31_slam/`）

| 产物 | 说明 |
| --- | --- |
| `background_video.mp4` | 最终背景视频（1799 帧，960×540@29.97，h264+aac）。链路：V1.c 漏检精修（文本/box/证据种子）→ ProPainter n=40 → 光流对齐 3 帧中值 → 两遍残差反哺 ×3 |
| `background_scene.glb` / `background_mesh.ply` / `interactive.html` | 由纯背景视频重估深度重建：29,120 点、`plan_ransac` 8 面真墙、TSDF 6,997 顶点 |
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