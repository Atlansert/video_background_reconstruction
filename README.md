# Video Background Reconstruction

从单目室内视频恢复接近空房间的背景。当前生产路径按"纯背景"语义移除全部家具——包括冰箱、厨房柜体/橱柜、水槽等固定家具——只保留建筑结构：墙、地板、天花板、门窗和楼梯。

## 两条几何路线

3D 产物（`background_scene.glb` / `background_mesh.ply` / `structural_planes.ply` / `interactive.html`）有两条可切换的路线，二者共用同一套几何构建代码（`vbr/geometry.py`），区别只在"深度观测来自哪里"：

| | 路线 P（生产默认） | 路线 A（`geometry-prior-a` 分支，实验） |
| --- | --- | --- |
| 输入帧 | 生成视频 `background_video.mp4`（SVOR 补全后） | 原始 `001.mp4` 全部帧 |
| 前景处理 | 不再需要（前景已在视频中去除） | 当前逐帧 SAM 掩膜在 VGGT-SLAM 匹配/反投影两处排除 |
| 深度来源 | 扩散模型生成纹理经 TSDF 融合 | 真实像素经 TSDF 融合；被家具遮挡区无观测 |
| 遮挡区 | 生成模型"想象"出的表面参与多视角融合 | 由结构先验补全（重力对齐平面 + 墙线延展闭合） |
| 构建工具 | 主链 `vbr.cli run` + `tools/rebuild_geometry_from_background` | `python -m tools.rebuild_geometry_prior --out outputs/geometry_prior_a` |
| 耗时 | 数小时（含 SVOR 重生成） | ~3 分钟（复用 10–15 秒的 SLAM） |

**路线 A 的动机**：扩散模型稳定了画面观感，但其生成纹理跨帧不一致，会将误差带进多视角几何——表现为地面叠影与房间歪斜。A 版用真实帧做几何、用"墙直上直下"的结构先验填补遮挡区，不引入生成幻觉；代价是遮挡区没有真实纹理（现为对应平面的颜色外推）。路线 B（计划中，`geometry-hybrid-b` 分支）将组合二者：几何取自 A、纹理从生成视频投影。

### 结构先验补全（路线 A 的核心）

1. **墙线检测**：俯视平面 2D RANSAC 拟合竖直墙线，同一堵墙的反平行重复检测合并（`fit_wall_lines`），支持直线支撑段连续性过滤。
2. **墙角闭合**：相邻墙线在平面内求交，墙端延展到交点（限幅 `wall_corner_max_extension_room_fraction`，避免把真实短墙拉进别的房间段）；墙端同时延展到相机足迹，保证相机走过的位置始终在墙内。
3. **地板/天花板**：按重力轴分位数取楼板高度，footprint 扩到相机中心分位数包络——被家具遮挡处地板缺失时仍覆盖相机脚下。
4. **开口雕刻**：默认门/窗/楼梯提示词掩膜按深度可见性投影到墙面网格，多数票雕刻；另有 see-through 证据（墙后有观测点）雕刻门洞。
5. **装配顺序（关键）**：先对 TSDF 表面单独做碎片过滤/孔洞修补/抽稀，再拼入结构先验面。早期版本把二者一起过 `clean_mesh`，每个先验 quad 只有 2 个三角形、低于 `mesh_min_component_triangles` 被整体删除，渲染缺口达 28%；修正后降至 4.6%。

### 路线 A 的验证口径

- 用重建位姿渲染 4 个轨迹视角 + 俯视图，品红背景上图元覆盖率达 95%+（缺口 0–14%，现版 P 为 3–31%）；
- 房间高度由重力轴包络锚定 2.6 m（与主链一致）；平面重力对齐度 0.94–0.97（路线 P 为 0.92–0.96，基线 0.96–0.97）；
- 网格顶点色完整（GLB 含 COLOR_0），结构先验面数/位置进报告可核。

## Pipeline

1. 按 `sampling.segmentation_stride` 抽取语义关键帧，同时保留全部视频帧。
2. SAM 3.1 使用开放词汇提示在关键帧检测前景，并用 `preserve_prompts` 从移除掩码中扣回固定结构。
3. SAM2.1 将关键帧联合掩码传播到全部视频帧；封闭孔洞后处理用于覆盖靠垫等家具内部物体。
4. VGGT 预测相机位姿、内参、深度和置信度，在像素反投影之前排除 SAM 前景，生成彩色背景点云。
5. 从相机姿态估计重力方向，RANSAC 拟合水平/竖直平面；墙面补到地板和天花板之间，并补全地板、天花板。若单目深度的墙面不满足稳定 3D RANSAC，则使用鲁棒水平足迹生成四面贯通墙作为显式后备先验，并在报告中标记来源。
6. 掩码深度通过 TSDF 融合（失败时回退 Poisson），与结构先验网格合并。
7. 视频修复后端根据全部逐帧掩码完成背景视频，最终恢复原分辨率、帧率和音轨。
   当前默认后端为 SVOR（Stable Video Object Removal，Wan2.1-VACE-1.3B + 两阶段 remove LoRA，Apache-2.0）；
   传统传播式后端 ProPainter 仍保留为可选（`video_completion.backend: propainter`）。
   SVOR 采用 77 帧窗口分块 + 重叠淡化，并在掩膜外做光流 EMA 时序稳定（`fill_ema_alpha`/`wall_ema_alpha`）。

该流程不会把“生成了文件”当作成功：全零掩码、缺失帧、未过滤任何 VGGT 前景点或空点云都会直接终止并写入 `pipeline_status.json`。

## Environments

项目使用三个环境是为了解开上游版本冲突：

- `vbr`：主调度、OpenCV、Open3D、TSDF/Poisson、Plotly。
- `vbr-seg`：Python 3.12、PyTorch 2.10 + CUDA 12.8、SAM 3.1、SAM2 CUDA 扩展、ProPainter（含 RAFT 光流平滑）。
- `svor`：Python 3.10、PyTorch 2.7.0 + CUDA 12.6、diffusers 0.31（SVOR 修复后端，需 `external/svor` 与 `models/` 权重）。
- `vbr-slam`：Python 3.11、PyTorch 2.3 + CUDA 12.1、VGGT-SLAM、`gtsam-develop` 的 SL(4) 接口。

把 SAM 与 VGGT-SLAM 放在同一环境会让 NumPy ABI、Torch/Torchvision 和 CUDA 扩展版本互相覆盖。主程序使用子进程传递 PNG/PLY/NPZ，不会在两个 Torch 运行时之间共享进程状态。

当前安装可用以下命令检查：

```bash
conda activate vbr
python -m vbr.cli doctor --config configs/default.yaml
```

约束文件为 `requirements-seg-constraints.txt` 和 `requirements-slam-constraints.txt`。源码位于 `external/`，模型全部位于 `checkpoints/`；运行不依赖在线模型下载。

项目测试依赖位于 `requirements-dev.txt`，运行：

```bash
pytest -q
```

## Run

先只跑并检查分割：

```bash
CUDA_VISIBLE_DEVICES=7 python -m vbr.cli run \
  --config configs/default.yaml --stop-after segmentation
```

确认 `mask_overlay.mp4` 后继续完整流程；掩码输入和配置指纹一致时会复用：

```bash
CUDA_VISIBLE_DEVICES=7 python -m vbr.cli run --config configs/default.yaml
```

需要无条件重做分割时添加 `--force`。默认输入为 `video/001.mp4`。**生产配置为 `configs/vggt_slam.yaml`**（vggt_slam 后端，输出到 `outputs/001_sam31_slam/`）；`configs/default.yaml`（vggt_direct 后端）输出到 `outputs/001_sam31/`（该目录删除后会在运行时自动重建）。当前状态与常用命令见 `CURRENT_STATE.md`。

### 轨迹漫游视频（几何验收）

把重建网格按**原视频相机轨迹**渲染成视频：关键帧位姿（平移线性、旋转 slerp）插值为逐帧位姿，内参从模型空间换算到输出分辨率，EGL 离屏渲染后由 ffmpeg 编码 H.264 并带上原音轨。用于直接对照原片检查几何（墙的平直度、开口位置、地面完整性）：

```bash
# B 版（混合纹理）
python -m tools.render_trajectory_video --run-dir outputs/geometry_hybrid_b \
    --out outputs/geometry_hybrid_b/trajectory_video.mp4
# A 版（原帧+先验）
python -m tools.render_trajectory_video --run-dir outputs/geometry_prior_a \
    --out outputs/geometry_prior_a/trajectory_video.mp4
# 生产版（生成视频几何，redepth npz）
python -m tools.render_trajectory_video --run-dir outputs/001_sam31_slam \
    --npz outputs/001_sam31_slam/slam_bg/points_background.npz \
    --out outputs/001_sam31_slam/trajectory_video_redepth.mp4
```

默认输出 960×540@29.97（对齐原片）、1799 帧 ≈ 60 秒、约 11 分钟/条（EGL 渲染约 2.5–3 fps）。`--stride 2` 可跳帧减半；`--no-audio` 不带音轨。已产出的两条视频（A/B）在 `outputs/geometry_prior_a/` 与 `outputs/geometry_hybrid_b/` 中，抽帧与原片逐帧对齐（走廊/梁/柱位置重合）。

## Outputs

- `mask_overlay.mp4`：红色叠加的全帧前景掩码，用于质量验收。
- `pointcloud_background.ply`：前景过滤后的彩色背景点云。
- `background_mesh.ply`：TSDF/Poisson 表面与房间结构先验合并网格。
- `structural_planes.ply`：单独的墙、地板和天花板先验网格。
- `background_scene.glb`：通用三维场景文件。
- `interactive.html`：内嵌 Plotly、无需联网的交互场景和相机轨迹。
- `background_video.mp4`：视频修复后端（默认 SVOR）完成的视频，包含原音轨；
  `mask_overlay.mp4` / `mask_overlay_inpaint.mp4`：分割掩膜与修复掩膜的叠加预览。
- `pipeline_status.json`：各阶段状态和关键质量指标。
- `logs/`：SAM3.1、SAM2、VGGT 和视频修复后端（SVOR/ProPainter）完整日志。

## Prompt Interface

当前没有可用 GPT/VLM 端点，因此使用 `configs/default.yaml` 中经视频抽样验证的提示词。`gpt` 配置段保留给后续“每 4/8 帧拼图后自动描述前景”的提供器；更换描述来源只需要更新 `segmentation.prompts`，不会改变 SAM3.1、SAM2 或三维重建接口。

## WorldAct Reference

[WorldAct (arXiv:2605.15843)](https://arxiv.org/abs/2605.15843) 的输入是已经生成的整体 3DGS 场景，与本项目的单目视频输入不同，因此不替换当前 Pipeline。可以借鉴的部分有：从稀疏轨迹帧由多模态模型生成对象级提示、把多视角对象 mask 融合进 3D 后再投影以提高完整性，以及用视频扩散修复后将新内容按预测深度提升回 3D。当前项目已预留第一项的 GPT/VLM 接口；后两项可作为后续质量升级，现版本继续使用 SAM2 时序传播、ProPainter 和墙/地/顶平面先验。

## Known Limitation

单目视频中被家具全程遮挡的真实纹理没有观测值，任何方法都只能推断而不能精确还原。当前 `background_video.mp4` 会优先保证前景被移除和时间连续性，永久遮挡的大区域可能呈现平滑或模糊的生成纹理；`background_scene.glb` 与 `structural_planes.ply` 则用显式墙/地/顶先验保证结构完整性。若后续需要照片级未知区域，可在保留本 Pipeline 的前提下，把 WorldAct 使用的 3D mask 重投影和视频扩散修复作为可选增强阶段。

许可说明：SVOR 采用 Apache-2.0（可商用）；ProPainter 代码及模型采用其上游的**非商业许可**，
仅在选择 `backend: propainter` 时才会使用，商用前需单独核对该许可。
项目依赖的其余权重（SAM3.1/SAM2/VGGT/RAFT 等）请分别核对其上游许可。
