# Video Background Reconstruction 项目交接文档

更新日期：2026-09-05

项目根目录：`/data/lzx/video_background_reconstruction`

注意：该路径与历史记录中的 `/home/lyx/lzx/video_background_reconstruction` 是同一目录（当前环境下 inode 相同）。代码通过 `vbr/cli.py` 自动按源码位置解析项目根目录，不要把输出路径硬编码成其他路径。

## 1. 项目目标

输入一个室内单目视频，尽量恢复接近空房间的背景，不保留可移动前景物体。

目标输出有两类：

1. 去除可移动家具、杂物后的背景视频，保留原始帧率、分辨率和音轨。
2. 可交互的三维背景场景，包括背景点云、TSDF/Poisson 表面，以及墙、地板、天花板等结构先验。

用户要求保留的固定物体包括洗手池、厨房柜体、楼梯、门窗、冰箱等；当前通过 SAM3.1 的 `preserve_prompts` 从待移除 mask 中扣除这些区域。

原始输入：`video/001.mp4`，960×540，约 60 秒，1799 帧，29.97 FPS。

## 2. 当前已经完成什么

当前生产 Pipeline 已完成实现，并且已经对 `video/001.mp4` 跑通一次完整流程。

已完成的功能：

- 全视频帧抽取和稀疏关键帧抽取。
- SAM3.1 文字提示分割关键帧中的可移动前景。
- SAM3.1 preserve prompts：保护洗手池、柜体、楼梯、门窗、冰箱。
- SAM2.1 分段时序传播到全部视频帧。
- 封闭 mask 孔洞填充。
- 使用 mask 在 VGGT 深度反投影前过滤前景点。
- 生成彩色背景点云。
- 使用相机姿态估计重力方向。
- RANSAC 平面拟合。
- 地板、天花板、墙壁结构先验补全。
- TSDF 网格融合，失败时支持 Poisson 回退。
- GLB 导出。
- 内嵌 Plotly 的交互式 HTML 导出。
- ProPainter 视频背景修复。
- ProPainter 失败时的时间帧单应性 + Telea 回退方案。
- `doctor` 环境检查。
- 6 个核心单元测试。
- GPT/VLM 动态 prompt provider 接口预留。

最近一次运行结果位于 `outputs/001_sam31/`，状态文件显示：

```text
state: complete
1799/1799 帧有 mask
平均前景覆盖率：0.3393
VGGT 拒绝前景点：1,621,944
输出背景点云：19,389 点
TSDF 表面顶点：5,780
合并网格顶点：5,788
合并网格三角形：10,308
视频修复：ProPainter
```

主要产物：

- `outputs/001_sam31/background_video.mp4`
- `outputs/001_sam31/interactive.html`
- `outputs/001_sam31/background_scene.glb`
- `outputs/001_sam31/background_mesh.ply`
- `outputs/001_sam31/structural_planes.ply`
- `outputs/001_sam31/pointcloud_background.ply`
- `outputs/001_sam31/pipeline_status.json`

## 3. 当前正在做什么

当前没有正在运行的训练、推理或长时间后台进程；项目处于“已跑通、等待下一步质量升级或完整 VGGT-SLAM 接入”的交接状态。

当前配置仍然使用：

```yaml
slam:
  backend: vggt_direct
```

这意味着当前并不是完整的 VGGT-SLAM，而是直接调用 VGGT 的相机、深度和置信度预测，再进行 mask-aware 点云反投影。完整 VGGT-SLAM 尚未接入主 Pipeline。

## 4. 下一步应该做什么

建议按以下顺序推进：

### 第一优先级：先验证当前结果

1. 查看 `outputs/001_sam31/mask_overlay.mp4`，确认红色区域是否真的对应可移动前景。
2. 查看 `outputs/001_sam31/background_video.mp4`，重点观察前景边缘、遮挡区域和时间闪烁。
3. 打开 `outputs/001_sam31/interactive.html`，确认点云、结构网格和相机轨迹是否合理。
4. 检查 `pipeline_status.json`、`geometry_report.json` 和 `slam/points_background.json`。

### 第二优先级：解决完整 VGGT-SLAM 接入问题

需要新增一个真正的 `vggt_slam` backend，不能简单把 `vggt_direct.py` 改名。应完成：

1. 将当前逐帧 mask 对齐到完整 VGGT-SLAM 使用的图像序列。
2. 确定使用全部 1799 帧还是稀疏帧；完整 SLAM 的 submap/loop closure 参数需要重新调节。
3. 调用 `external/VGGT-SLAM/main.py` 或直接封装 `vggt_slam.solver.Solver`。
4. 让完整 SLAM 输出最终优化后的相机外参、深度/子地图点云和可追踪的帧索引。
5. 设计 foreground mask 如何进入 SLAM：至少要在点云融合和 TSDF 阶段过滤；若希望前景不干扰姿态估计，还要把 mask 传入 VGGT/子地图匹配过程。
6. 处理完整 SLAM 输出与当前 `NPZ` 接口的兼容。
7. 对比 `vggt_direct` 和完整 `vggt_slam` 的轨迹漂移、墙面拟合和点云质量。

### 第三优先级：提高背景质量

- 完善墙面、门窗、楼梯开口的结构建模。
- 增加多视角 3D mask 融合后再重投影的流程。
- 将 VLM 自动 prompt 发现接入实际 provider。
- 对 ProPainter 输出做前景残留和闪烁评估。
- 增加缓存版本号或代码 hash，避免代码变化时误复用旧缓存。
- 处理 VGGT 相对尺度，必要时加入房间尺度标定接口。

## 5. 项目目录结构

```text
video_background_reconstruction/
├── video/
│   └── 001.mp4                         # 输入视频
├── configs/
│   └── default.yaml                    # 主配置
├── vbr/                                # 主项目 Python 包
│   ├── cli.py                          # Pipeline 总调度和 CLI
│   ├── config.py                       # YAML 配置读取
│   ├── video.py                        # 视频、帧、mask 和回退修复
│   ├── sam31_keyframes.py              # SAM3.1 关键帧分割
│   ├── sam2_propagate.py               # SAM2 时序传播
│   ├── vggt_direct.py                  # 直接 VGGT 重建
│   ├── geometry.py                     # 平面、TSDF、Poisson、网格
│   ├── interactive.py                  # Plotly 交互式 HTML
│   ├── prompts.py                      # GPT/VLM prompt 接口
│   ├── sfm.py                          # 旧/预留 SfM 相关代码
│   └── models/
│       ├── segmentation.py             # vbr-seg 子进程适配器
│       ├── slam.py                     # vbr-slam 子进程适配器
│       └── inpainting.py               # ProPainter + FFmpeg 适配器
├── external/
│   ├── VGGT-SLAM/                      # VGGT-SLAM 上游源码
│   ├── ProPainter/                     # ProPainter 上游源码
│   └── sam2/                           # SAM2 上游源码
├── checkpoints/
│   ├── sam3.1/sam3.1_multiplex.pt
│   ├── sam2/sam2.1_hiera_large.pt
│   ├── vggt/model.pt
│   └── propainter/*.pth
├── outputs/001_sam31/                  # 001.mp4 最近一次完整运行产物
├── tests/test_core.py                  # 核心单元测试
├── requirements*.txt                   # 环境约束
├── tools/                              # 下载/辅助脚本
├── source/                             # 外部资源暂存区，通常不参与运行
├── offline_bundle/                     # 离线安装/资源包
├── README.md                           # 面向使用者的项目说明
└── HANDOFF.md                          # 本交接文档
```

## 6. 关键文件及其作用

### 主调度

`vbr/cli.py`

- `run()`：按帧、分割、重建、几何、视频顺序执行。
- `_fingerprint()` / `_cached()`：根据视频状态和配置复用 mask 缓存。
- `doctor()`：检查权重、环境、CUDA、GTSAM 和 FFmpeg。
- CLI 支持 `run`、`doctor`、`--stop-after` 和 `--force`。

### 分割

`vbr/sam31_keyframes.py`

- 对每个文字 prompt 调用 SAM3.1。
- remove prompts 做并集。
- preserve prompts 单独分割，并从 remove mask 中扣除。
- 输出关键帧 mask 和 `sam31_report.json`。

`vbr/sam2_propagate.py`

- 读取 SAM3.1 关键帧 seed。
- 按相邻关键帧区间独立运行 SAM2。
- 清理小连通区域。
- 输出全视频逐帧 mask。

`vbr/models/segmentation.py`

- 从 `vbr` 环境启动 `vbr-seg` 子进程。
- 依次执行 SAM3.1 和 SAM2。
- 用日志文件保存完整 stdout/stderr。

### 三维重建

`vbr/vggt_direct.py`

- 在 `vbr-slam` 环境加载 VGGT。
- 均匀选取最多 32 帧。
- 预测 pose、intrinsics、depth、confidence。
- 将原视频 mask 映射到 518×518 模型空间。
- 在反投影前过滤前景像素。
- 输出 PLY、NPZ 和 JSON 报告。

`vbr/models/slam.py`

- 当前只支持 `backend: vggt_direct`。
- 通过 `conda run -n vbr-slam python -m vbr.vggt_direct` 调用。
- 这是未来接入完整 VGGT-SLAM 的主要替换点。

### 几何和网格

`vbr/geometry.py`

- 根据相机外参估计重力方向。
- 对点云进行 RANSAC 平面拟合。
- 用高度分位数拟合地板和天花板。
- 用竖直平面或 footprint fallback 生成墙。
- 用 mask-aware depth 进行 TSDF。
- TSDF 失败时回退 Poisson。
- 合并结构网格和表面网格。
- 导出 PLY、GLB 和 `geometry_report.json`。

### 视频修复

`vbr/models/inpainting.py`

- 通过 `vbr-seg` 启动 ProPainter。
- 处理尺寸当前为 960×536。
- 最终使用系统 `/usr/bin/ffmpeg` 的 `libx264` 编码。
- 从原视频复制音轨并恢复到 960×540。

`vbr/video.py`

- 视频元信息和帧抽取。
- mask 统计和红色 overlay。
- 封闭孔洞填充。
- ProPainter 失败后的 ORB/单应性/Telea 回退。

## 7. 已经做出的重要设计决策

### 分割、重建、视频修复拆成三个环境

当前使用：

- `vbr`：Python 3.10，调度、OpenCV、Open3D、TSDF/Poisson、Plotly。
- `vbr-seg`：Python 3.12，PyTorch 2.10 + CUDA 12.8，SAM3.1、SAM2、ProPainter。
- `vbr-slam`：Python 3.11，PyTorch 2.3 + CUDA 12.1，VGGT/VGGT-SLAM、GTSAM。

原因是不同上游项目的 PyTorch、Torchvision、CUDA 扩展、NumPy ABI 和 GTSAM 版本存在冲突。主程序通过子进程和文件接口交互，避免在同一个 Python 进程内加载多个不兼容 Torch 运行时。

### 当前先用直接 VGGT，不是假装已经接入完整 VGGT-SLAM

`external/VGGT-SLAM` 已下载并安装，但当前集成使用其中的 VGGT 模型做一次性预测。原因是原始完整入口的输出接口、submap、loop closure、SL(4) 优化结果尚未适配当前的“逐帧 mask → 前景过滤 → TSDF”数据流。

完整 VGGT-SLAM 不是不能用，而是需要专门适配，特别是：

- 如何让 foreground mask 参与姿态/匹配过程；
- 如何处理 1799 帧视频而不是示例目录；
- 如何导出优化后的每帧相机和深度；
- 如何保证 frame id 和 mask 严格对应；
- 如何将 submap/loop closure 的点云合并到现有 NPZ/TSDF 接口。

### 先做 mask，再做点云过滤

前景不是在点云生成后凭颜色删除，而是在深度反投影之前直接丢弃 mask 区域。这可以减少前景深度污染，但依赖 mask 质量。

### 结构先验必须有 fallback

当前单目深度中竖直墙面可能不满足 RANSAC 的稳定性，所以 `footprint_wall_fallback: true`。当检测不到墙时，系统会生成四面基于点云水平足迹的贯通墙，确保场景不会缺墙。

### ProPainter 优先，传统时序方法回退

视频质量优先使用 ProPainter；为了防止模型、显存或编码失败导致整条 Pipeline 不可用，保留 ORB + 单应性 + Telea 的可运行回退。

## 8. 当前已知问题

1. 当前不是完整 VGGT-SLAM。没有接入完整的 submap、loop closure 和 GTSAM 位姿图优化。
2. 单目视频中被家具全程遮挡的真实纹理不可观测，ProPainter 只能生成合理纹理，不能保证是真实纹理。
3. 当前运行的墙壁来源是 `robust_footprint_fallback`，而不是稳定的 RANSAC 竖直墙。
4. fallback 生成的是四面完整矩形墙，不会自动开门、窗、楼梯和柜体洞口。
5. VGGT 坐标是相对尺度，当前 `room_height: 0.6442` 不是实际米制高度。
6. `segmentation_stride: 30` 对快速运动物体可能过稀；提高关键帧密度会显著增加 SAM3.1 时间和显存占用。
7. SAM2 当前按独立区间传播，可减少长序列漂移，但区间边界可能产生 mask 变化。
8. 当前 mask 平均覆盖率约 33.93%，提示词较宽泛；应人工检查 `mask_overlay.mp4`，不能只看“Pipeline complete”。
9. `gpt.enabled` 当前为 `false`，VLM 自动发现前景只预留了接口，没有接入实际 provider。
10. cache fingerprint 包含视频文件状态和配置，但不包含源码版本 hash；改代码后可能误复用旧 mask 缓存，调试时应使用 `--force`。
11. 最近一次状态中的 `filled_enclosed_hole_pixels: 0` 表示最后一次运行时 mask 已经没有新孔洞，不代表算法没有实现孔洞填充。
12. `interactive.html` 是点云/网格浏览器，不是带完整语义层级、碰撞和导航的引擎场景。

## 9. 已经尝试过但失败或效果不理想的方法

### 直接对整段视频使用单一 SAM2 状态

早期尝试让 SAM2 状态贯穿整段视频，长视频中 mask 会逐渐漂移、消失或覆盖率明显下降。现已改成相邻 SAM3.1 关键帧之间的独立传播区间。

### 只用 SAM3.1 关键帧 mask，不传播到所有帧

不能满足视频输出和点云输入要求。当前已改为 SAM3.1 关键帧检测 + SAM2 全帧传播。

### 依赖 Conda 环境中的 FFmpeg 编码 H.264

当前 Conda 中 FFmpeg 不一定带 `libx264`，导致视频最终编码失败。已改为优先检查并使用 `/usr/bin/ffmpeg`，`doctor` 会检查 `libx264`。

### 只用 RANSAC 拟合墙壁

当前点云中 RANSAC 找到的主要是水平面，稳定竖直墙不足。因此增加了基于水平足迹的四墙 fallback。

### 仅用 Poisson 生成完整房间网格

Poisson 对稀疏、遮挡严重、法线不稳定的室内点云容易出现封口、漂浮面或孔洞。当前优先 TSDF，并将结构先验单独合并；Poisson 仅作为 TSDF 失败回退。

### 仅靠背景点云恢复被遮挡纹理

对于整个视频都没有被观察到的墙面纹理，点云无法恢复真实颜色。当前接受这一物理限制，使用 ProPainter 补全视频、墙/地/顶先验保证几何完整性。

## 10. 当前 TODO

### 必做

- [ ] 在保留 `vggt_direct` 的情况下，实现可选的 `vggt_slam` backend。
- [ ] 设计完整 VGGT-SLAM 的 mask-aware 接口。
- [ ] 输出优化后每帧 extrinsics、frame ids、submap/loop closure 报告。
- [ ] 对比 direct VGGT 与完整 VGGT-SLAM 的轨迹和网格质量。
- [ ] 增加完整运行的回归测试或小视频 smoke test。

### 质量提升

- [ ] 调整 SAM3.1 prompt，减少宽泛的 `furniture` / `decoration` 误删。
- [ ] 对 mask 做边界评估、时序一致性评估和固定物体保护评估。
- [ ] 墙面加入门窗、楼梯、固定柜体的开口/遮挡建模。
- [ ] 将多视角 3D mask 融合和重投影作为可选增强阶段。
- [ ] 接入 GPT/VLM provider，并限制其输出为稳定的短名词 prompt。
- [ ] 增加真实尺度输入接口，例如已知门高、房间尺寸或相机高度。
- [ ] 给缓存添加源码/模型版本号。

### 工程维护

- [ ] 将项目根目录正式纳入 Git；当前根目录没有 `.git`。
- [ ] 固定外部依赖 commit，并记录当前被修改的上游文件。
- [ ] 统一输出报告中的绝对路径和相对路径策略。
- [ ] 清理不需要提交的 `__pycache__`、`.pytest_cache` 和大型临时文件清单。

## 11. 测试方法

### 单元测试

在 `vbr` 环境：

```bash
conda activate vbr
cd /data/lzx/video_background_reconstruction
pytest -q
```

当前测试共 6 项，覆盖：

- 相机重力方向；
- 四墙 fallback；
- mask 映射到 VGGT 模型空间；
- 封闭 mask 孔洞填充；
- 禁用 GPT provider 时使用静态 prompts；
- FFmpeg H.264 可用性。

### 环境检查

```bash
conda activate vbr
CUDA_VISIBLE_DEVICES=7 python -m vbr.cli doctor \
  --config configs/default.yaml
```

### 分阶段运行

先只跑分割：

```bash
CUDA_VISIBLE_DEVICES=7 python -m vbr.cli run \
  --config configs/default.yaml \
  --stop-after segmentation
```

确认 `mask_overlay.mp4` 后运行重建：

```bash
CUDA_VISIBLE_DEVICES=7 python -m vbr.cli run \
  --config configs/default.yaml \
  --stop-after reconstruction
```

最后完整运行：

```bash
CUDA_VISIBLE_DEVICES=7 python -m vbr.cli run \
  --config configs/default.yaml
```

如果修改了 prompts、阈值、传播逻辑或代码：

```bash
CUDA_VISIBLE_DEVICES=7 python -m vbr.cli run \
  --config configs/default.yaml --force
```

## 12. 环境配置

### `vbr`

- Python 3.10.20。
- 主调度、OpenCV、PyYAML、SciPy、Open3D、trimesh、Plotly。
- 运行 CLI、几何、交互式 HTML、测试。

### `vbr-seg`

- Python 3.12.14。
- PyTorch 2.10 + CUDA 12.8。
- SAM3.1、SAM2、SAM2 CUDA 扩展、ProPainter。
- 使用 GPU 7。

### `vbr-slam`

- Python 3.11.16。
- PyTorch 2.3 + CUDA 12.1。
- VGGT、VGGT-SLAM、Open3D、GTSAM SL(4)。
- 使用 GPU 7。

### 模型权重

- `checkpoints/sam3.1/sam3.1_multiplex.pt`，约 3.5 GB。
- `checkpoints/sam2/sam2.1_hiera_large.pt`，约 898 MB。
- `checkpoints/vggt/model.pt`，约 5.0 GB。
- `checkpoints/propainter/ProPainter.pth`。
- `checkpoints/propainter/recurrent_flow_completion.pth`。
- `checkpoints/propainter/raft-things.pth`。
- 完整 VGGT-SLAM 的 loop closure 相关权重还包括 `checkpoints/dinov2/` 和 `checkpoints/salad/`；当前 direct VGGT 路径不依赖它们。

### 为什么不能合并环境

SAM2 CUDA 扩展、SAM3.1、ProPainter、VGGT 和 GTSAM 对 Python、PyTorch、CUDA、NumPy ABI 的要求不同。不要在同一个解释器里强行 import 全部组件；跨环境接口使用 PNG/JPG、PLY、NPZ 和子进程。

## 13. 重要命令

```bash
# 进入项目
cd /data/lzx/video_background_reconstruction

# 检查环境
conda activate vbr
CUDA_VISIBLE_DEVICES=7 python -m vbr.cli doctor --config configs/default.yaml

# 单元测试
pytest -q

# 分割阶段
CUDA_VISIBLE_DEVICES=7 python -m vbr.cli run \
  --config configs/default.yaml --stop-after segmentation

# 重建阶段
CUDA_VISIBLE_DEVICES=7 python -m vbr.cli run \
  --config configs/default.yaml --stop-after reconstruction

# 完整 Pipeline
CUDA_VISIBLE_DEVICES=7 python -m vbr.cli run \
  --config configs/default.yaml

# 强制重跑
CUDA_VISIBLE_DEVICES=7 python -m vbr.cli run \
  --config configs/default.yaml --force

# 检查主要输出
ls -lh outputs/001_sam31/
cat outputs/001_sam31/pipeline_status.json
cat outputs/001_sam31/geometry_report.json
```

## 14. Git 当前状态

项目根目录当前不是 Git 仓库：

```text
git status --short --branch
fatal: not a git repository
```

因此主项目的源码、配置和输出目前没有根级 Git 提交历史。不要把这个状态误认为“工作区干净”；它只是没有根级版本控制。

外部仓库状态：

```text
external/VGGT-SLAM: main...origin/main
  M main.py
  M vggt_slam/loop_closure.py

external/ProPainter: main...origin/main
  M inference_propainter.py
```

这些外部修改是此前为当前环境/运行方式做的适配，不能未经检查直接 `git reset --hard` 或覆盖。`external/sam2` 当前目录不是一个可正常读取状态的 Git 工作树。

## 15. 不能破坏的约束及重要事项

1. 不要删除或移动以下模型权重，除非先确认有备份：SAM3.1、SAM2、VGGT、ProPainter、DINOv2、SALAD。
2. 不要把 `vbr-seg` 和 `vbr-slam` 合并成同一个环境；这很可能破坏 SAM2 CUDA 扩展或 VGGT/GTSAM 依赖。
3. 所有 GPU 推理默认使用 GPU 7，通过 `CUDA_VISIBLE_DEVICES=7` 控制。
4. 任何分割代码修改后都必须检查 `mask_overlay.mp4`，不能只依据命令成功退出。
5. 不要把 preserve prompts 删除；它们是保留洗手池、柜体、楼梯、门窗、冰箱的主要机制。
6. 不要默认把所有 `furniture`、`decoration` 等宽泛 prompt 都当作正确；它们可能误删固定结构。
7. 修改 mask 语义后，必须重新运行 `--force`，否则 CLI 可能复用旧 mask 缓存。
8. 不要删除 `outputs/001_sam31/slam/points_background.npz`；TSDF 阶段需要其中的深度、相机内外参和模型空间 mask。
9. 不要把 `background_scene.glb` 当作真实米制模型；当前 VGGT 输出只有相对尺度。
10. 不要声称当前已经接入完整 VGGT-SLAM；真实 backend 名称仍是 `vggt_direct_masked`。
11. 不要删除 `footprint_wall_fallback`，除非已经有稳定的竖直墙检测替代方案。
12. ProPainter 上游代码和模型有非商业许可限制；商业使用前需要单独确认许可。
13. 大型目录和输出文件合计约 15 GB，清理前要明确保留模型、输出和源码的范围。
14. 对大型输出进行覆盖或删除前，先确认是否需要作为回归基线；尤其是 `outputs/001_sam31/`。
15. 完整 VGGT-SLAM 接入必须保留 `vggt_direct` 作为 fallback backend，不能一次性替换掉当前可运行路径。

