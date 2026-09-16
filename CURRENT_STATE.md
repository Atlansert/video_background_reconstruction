# 当前状态与常用命令（2026-09-11）

## 当前状态

项目处于**纯背景模式**定稿状态：前景 = 全部家具（含冰箱、厨房柜体、水槽等固定家具）与移动物；保留 = 建筑结构（墙、地板、天花板、门窗、楼梯）。

### 产出发布（GitHub）

每次修改并出片后运行 `python -m tools.publish_deliverables`（vbr 环境）：把 `background_video.mp4 / background_scene.glb / background_mesh.ply` 与关键报告复制进 `deliverables/`、写 manifest、commit 并 push 当前分支。GitHub 凭据已配置（`~/.git-credentials`，chmod 600，两个 host：github.com 与 mirror）；因本机直连 github.com TLS 不稳定，`origin` 指向 ghfast 镜像（`https://ghfast.top/https://github.com/Atlansert/...`）。恢复直连：`git remote set-url origin https://github.com/Atlansert/video_background_reconstruction.git`。远端分支：`main`（基线）+ `svor-trial`（当前工作分支，含 deliverables，已推送至 9dec494）。**注意：推送用的 PAT 经聊天与镜像传输过，用后建议在 GitHub 页面吊销并换新。**

### SVOR 试验后端（`svor-trial` 分支，当前 `backend: svor`）

`video_completion.backend` 已切换为 `svor`（Wan2.1-VACE-1.3B + 两阶段 remove LoRA，Apache-2.0，比 ProPainter 非商用许可更友好）；ProPainter 成品备份在 `snapshots/videos/background_video_presvor.mp4`（md5 与切换前 deliverable 一致）：

- 代码：`vbr/models/svor.py`（`SVORAdapter`：帧/掩膜 → 4k+1 对齐分块（77 帧 + 20 帧重叠）→ 逐块 `conda run -n svor` 调 `predict_SVOR.py` → 重叠区线性交叉淡化 → **掩膜外透传合成**（`composite_source: true`，修复 SVOR 全帧重生成导致的墙位漂移/未遮挡物体闪现）→ 可选 RAFT 时序平滑（复用 `video_completion.temporal_smooth`，抑制填充区闪烁）→ h264+aac 封装）；`cli.py` 按 backend 分发，默认仍 `propainter`
- 资产：`external/svor/`（上游 tarball，gitignore）+ `external/svor/models/`（两 LoRA + Wan2.1-VACE-1.3B 原始格式基座，17.7GB，hf-mirror 下载；LoRA 原件备份在 `checkpoints/svor-lora/`）；独立环境 `svor`（python3.10 + torch2.7.0 + diffusers 0.31）
- **正式流水线切换运行（2026-09-12）**：`backend: svor` 全链重跑（分割指纹含 cli.py 触发重分割，掩膜逐像素一致），产物与 `svor_full/background_video_v2.mp4` 逐像素一致（生成确定性强）。同口径对比 ProPainter 快照：整体透传率 **0.071** vs 0.126，glitch 双方 0
- **边缘框与时序修复（2026-09-12 第二轮）**：`svor.mask_dilation: 8`（生成+合成共用膨胀掩膜，吞掉欠覆盖前景边缘，消除透传暴露的边缘框/墙面灰斑）+ svor 块内 `temporal_smooth.enabled: false`（恢复原始生成时序特性）。中间版指标：透传率 0.070、闪烁比 1.0002、glitch 0
- **填充突变修复（2026-09-12 第三轮）**：新增 `svor.fill_ema_alpha: 0.65` 光流 EMA 稳定器——上一帧稳定结果沿 Farneback 光流 warp 后在填充区内按权重融入，生成结构跟随相机运动而非逐帧重画（RAFT 中值治不了形变：中位数恒取当前值）。顺序关键：EMA 必须在透传合成之前（在合成后跑会把真实物体边缘拖进填充区，边框复现）
- **回归 v1 观感（2026-09-12 第四轮，当前配置）**：透传合成会把物体投影（阴影）也如实贴回背景（不如 v1），故 `composite_source: false`，EMA 扩展为双区全帧稳定：填充区 `fill_ema_alpha: 0.65`、墙地区 `wall_ema_alpha: 0.3`（抑制生成墙纹逐帧爬行）。v4 指标 vs v1：墙地时序差 **3.57** vs 3.75、填充深处 **3.96** vs 4.43、透传率 0.068 ≈ v1；快摇段无拖影、阴影由生成自然去除。GLB 46,595 点、glitch 0。如需恢复透传语义（墙地为真实像素）改回 `composite_source: true`（此时 wall_ema_alpha 不生效）。历史版本留档：`snapshots/videos/background_video_svorsmooth.mp4`（合成+平滑）
- **末段橱柜修补（2026-09-12 第五轮）**：末段上排玻璃橱柜掩膜漏检（区域覆盖仅 0.17–0.35，全局覆盖达标故自动漏检不触发）。通过强制窗口精修（wrapper 临时关 evidence 门 + box_seeds `{1550,1798,[0.13,0,0.59,0.5]}`，已固化进 yaml）把橱柜纳入掩膜（区域覆盖 0.69），仅重生成受影响的 6 个分块（~15 分钟，复用其余 26 块）后重拼接+EMA。橱柜已去除；模型在填充处给出半透明衣柜/面板幻觉（大填充区+厨房上下文下的扩散自主发挥，可选迭代：加大该窗口膨胀）。GLB 52,909 点、透传率 0.071、glitch 0
- **逐项时序优化（2026-09-13 第六轮，当前配置）**：按参考文档逐项落地——① 新增 `tools/flow_metrics.py`（warping-error 光流归一化 flicker + mask 抖动，基线 1.549/1.77%）；② 1103 衣物区补 box 种子（强制窗口精修，掩膜覆盖 0.48–0.82，最大幻影锚点消除）；③ `overlap 20→33`（步长 44≡0 mod 4）+ `fade_frames 16`，全片 41 块重生成；④ 跨窗噪声对齐（全局潜噪声切片，改了 `predict_SVOR.py`/pipeline）：**双窗实测变差（9.03→10.03）已关闭**——证明不匹配主导因素是 VACE 条件上下文而非初始噪声。最终指标：warping-error **1.502**（p95 2.59）、透传率 **0.058**、闪烁比 **0.989**、glitch 0、GLB 19,640 点。1103 幻影基本消除，末段 ghost 明显减弱（残余家具幻觉为扩散固有）
- **架构对照与最终定版（2026-09-13 第七轮）**：ProPainter 在当前掩膜上重跑后同尺对照——**warping error 2.00（填充区 2.44）远差于 SVOR+EMA 的 1.44**，"PP 稳定/SVOR 闪"的直觉被数据推翻（PP 传播伪影逐帧随光流漂移）；置信度融合（`tools/confidence_fusion.py`，C=PP 时序自洽度）实测恶化至 1.97 且阴影/糊影回归，**放弃**。最终采纳 EMA 0.75/0.45（v6）：warping-error **1.442**、透传率 **0.057**、闪烁比 **0.988**、glitch 0、GLB 28,168 点。感知层面的剩余不一致来自幻觉事件与长时程漂移（架构固有），逐帧 jitter 已优于 ProPainter
- **注意**：`box_seeds` 位于 segmentation 配置节内，**改动会使分割指纹失效触发全量重分割**（本轮流水线因此绕过、改由适配器直连出片）；下次全链运行会补一次重分割
- **剩余已知问题**：填充幻觉（帧 300/末段板块与 ghost 家具）为扩散生成固有，EMA 使其稳定；如需进一步可测试参考文档的置信度融合架构（ProPainter 先验+置信度图）或 VideoPainter 基线
- 切换方式：`configs/vggt_slam.yaml` → `video_completion.backend`（当前 svor）；SVOR 试验产物与诊断图集中在 `outputs/001_sam31_slam/experiments/`（含 `svor_full/`、`svor_trial/`、`diagnostics/`）

### 与官方 SVOR 的配置一致性核对（2026-09-16）

逐项对照 `external/svor` 官方默认值，发现两处偏离并已修正：

| 参数 | 曾用值 | 官方默认 | 影响 |
| --- | --- | --- | --- |
| `chunk_frames` | 77 | **81**（=4×20+1） | 77 帧窗口的黑斑退化率是 81 帧的 2–3 倍（实测帧84：0.95% vs 0.32%） |
| `dilation`（模型预处理层） | 0 | **6** | 官方在推理前再膨胀掩膜 6px 提供上下文；实测 81帧+dil6 组合把黑斑从 0.95% 降到 **0.20%（改善 79%）** |
| `overlap` | 33 | — | 随 chunk_frames 调整到 41（步长 40，仍 ≡0 mod 4，保持潜帧网格对齐） |

两项偏离是"帧 1–3 秒墙壁黑斑"的直接成因（大填充区上下文不足→生成退化）。

同时修正掩膜精修配置 `max_coverage_increase` 0.15 → 0.45：该上限会拒绝大面积物体（冰箱/壁橱占画面 >15%）的精修种子，导致掩膜覆盖在帧间剧烈抖动（实测 0.00↔1.00），物体时去时留——这是"52 秒处冰箱壁橱没去掉且杂乱"的成因。

### 目录结构约定（2026-09-13 重组）

- 项目根：代码（`vbr/ tools/ tests/`）、配置（`configs/`）、权重（`checkpoints/`，含 `svor-lora/`）、外部仓库（`external/`）、输入视频（`video/`）、发布物（`deliverables/`，git 跟踪）
- `outputs/<run>/` 根部只放流水线契约产物：交付物（视频/GLB/PLY/html）、帧与掩膜目录、状态与报告 JSON、`svor/`（适配器工作目录，代码按固定路径读写）、`slam*/`、`snapshots/{videos,geometry,masks}/`（版本回退快照）
- 一次性试验与诊断图 → `outputs/<run>/experiments/`；可再生中间目录（`frames_background/`、`masks_empty/`、`masks_*_baseline/`）按惯例直接删除，由流水线重建
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