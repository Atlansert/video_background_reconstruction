# 项目交接文档（HANDOVER）

> 生成日期：2026-09-22 · 交接时 HEAD：`geometry-hybrid-b` @ `3f97642`
> 配套文档：`README.md`（技术路线说明）、`CURRENT_STATE.md`（截至 9/18 的旧状态，部分内容已过期）、`deliverables/`（交付物与评估报告）。

---

## 0. 先读这个（TL;DR）

- **项目目标**：从一段单目室内视频（`video/001.mp4`，1799 帧、960×540@29.97、60 秒）恢复"空房间"背景——移除全部家具，保留建筑结构（墙/地板/天花板/门窗/楼梯）——并输出修好的视频 + 3D 场景（GLB/PLY/交互 HTML）。
- **当前生产链路可用且完整**：`configs/vggt_slam.yaml` + `outputs/001_sam31_slam/`（`pipeline_status.json` 全阶段 `complete`）。视频阶段后端是 SVOR（扩散模型），几何有 3 条可选路线（P/A/B），见第 3、4 节。
- **最重要的三个交接点**：
  1. ~~几何改进集中在 `geometry-prior-a` / `geometry-hybrid-b`，尚未合并回 `svor-trial`~~ **已完成（2026-09-22，`7336469`）**：A/B 的 `geometry.py` 与几何工具已合并进生产分支 `svor-trial`。
  2. ~~生产几何地板先验面只剩 0.7 m²，属 P0~~ **已修复（2026-09-22，`ec177e4` + redepth 重跑）**。复核确认原缺陷成立且比文档记录更严重：生产中丢失的是**全部**结构先验面（不只是地板），且合并层还有一个**共面重复发射**缺陷（44.59 → 89.18 m²）。修复后生产几何 `prior_survival.ratio = 1.0`，地板先验 53.71 m²。对比见 `deliverables/reports/p0_prior_fix/`（含 `REPORT.md` 与三版漫游视频）。
  3. **`main` 分支本地领先 origin 7 个提交未推送**；`CURRENT_STATE.md` 停在 9/18，9/19–9/22 的工作未同步进去。
- **环境自检 + 测试是全绿基线**：`python -m vbr.cli doctor`（13 项）全过；`python -m pytest tests/ -q` → **83 passed**（P0 修复新增 3 项几何回归测试；此前 `geometry-hybrid-b` 68、`geometry-prior-a` 63、`svor-trial` 59）。接手第一步请重跑这两条。
- **安全提醒**：`~/.git-credentials` 里的 GitHub PAT 曾经过聊天与镜像传输，建议在 GitHub 页面吊销并换新（见第 7 节）。

---

## 1. 项目与输入输出

| 项 | 值 |
| --- | --- |
| 输入视频 | `video/001.mp4`（1799 帧 / 960×540 / 29.97fps / 含音轨） |
| 生产配置 | `configs/vggt_slam.yaml`（后端 `vggt_slam`） |
| 生产输出目录 | `outputs/001_sam31_slam/` |
| 语义 | "纯背景"：前景 = 全部家具（含冰箱/橱柜/水槽等固定家具）；保留 = 建筑结构 |
| **约定（每次完成工作后都要做）** | **① 把产物渲染成视频 → ② 推送到 GitHub Release → ③ 每个产物都写说明**。已固化成 `tools/publish_release.py`（`render` + `push` 两个子命令），**没有说明的产物不允许发布**（工具会拒绝），说明会自动生成到 Release 正文的"产物说明"表格里。详见 §4.10 |
| 主交付物 | `background_video.mp4`（去家具后的视频）、`background_scene.glb` / `background_mesh.ply` / `interactive.html`（3D）、`mask_overlay*.mp4`（掩膜预览）、若干 JSON 报告 |

**关键背景**：被家具永久遮挡的区域没有任何真实观测，纹理只能"推断"。项目历史结论：生成式补全（SVOR/扩散）观感稳定但会把跨帧不一致的纹理带进多视角几何（地面叠影、房间歪斜）；纯几何路线（A/B）用"墙直上直下"的结构先验填补遮挡区，代价是遮挡区无纹理（A）或纹理来自生成视频（B）。

---

## 2. 仓库、分支、worktree 现状

主仓库：`/data/lzx/video_background_reconstruction`（git 仓库）。
`origin` = `https://ghfast.top/https://github.com/Atlansert/video_background_reconstruction.git`（本机直连 github TLS 不稳，走 ghfast 镜像；直连恢复：`git remote set-url origin https://github.com/Atlansert/video_background_reconstruction.git`）。

| 分支 | HEAD | 与 origin | 角色 |
| --- | --- | --- | --- |
| `main` | `25f3c99` | **本地领先 7 个提交（未推送）** | 基线 |
| `svor-trial` | `572de90` | 同步 | **生产分支**（`deliverables/`、`CURRENT_STATE.md` 在这里演进） |
| `geometry-prior-a` | `79aabbf` | 同步 | 几何路线 A（原帧+结构先验） |
| `geometry-hybrid-b` | `3f97642` | 同步 | 几何路线 B（A 的几何 + 生成视频纹理）+ 基线去噪/正则化工具 —— **交接时所在分支** |
| `effecterase-trial` | `d581e31` | 同步 | 替代后端 EffectErase 的试验分支（**本地 worktree 已于 2026-09-22 删除**，仅存分支与远端） |
| `videopainter-trial` | `f2e69e8` | 同步 | 替代后端 VideoPainter 的试验分支（同上，worktree 已删） |

Worktree：现在只有主目录 `/data/lzx/video_background_reconstruction`。`_ee` / `_vp` 两个附加 worktree 已于 2026-09-22 移除（释放约 48G）；试验视频/报告归档在 `deliverables/trial_archive/`（EffectErase 视频+GLB、VideoPainter 视频+评估，共 27M）。分支本身保留在本地与远端，需要时 `git worktree add` 恢复。

### 分支 × 工具对照（交接要点）

不同分支的工具集不同——**在 `svor-trial` 上看不到几何路线/渲染工具**，不要以为丢了：

| 工具（`tools/`） | `svor-trial` | `geometry-prior-a` | `geometry-hybrid-b` |
| --- | :-: | :-: | :-: |
| `refine_late_masks` / `verify_late_masks` / `subtract_walls` / `clip_repair_masks` / `rerun_svor_partial` | ✓ | ✓ | ✓ |
| `rebuild_vggt_baseline` | ✓ | ✓ | ✓ |
| `rebuild_geometry_prior`（路线 A） | – | ✓ | ✓ |
| `render_trajectory_video`（漫游视频） | – | ✓ | ✓ |
| `rebuild_geometry_hybrid`（路线 B） | – | – | ✓ |
| `denoise_baseline_mesh` / `regularize_planes`（几何后处理） | – | – | ✓ |

`vbr/geometry.py` 同样分叉：`svor-trial` 是旧版；A/B 分支含先验面保活、墙角闭合、相机足迹延展等修复；B 另含纹理投影/细分/共面裁切。

**Tags / Releases**：

- `presvor-baseline-20260911`、`svor-ema-baseline`、`svor-final-20260917`（历史节点）
- Release **`svor-final-20260917`**：SVOR 交付物 + 官方代码核验产物（19 个 asset：视频/GLB/掩膜包/对比图等）
- Release **`vggt-slam-baseline-20260922`**（2026-09-22 新建）：VGGT-SLAM 基线漫游视频 3 版
  - 页面：https://github.com/Atlansert/video_background_reconstruction/releases/tag/vggt-slam-baseline-20260922
  - `trajectory_video_raw_slam.mp4`（原始，含空气噪点）、`trajectory_video_denoised.mp4`（去噪）、`trajectory_video_regularized.mp4`（+墙面正则化，推荐）
- Release **`p0-prior-fix-20260922`**（2026-09-22 新建）：P0 结构先验面修复的几何对比漫游（**只含 mp4，便于直接查看**）
  - 页面：https://github.com/Atlansert/video_background_reconstruction/releases/tag/p0-prior-fix-20260922
  - `walkthrough_p0_fixed.mp4`（修复后生产几何）、`walkthrough_p0_before.mp4`（修复前，先验被删）
  - `compare_p0_2panel.mp4`（BEFORE ｜ AFTER）、`compare_p0_3panel.mp4`（BEFORE ｜ CONTROL ｜ AFTER）
  - 全部由同一条 SLAM 相机轨迹渲染、均启用 `--ceiling-clearance 0.35`（渲染口径一致）
- Release **`regularize-fix-20260923`**（2026-09-23 新建）：墙面正则化修复的对比漫游（5 个 asset）
  - 页面：https://github.com/Atlansert/video_background_reconstruction/releases/tag/regularize-fix-20260923
  - `walkthrough_DENOISED_regularized.mp4`（当前工具，denoised 链路）/ `walkthrough_RAW_regularized.mp4`（当前工具，RAW 链路）/ `walkthrough_regularized_shipped.mp4`（原发布版，作对照）
  - `compare_regularize_on_DENOISED.mp4` / `compare_regularize_on_RAW.mp4`（左：未正则化 ｜ 右：当前正则化）
  - **2026-09-23 资产已刷新**：早期上传的 4 个 asset（`compare_regularize_2panel/3panel`、`reg_shipped_vs_fixed.jpg`、`walkthrough_regularized_fixed.mp4`）由过时代码产生，已替换
  - 修复实质（经更正）：**`fit_planes` 播种**（未播种时 RANSAC 每次返回不同平面集，前/后对比不可复现——这是让该问题久攻不下的根因）＋ **snap ramp 在 `max_shift` 处归零**（旧版用 `band`=0.10 > `max_shift`=0.04，截止处仍有约 24mm 位移，位移场断崖撕裂平整 patch）＋ **位移场 Laplacian 平滑 1 次**
  - 实测：RAW 墙面 RMS 17.25→12.39mm（平整 +0.0018、粗糙 −0.1071）；DENOISED 18.02→11.82mm（平整 −0.0236、粗糙 −0.1387）
  - **注意渲染器不使用存储法线**：`render_trajectory_video.py` 载入后调用 `compute_vertex_normals()`，会覆盖 PLY 中的法线（用垃圾法线渲染得到逐像素相同图像）。故法线侧改动对任何渲染视频零影响，只有顶点位置有效。
  - **验收勿用整帧高频能量**：该指标奖励过度平滑，压低真实表面起伏即可刷低数值。请看逐墙/分块指标或直接看视频。详见 `deliverables/reports/regularize_fix/REPORT_CORRECTED.md` 与 `COMPARISON_RAW_vs_DENOISED.md`

> **渲染陷阱**：`tools/render_trajectory_video.py` 从 `--run-dir` 下的 `geometry_report.json`
> 读天花板高度来做高机位钳制；若该文件不存在，工具只打印 `ceiling clamp skipped` 并照常出片，
> 结果是 1450–1650 段出现近乎全黑画面（实测帧均值 7.5–22.1）。做对比渲染前务必确认该文件在，
> 否则黑屏会被误读成几何缺陷。
>
> **ffmpeg 陷阱**：conda `vbr` 环境里的 `ffmpeg` **不含 libx264**（`ffmpeg -encoders | grep -c libx264` = 0），
> 直接调用会报 `Encoder not found`；须用 `/usr/bin/ffmpeg`（项目内 `vbr/models/inpainting.py::_resolve_ffmpeg`
> 已按此逻辑挑选）。

---

## 3. 当前 Pipeline（生产链路 P）

按 `python -m vbr.cli run --config configs/vggt_slam.yaml` 的顺序（`vbr/cli.py`）：

### 3.1 帧抽取（frames）
- `frames_all/` 全帧（1799 张，由 `video_source.json` marker 校验缓存）；关键帧按 `sampling.segmentation_stride: 30` 抽取。

### 3.2 分割（segmentation）
- SAM 3.1 在关键帧用 39 个开放词汇提示检测前景（`vbr/sam31_keyframes.py`；prompts 见 yaml），并用 `preserve_prompts: [staircase, stairs, door, window]` 从掩膜中扣回固定结构。
- SAM2.1 把关键帧联合掩膜传播到全部帧（`vbr/sam2_propagate.py`，双锚定模式），做封闭孔洞填充 + 时序稳定（`temporal_stabilize: true`，xor 阈值 0.06）。
- 掩膜链：`masks/`（重建用）→ `masks_inpaint/`（修复用，close 9 + 时序并集 ±2 + 扩张 1）。另有 `masks_keyframes_sam31_onset/`（种子集）、`masks_openings/`（门窗遮挡雕刻用 preserve 掩膜）。
- **重要陷阱**：`preserve_prompts` 的 `door`/`window` 会误匹配木纹冰箱、玻璃柜门（木纹≈门、玻璃≈窗），且 preserve 减除发生在 box prompt 之后，任何提示都救不回来——这是 9/19 两轮掩膜修复的根因之一。修复用 `tools/refine_late_masks.py --no-preserve` 绕开。

### 3.3 重建（reconstruction）
- VGGT-SLAM（`vbr/vggt_slam_backend.py`，跑在 `vbr-slam` 环境）：光流选关键帧 → 分块 VGGT 推理（子图）→ SL(4) 位姿图优化 + 回环；掩膜感知匹配（前景像素不参与匹配/尺度估计/检索）。
- 导出 NPZ 接口：`extrinsics / intrinsics / depth / confidence / foreground_masks / original_coords / frame_paths / frame_ids`；子图尺度已折算进深度与位姿；再用重力轴包络把房间高度锚定到 2.6m（公制阈值才成立）。
- 当前生产配置（对齐官方）：`min_disparity: 50`、`model_mode: crop`、`confidence_percentile: 0`（稠密输出，几乎不过滤置信度）。产物在 `slam/`（掩膜版，用于首版几何）与 `slam_bg/`（redepth 版，见 3.6）。

### 3.4 几何（geometry）
- `vbr/geometry.py::build_mesh`：重力方向（由位姿估计）→ RANSAC 拟合平面 → 墙线 2D RANSAC（俯视平面，强制竖直）→ 墙补到地板/天花板之间 + 门窗/楼梯开口雕刻（preserve 掩膜按深度可见性投影多数票）→ TSDF（体素 0.02、trunc 0.10）与结构先验合并 → 网格清理/抽稀（目标 45 万三角）→ PLY/GLB/HTML。
- 关键指标进 `geometry_report.json`（房间高/墙数/开口数/平面清单/网格统计）。

### 3.5 视频补全（video）
- 后端 **SVOR**（`vbr/models/svor.py`，跑在 `svor` 环境；Wan2.1-VACE-1.3B + 两阶段 remove LoRA，Apache-2.0）：81 帧窗口 / 重叠 41 / 步长 40（≡0 mod 4，保持潜帧对齐）/ 16 帧交叉淡化 / 20 步推理 / guidance 6.0 / seed 43 / 模型预处理 dilation 6。
- 关键机制：`composite_source: true` = 掩膜外透传真实像素（防墙位漂移/闪现）；`fill_ema_alpha: 0.75`（填充区光流 EMA 稳定）、`wall_ema_alpha: 0.45`（墙地区）——这两个是抗闪烁/抗纹理爬行的核心，顺序必须在透传合成之前。
- **膨胀参数分三层，不要混淆**（`configs/vggt_slam.yaml`）：
  - `segmentation.mask_dilation_px: 7` —— 分割后的重建掩膜（给 SLAM 用）；
  - `video_completion.mask_dilation: 8` + `mask_expand_px: 1` + `mask_close_px: 9` —— `masks_inpaint/` 修复掩膜的构成（吃欠覆盖前景边缘，消"边缘框"）；
  - `video_completion.svor.dilation: 6` —— SVOR 模型内部预处理（官方默认，供扩散上下文，修黑斑）。
- 块级重跑：`tools/rerun_svor_partial.py`（复用未受影响分块；生产视频 44 块中 22 块是 9/19 重跑的，见 `background_video.json.rerun_chunks`）。
- 输出 `background_video.mp4`（h264+aac，含原音轨）。

### 3.6 几何重估（redepth，非 cli 主链）
- `tools/rebuild_geometry_from_background.py`：对**生成视频**抽帧 → 空掩膜跑同一 VGGT-SLAM → 重建 GLB/PLY/HTML（覆盖主链首版，首版留档 `background_scene_pre_redepth.glb`）。9/20 已用最新 3D 优化重跑（`geometry_redepth_report.json`）。
- 动机（当年切换原因）：首版几何的遮挡区没有观测、墙只能退化；生成视频"补上"了观测。**注意**：这条路线的副作用是扩散纹理的跨帧不一致会污染几何（地面叠影/歪斜）——这正是几何路线 A/B 想解决的问题。

### 3.7 发布（publish）
- `tools/publish_deliverables.py`：把视频/几何/掩膜/报告复制进 git 跟踪的 `deliverables/` + 写 manifest + commit + push。交付版本 revision 记录在 `deliverables/manifest.json`（当前 `c1955b5`，2026-09-19 22:46）。

### 3.8 缓存与重跑行为（重要）
- 改任何掩膜/提示/相关源码 → 分割指纹失效 → 全量重分割（预期行为）。
- `box_seeds` 位于 segmentation 配置节内，改动同样触发全量重分割。
- 帧缓存 marker 不可放进 frames 目录（ProPainter 会读目录内所有文件）。
- SVOR 每块单独 `conda run` 重新加载模型（约 2–3 分钟/块；全片 44 块约 2.5 小时）。

---

## 4. 已完成工作（按主题）

### 4.1 SVOR 后端接入与调优（9/12–9/17，Artifact 在 `svor-trial`）
- 切换 `video_completion.backend: propainter → svor`；实现独立适配器 `vbr/models/svor.py`（分块 → 交叉淡化 → 透传合成 → EMA）。ProPainter 成品备份在 `snapshots/videos/background_video_presvor.mp4`。
- 多轮迭代结论（都有数据留档）：膨胀 8 修边缘框；填充 EMA 消闪烁；恢复 v1 观感需 `composite_source: false`→后改 true + 双区 EMA 0.75/0.45；末段橱柜漏检用强制窗口精修；overlap 20→33→41；**"ProPainter 稳定/SVOR 闪"直觉被数据推翻**（warping error 2.00 vs 1.44，见 `ab3567f`）。
- 物体边缘残留根因是合成**对称羽化**（最外 1px 环只有 0.621 alpha，38% 源像素混回）→ 改为只向掩膜外生效；掩膜内边缘残留 +0.343 → +0.007。

### 4.2 官方一致性核验（9/16–9/17，`047de04`）
- `external/svor` 还原为官方 tarball 原版（`diff -rq` 仅剩权重/MODEL_CARD 差异）；用官方 Quick Test 素材复跑并与官方发布结果定量对比：移除力度/背景保留一致，掩膜外结构保留略好（0.781 vs 0.686）；闪烁略高（22.0 vs 19.1）同量级。
- 关键发现：SVOR 整帧重生成会连带重画掩膜外真实结构（原生 bmx 仅保留 83.4% 未遮挡结构）——这是主流程 `composite_source: true` 的依据。
- 修正两处偏离官方：`chunk_frames` 77→81、模型预处理 `dilation` 0→6（黑斑 0.95%→0.20%）。产物在 Release `svor-final-20260917` 与 `deliverables/svor_validation/`。

### 4.3 掩膜两轮修复（9/19–9/20，`9d6a169`、`c1955b5` 等）
- **R4 后段（1440–1799）/ R5 中段（630–1030）**：冰箱/衣柜/吊柜门板漏检；根因①preserve 的 door/window 误判（见 3.2）②box 种子坐标打偏。修复工具 `tools/refine_late_masks.py`（`--no-preserve / --windows-json / --boxes-json / --union-existing`，SAM2 重锚定传播）+ 校验 `tools/verify_late_masks.py`。效果：冰箱区覆盖 0.429→0.713、0.355→0.727；0–630 帧零扰动（XOR=0）。
- **减墙**：`tools/subtract_walls.py`（SAM3.1 `wall` 提示 + SAM2 传播补齐，从厨房窗口掩膜减除 2910 万像素），墙/地/楼梯回归真实像素透传。
- **掩膜塌陷修正**：全量套件 + `--no-preserve` 把墙/天花板带进掩膜（填充区胀到 0.74 → SVOR 生成灰糊/黑块），用 `tools/clip_repair_masks.py` 裁回"修补前底座 + 仅物体新增"，`tools/rerun_svor_partial.py` 只重跑受影响 22/44 块。
- 这些修复已固化进生产视频（`background_video.json` 的 `rerun_chunks`，md5 与 `deliverables/background_video.mp4` 一致）。

### 4.4 3D 几何优化（9/20，`572de90`，`svor-trial`）
- 修 `clean_mesh`/`fill_holes` 重建时丢顶点色（GLB 恢复真实色彩，含 COLOR_0）；
- TSDF 0.03→0.02 + trunc 0.10 + 目标 45 万三角（表面顶点 31.8 万→83.5 万）；
- `fit_wall_lines` 与 3D RANSAC 融合去重（wall 1→3，`wall_source: ransac+plan`）；
- 两项被数据证伪不采用（无硬切；EMA 前输入无结构增益）。

### 4.5 几何路线 A / B（9/21 新增，`geometry-prior-a` / `geometry-hybrid-b`）
- **A**（`tools/rebuild_geometry_prior.py`）：原始帧 + 当前掩膜 → masked VGGT-SLAM → 结构先验补全 → 独立目录 `outputs/geometry_prior_a/`。约 4 分钟全链，扩散不参与几何。缺口（品红渲染口径）0–14%。
- **B**（`tools/rebuild_geometry_hybrid.py`）：几何与 A 逐位一致，纹理改为从 SVOR 视频多视角投影（真实区取真实像素、遮挡区取生成纹理）。纹理覆盖 99.2% 顶点、平均 8.6 帧/顶点。
- 期间修复多项问题（都在 `vbr/geometry.py`，**仅在 A/B 分支**）：
  - **A 与 B 共享**：先验面被 `clean_mesh` 当碎片删除（缺口 28%→4.6%）；墙线反平行去重死分支；墙角闭合 + 相机足迹延展（`_extend_wall_extents`）；相机中心公式简化。
  - **B 专属**（纹理相关）：细分共享中点去重（消除棋盘纹）、TSDF 共面裁切 `_prune_surface_against_priors`（消 z-fighting）、纹理帧"先截后匹配"顺序 bug。
- 三方对比图：`deliverables/reports/geometry_routes_3way.jpg`。指标：渲染缺口 A ~4.6% / B ~5.0% / P ~17%；平面重力对齐 A 0.95–0.97 / B 0.94–0.99 / P 0.92–0.96。

### 4.6 VGGT-SLAM 基线 + 去噪 + 正则化（9/21–9/22，`geometry-hybrid-b`）
- **基线**（`tools/rebuild_vggt_baseline.py`）：原始帧 + **空掩膜**（不去前景、家具保留），TSDF-only 网格渲染（结构先验面会生成幽灵墙挡相机，必须排除）→ `outputs/vggt_slam_baseline/`。
- **轨迹视频**（`tools/render_trajectory_video.py`）：SLAM 位姿（平移线性 + 旋转 slerp）插值到 1799 帧，EGL 离屏渲染，ffmpeg 编码 H.264 + 原音轨。默认 960×540@29.97。`--ceiling-clearance 0.35` 把高机位段（1450–1650 高举俯拍，SLAM 位姿高于估算天花板）的相机沿重力压到天花板下，避免黑屏（黑占比 68–82%→≤5%）。`--mesh/--npz/--stride/--no-audio` 可调。
- **去噪**（`tools/denoise_baseline_mesh.py`）：41% 融合样本处于最低置信度档，TSDF 化成 8365 个 <300 面的悬浮碎片；保持置信度截止不变（提高会削薄掠射角墙面），只滤小连通分量：8397→32 个分量。
- **墙面正则化**（`tools/regularize_planes.py`）：归因实验（纯灰光照 vs 无光照）证明"坑洼"是**法线抖动被方向光放大**（位置残差仅 6mm，法线偏差中位 21°）。修复 = 平面位置吸附（迭代 4 轮，残差 16→0.8–6.7mm）+ **法线按 band 全域混向平面法线**（21.1°→2.5°）；实现中修过两个缺陷（法线混合曾被 `max_shift` 错误限制；Open3D 写 PLY 需显式赋值法线）。家具/纹理经门控保护。
- 三版视频已上传 Release `vggt-slam-baseline-20260922`。

### 4.7 网格去前景（用位姿 + mask 雕刻，9/23，`svor-trial`）
- **动机**：基线网格是**故意用空掩膜**建的（4.6），家具被融进网格。本工作事后在网格上把它挖掉，不改动重建本身。
- **工具**（`tools/subtract_foreground.py`）：把每个顶点用 NPZ 位姿投到 71 关键帧 → 读该像素 mask 与深度 → 投票 → 删除「**从未被看到背景**且被看到前景 ≥3 次」的顶点所属三角面。
- **关键结论：不能用 mask 占比阈值**。「前景占比 ≥70% 就删」在基线上标出 ~25%，但在**控制组**（route A，本来无家具）上照样标出 **9.8%**——那是家具**背后**的墙：家具挡在前面，墙继承了家具的 mask。占比判据分不清「这面就是家具」与「家具大部分时间挡在它前面」。
- **采用判据**：`background_views == 0 且 foreground_views >= 3`。依据是真表面终究会被看见（沙发背后的墙至少 1 帧直接可见，沙发本体则帧帧都是前景）。实测分离度：基线 **16.34%** vs 控制组 **0.40%**（**41×**）；占比阈值最好只有 ~10×。控制组的 0.40% 就是该判据诚实的误报率。
- **结果**：三角面 444059 → 382093（−14.0%），顶点 232422 → 205376，边界边 4.05% → 5.85%。
- **诚实的局限**：① 挖后留空洞——相机从未看到家具背面的表面，本就没有几何（补只能靠先验/扩散）；② 部分柜体存活（它在某些视角被看到了背景，判据保守保留）；③ 阈值是在这条 71 帧走位上标定的，换序列须用同一套「控制组」做法重新标定。
- **坐标坑**（已写进 docstring + 单测）：深度是**模型空间**（518×294），mask 是**原图空间**（960×540，来自 `original_coords`）。深度用模型投影索引、mask 用原图信箱投影索引；第一版把 mask resize 到模型尺度却继续用原图坐标索引，命中率被静默错配成 0.33%。
- **产物**：`deliverables/reports/foreground_removal/`（含 `REPORT.md`、同机位对比片 `compare_with_vs_without_furniture.mp4`、两版全长漫游、静帧表）；网格 `outputs/vggt_slam_baseline/background_mesh_nofurniture.ply`。

#### 4.7.1 v1 失败与 v2 修正（9/24）——**重要，接手请先读**
- **用户反馈**：v1 效果很差——部分物体**只扣掉中间一块**、扣得**非常破碎**、还有**压根没扣掉**。
- **根因**：v1 是在**已融合的网格**上事后雕刻。桌面与它脚下的地板在 TSDF 里是同一张连续曲面，事后规则分不清。三个症状同一根因。
- **v1 实测**（射线投射，留出关键帧）：家具**残留 53.02%**——只去掉了 47%。
- **v2**（`tools/refuse_with_masks.py`）：**融合前**把掩膜像素的深度置零，Open3D 跳过深度 0，家具在构造上就不存在。保留基线自己的位姿/深度/置信度，只换掩膜。
- **v2 实测**：家具残留 **11.41%**——**该数字已证实不可复现，勿再引用**（见 4.7.2）；按 REPORT_V2 所述配置重算为 **23.89%**；基线 97.46% 是该度量有效性的对照。
- **掩膜膨胀消融**（掩膜边缘外常仍是家具，会长出"领口"残留）：0px→21.92%、2px→16.78%、**4px→12.15%（采用）**、5px→18.13%（`masks/`）。默认用 `masks_inpaint` + `--mask-dilate 4`。
- **度量方法**（关键，勿再用像素/顶点覆盖数）：`对角线/采样密度`会污染指标。改用**射线投射**：对家具像素投射，比较网格命中深度 vs 基线记录深度（基线融了家具，其深度即家具表面）——命中≈家具深度=仍在；明显更远=渲染到后面的墙=已去。脚本见 `/tmp/raycast*.py` 思路，方法已写入 `REPORT_V2.md`。
- **诚实的局限**：① 挖除处留空洞（无命中 82%）——相机从未看到家具背面，补面属于先验/扩散的后续工作；② 残留（重算 23.89%）非 0，主因是掩膜漏检与家具真正贴结构处，要再降需要更好的掩膜而不是更好的规则；③ 网格三角面 444059→683521（掩膜留洞后 TSDF 在洞缘生成更密几何，是移除的证据不是缺陷）。
- **v1 处置**：`tools/subtract_foreground.py` 保留（其坐标约定有单测），但**不再是推荐路径**。
- **测试**：新增 `RefuseWithMasksTests` 3 项，全库 **83 passed**；已做变异验证（把"置零深度"改成"置零颜色"→测试失败）。
- **Release**：`foreground-removal-20260923`（4 个资产：对比片 / 两版漫游 / 静帧表）。上传前用 `-c copy -movflags +faststart` 重封装——原始渲染的 `moov` 在文件 99.4% 处，浏览器要几乎下完才开播；重封装无损（视频流+音频流 md5 与原件一致），`moov` 落到第 36 字节，并逐字节回校过。**后续发布 mp4 到 Release 时请沿用 faststart。**


#### 4.7.2 v2 之后仍有残留 → v3 多视角一致性（9/24）——**最新，接手请先读**
- **用户反馈**：v2 之后**电视、部分冰箱、壁橱仍未去除干净**。
- **根因不在规则，在掩膜**：① `masks_keyframes_sam31/by_prompt/remove/` 下
  `television`/`refrigerator`/`kitchen_cabinet`/`cabinet`/`cupboard`/`furniture` 等
  **大量提示词目录为空**（0 个文件），而 `sam31_report.json` 却记着它们有几十万像素——
  **报告与磁盘不一致，逐提示词掩膜没落盘**；② 最终掩膜覆盖率在关键帧间抖动
  **13.05% ↔ 80.22%**；③ 全像素射线投射定位：**74.27% 的残留落在掩膜未覆盖的像素**，
  只有 25.73% 落在掩膜覆盖处 → 漏检为主。
- **真正的机制**：TSDF **对帧取并集**——"任一帧融进去就存在"。所以物体只有在**所有
  看到它的关键帧**都被掩膜覆盖时才消失；掩膜一抖，漏检那一帧就把家具重新融回来。
  v2 逐帧独立决定，无法克服这一点。
- **v3**（`tools/consensus_masks.py`）：把决定从"2D 像素"改成"**3D 点**"。一个点被
  "真正看到它的视角"中的**多数**判为前景，就从**所有**视角剔除。两个关键设计：
  - **只统计看得见的视角**：看不见 = **弃权**，不是投背景 → 不受覆盖率抖动影响。
  - **排除自身视角（leave-one-out）**：投票只数 `j != i`，因此**自身掩膜漏检的那一帧
    会被其他视角推翻**，而不是自己给自己背书；无其他视角可见时回退到该帧自己的掩膜。
- **v3 实测**（同一判据下重算，71 帧，射线投射）：v2 **23.89% → v3 12.33%**；基线 97.46%。
- **背景代价（用干净对照量化）**：掩膜外像素本身含漏检家具，故另取**距掩膜 ≥12px 的
  内部背景**——v2 97.97% → v3 **95.64%**（代价约 2.3pp），残留近乎减半。
- **网格生长消融**（`--consensus-dilate`，默认 = stride=3）：默认 12.33%/95.64%，
  0（纯最近邻）17.99%/96.89%。**是真实取舍**：去得更多也碰更多背景，默认取残留更低一档。
- **⚠️ v2 的 11.41% 不可复现，勿再引用**：按 `REPORT_V2.md` 自己写的配置
  （`masks_inpaint` + 4px）复算 v2 得 **23.89%**（5 个留出帧 19.83%），复现不出 11.41%。
  差异可定位到旧脚本用的是**未膨胀的 `masks/`**。本文与 `REPORT_V3.md` 的数字全部
  在**同一套配置**下重算，v2↔v3 对比自洽。另：旧的"5 个留出关键帧" `[6,20,34,48,62]`
  是**位置下标**不是 `frame_id`，按 id 取会全空——这也是旧数字难复现的原因之一。
- **诚实的局限**：① 残留 12.33% 非 0，主因仍是掩膜漏检，**要再降需修掩膜而非调规则**；
  ② 背景代价真实存在（约 2.3pp）；③ 挖除处留空洞（相机从未看到家具背面，补面属先验/
  扩散的后续工作）；④ 阈值在本条走位标定，换序列需重标。
- **测试**：新增 `ConsensusMaskTests` 8 项（决定规则抽成纯函数 `decide()` 后直接测），
  全库 **92 passed**。
- **产物**：`REPORT_V3.md`、`compare_v3_consensus_{2,3}panel.mp4`、
  `outputs/vggt_slam_baseline/trajectory_video_consensus.mp4`、
  `background_mesh_consensus.ply`（md5 `73ee15f67b4f26aea6add3b09d7ffb96`）。
- **Release**：`foreground-removal-v3-20260924`（4 个资产 / 71.6MB）：
  `compare_v3_consensus_3panel.mp4`（主验收片）、`compare_v3_consensus_2panel.mp4`、
  `background_mesh_consensus.ply`、`REPORT_V3.md`。
  上传后已**下载回来重新校验**（sha256 与本地一致、`moov` 在 0.00% 保持 faststart）。
  注意 GitHub 上传要走 `uploads.github.com`（与 api.github.com **不同主机**），
  用 api 基址拼会得到 `Name or service not known`。

### 4.9 GitHub Release 整理与最终交付（9/24）
- **最终交付物 Release**：`final-deliverables-20260924`，3 个资产：`background_mesh_no_furniture.ply`（v2 网格，26.3MB）、`interactive.html`（可交互 3D 查看器，22.5MB）、`DELIVERABLES_MANIFEST.json`（8 个交付物的 repo 路径 + sha256 + 大小 + `release_asset` 标记）。
- **为什么不把 6 个交付物也传上去**：它们**已随仓库跟踪**（`deliverables/`，`background_video.mp4` / `background_scene.glb` / `background_mesh.ply` / `mask_overlay{,_inpaint}.mp4` / `masks.tar.gz`），传 Release 只会同一文件存两份。清单里给了 repo 路径与校验和。
- **清理（987MB → 252MB）**：
  - `svor-final-20260917` 删 3 个原始 VGGT 中间件共 **750MB**（`vggt_pointcloud_original_video.ply` 278MB、两个 quickstart `.pcd` 146+326MB）。删除前逐个比对 **本地 sha256 完全一致**，且三个都**未在任何 Release 正文中被引用**。它们占当时全部 release 资产的 76%。
  - `regularize-fix-20260923` 删 `walkthrough_regularized_shipped.mp4`（11.9MB）：与 `vggt-slam-baseline-20260922` 的 `trajectory_video_regularized.mp4` **sha256 完全相同**（`b35e0618…`），保留一份即可；该 Release 正文已改写并注明去处。
  - 删两个**空壳 tag**（有 tag 无 release）：`presvor-baseline-20260911`、`svor-ema-baseline`。删前确认两者指向的提交（`25f3c99`、`afd353a`）**仍被 `main`/`svor-trial` 可达**，不丢历史。
- **刻意保留**：`foreground-removal-20260923` 里 v1 的两版视频（`walkthrough_no_furniture.mp4`、`compare_with_vs_without_furniture.mp4`）——作为"被否决做法"的对照证据。
- **校验口径**：所有 Release 资产的 sha256 与本地逐字节比对通过；每个 Release 正文只引用**确实存在**的资产（已用脚本核查无悬空引用）；所有 mp4 保持 faststart（`moov` 在 0.00%）。
- **注意**：GitHub Release 资产的删除**不走 SSH 安全策略**（走 API），因此这类操作已按用户逐项确认后执行，不默认自作主张。
- **`interactive.html` 上传前做过可用性核查**（不能只比字节就传）：Plotly **已内联**、无任何外部 `<script src=`（只有地图模板里的署名链接），所以**离线可开**；目标 `<div id>` 与 `Plotly.newPlot` 的 id 一致；payload 解码为 3 条 trace 且数值全 finite——背景点云 120000 点 / 补全网格 256476 点 / 相机轨迹 87 帧。**上传后又把下载回来的副本重新解码验证一遍**，确认不是坏文件。
- **它此前没进 Release 的原因**：`outputs/` 被 gitignore，该查看器既未被 git 跟踪、首版清单里也只列了路径没上传。↑至此补齐。
- **视频也一并补齐（同日）**：`background_video.mp4`、`mask_overlay.mp4`、`mask_overlay_inpaint.mp4` 此前**只存在于 git 仓库，不在任何 Release 里**（最核心的交付视频一直没被挂出）。现已上传 `final-deliverables-20260924`，该 Release 共 6 个资产 / 120MB。
- **同时修掉 faststart**：这三个视频的 `moov` 原本在 **99.59–99.97%** 处，即使从仓库下载也要几乎下完才能播。已 `-c copy -movflags +faststart` 无损重封装：**视频流 md5 与音频流 md5 与原文件完全一致**（已逐个校验），`moov` → 0.00%，并做**全片解码**验证（3 个视频各 1799 帧、零错误）。
- **同步 `outputs/` 的生产副本**，使 HANDOVER 5.1 表里"md5 与 deliverables 一致"这条不变量恢复成立。**注意口径**：重封装只改容器布局，容器 md5 必然改变，所以溯源要看**流** md5（`background_video` video `074775463ec5995c2413d84b05713175` / audio `98962f5d2f9d2f49617dd76d03a37de3`），不要再用容器 md5 判断同一性。

### 4.10 发布约定（长期有效，每次完成工作都要做）
**用户的长期要求**：以后每次完成工作，都 ① 把产物渲染为视频，② 推送到 GitHub Release，
③ 标注产物说明。

以前这件事每回都是临时写在 `/tmp` 的脚本里做，**没有工具**——`/tmp` 一清就没了，所以
既不可复现，也容易漏掉说明。现已固化为 `tools/publish_release.py`：

```bash
# ① 渲染（自动检查并无损重封装成 faststart）
python -m tools.publish_release render \
    --run-dir outputs/vggt_slam_baseline \
    --mesh outputs/vggt_slam_baseline/background_mesh_consensus.ply \
    --out deliverables/reports/foreground_removal/walkthrough_v3.mp4

# ②③ 发布，每个产物都要给说明
python -m tools.publish_release push \
    --tag <tag> --title "<标题>" --body-file <报告.md> \
    --asset <文件> "<这个产物是什么>" \
    --asset <文件> "<这个产物是什么>"
```

**关键点（都是踩过的坑）**：
- **说明是强制的**：没有说明的产物工具**直接拒绝**，且在任何网络调用之前就拒绝，
  不会发布出半个 Release。说明会自动追加成 Release 正文里的"产物说明"表格，
  所以它出现在 Release 页面上，而不只是本地报告里。
- `render` 必须给 `--run-dir` 或 `--npz`：底层渲染器靠它找相机轨迹，缺了会抛
  `TypeError: unsupported operand type(s) for /: 'NoneType' and 'str'`，工具已提前拦下。
- **子进程用 `sys.executable`**，不要写 `python`——conda 环境里 `python` 不在 PATH 上。
- **上传必须走 `uploads.github.com`**（与 `api.github.com` 是**不同主机**）。用 api 基址拼
  上传地址会得到 `Name or service not known`。
- **faststart**：mp4 的 `moov` 必须在 `mdat` **之前**，否则浏览器要几乎下完才开播。
  工具会在上传前检查并 `-c copy` 无损重封装。**校验同一性看流 md5，不要看容器 md5。**
- **校验用服务端 digest**：GitHub API 每个资产返回 `digest`（sha256），与本地比对即可，
  **无需把 27MB 再下一遍**（本链路批量下载不稳：urllib 会 `RemoteDisconnected`、
  curl 会 `SSL unexpected eof`）。
- 重名资产默认**报错**不覆盖；要覆盖显式给 `--replace`。
- **Token 不作为命令行参数**（会进 shell 历史和进程列表），只从 `~/.git-credentials` 读。
- Release/资产删除走 API，**不受 SSH 安全策略保护**，需用户逐项确认。

**测试**：`PublishReleaseTests` 6 项钉住上述要点（说明强制、先校验后取 token、
上传主机、faststart 检测、token 不走参数、镜像前缀的 origin 解析）。

### 4.8 单元测试
- `tests/test_core.py`：`svor-trial` 上 **83 项全绿**（`SubtractForegroundTests` 5 项 + `RefuseWithMasksTests` 3 项）；`geometry-prior-a` 63 项；`geometry-hybrid-b` 68 项。
- 去前景那 5 项做过**变异验证**（确认测试不是摆设）：去掉 `background` 条件 → 2 项失败；用模型坐标索引 mask → 坐标项失败；恢复后全绿。

---

## 5. 产物清单

### 5.1 `outputs/001_sam31_slam/`（生产，1.9G）
| 产物 | 说明 |
| --- | --- |
| `background_video.mp4` | 生产背景视频（1799 帧 / 60s / h264+aac）。**与 `deliverables/` 的同名文件已同步为同一 faststart 封装**；溯源看**流** md5（video `074775463ec5995c2413d84b05713175`、audio `98962f5d2f9d2f49617dd76d03a37de3`），容器 md5 因重封装而变（原 `3892f717…` → 现 `e5b140ed…`）属预期 |
| `background_scene.glb` / `background_mesh.ply` / `interactive.html` | 3D 交付物（来自 **redepth 路线**，9/20 重估） |
| `background_scene_pre_redepth.glb` / `background_mesh_pre_redepth.ply` | redepth 前的首版几何（留档） |
| `mask_overlay.mp4` / `mask_overlay_inpaint.mp4` | 掩膜预览 |
| `masks/` `masks_inpaint/` `masks_keyframes_sam31*` `masks_openings/` `masks_late_refinement/` `masks_wall_*` | 掩膜与种子 |
| `slam/` `slam_bg/` | VGGT-SLAM 两套重建源（掩膜版 / redepth 版） |
| `snapshots/{videos,masks,geometry}/` | 版本回退快照 |
| 报告 | `pipeline_status.json` `background_video.json` `video_evaluation.json` `geometry_redepth_report.json` `late_mask_refine_report.json` `miss_windows.json` 等 |
| `svor/` | SVOR 适配器工作目录（44 块缓存 + combined/composited/stabilized.mp4） |

### 5.2 几何路线（独立目录，不进生产链）
- `outputs/geometry_prior_a/`（381M）：`background_scene.glb` `background_mesh.ply` `structural_planes.ply` `interactive.html` `trajectory_video.mp4` `geometry_prior_report.json`
- `outputs/geometry_hybrid_b/`（537M）：同上 + `geometry_hybrid_report.json` + `frames_video/`（1799 张纹理采样帧）
- `outputs/vggt_slam_baseline/`（858M）：`pointcloud_original.ply`（1081 万点）`background_mesh_tsdf{,_decimated}.ply` `background_mesh_denoised{,_decimated,_reg}.ply` `background_scene.glb` + 三条轨迹视频
  - 另有 `background_mesh_nofurniture.ply`（去前景雕刻结果，9/23）+ 同名 `.foreground_report.json`；见 4.7 与 `deliverables/reports/foreground_removal/REPORT.md`

### 5.3 `deliverables/`（git 跟踪，`svor-trial` 上维护）
- 交付视频/GLB/PLY/掩膜包 + `manifest.json`（revision `c1955b5`）
- `reports/`：`geometry_routes_3way.jpg`（A/B/P 三方对比）、`vggt_slam_baseline_compare.jpg`（原片 vs 基线）、`foreground_removal/`（去前景：对比视频 + 报告）、各类评估 JSON
- `svor_validation/`：官方核验产物（两轮复核报告、对比视频图）

---

## 6. 环境与运行

### 6.1 四个 conda 环境（不可合并，原因见 README）
| 环境 | 用途 |
| --- | --- |
| `vbr` | 主调度、OpenCV/Open3D、TSDF/Poisson、几何、渲染、发布 |
| `vbr-seg` | Python 3.12 + SAM 3.1 / SAM2 / ProPainter / RAFT（分割、掩膜精修、光流指标） |
| `vbr-slam` | Python 3.11 + VGGT-SLAM（`external/VGGT-SLAM`） |
| `svor` | Python 3.10 + diffusers 0.31（SVOR 视频修复） |

### 6.1a 磁盘布局（2026-09-23 迁移后）

| 挂载 | 设备 | 容量 | 用途 |
| --- | --- | --- | --- |
| `/` | `/dev/nvme0n1p2` | 879G | 系统 + `/home`（**含 `/home/test/.cache`**） |
| `/data` | `/dev/nvme1n1p1` | 7.0T | 项目与大数据 |

`/home` 与 `/` 是**同一个文件系统**。2026-09-23 因根分区打到 100%（仅剩 3.1G），
把 `/home/test/.cache` 下 7 个占空间的目录迁到 `/data/cache/` 并**留软链接**，
根分区恢复到 **67%（277G 可用）**。

- 已迁移：`pip` 与 6 个 modelscope 模型（Qwen2-VL-72B / Qwen3-32B / Qwen3.5-27B /
  Qwen / Qwen3-Embedding-4B / Qwen3-8B-Base），合计约 276G
- **未迁移**：`Qwen3___5-122B-A10B`（234G，正在被 vLLM 端口 8002 服务）、
  `.cache/uv`（22G，正被 deer-flow 端口 8001 使用）
- 迁移脚本与完整记录：`/data/cache/move_cache.sh`、`/data/cache/MIGRATION_NOTES.md`
- **注意**：遇到 modelscope/工具报路径异常时，先确认它是否跟随符号链接。
- `/data` 已用到 98%，继续迁移前先确认余量。

### 6.2 GPU
- **优先 7 号卡**（0–3 被 vLLM 长期占用；4/5/6 可用）。推理前 `export CUDA_VISIBLE_DEVICES=7`。
- 交接时 GPU 4–7 空闲。

### 6.3 常用命令速查
```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate vbr
export CUDA_VISIBLE_DEVICES=7

# 自检（13 项）与测试（68 项）
python -m vbr.cli doctor --config configs/vggt_slam.yaml
python -m pytest tests/ -q

# 全链（改了掩膜/提示/源码后必须 --force；约 3–4 小时）
python -m vbr.cli run --config configs/vggt_slam.yaml --force

# 掩膜精修（缓存分割；late/mid 厨房窗口）
PYTHONPATH=. conda run -n vbr-seg python -m tools.refine_late_masks --no-preserve --no-overlays \
    --windows-json '[{"start":630,"end":1030}]' --boxes-json '[...]'
python -m tools.verify_late_masks                       # 前后覆盖对比

# SVOR 受扰分块重跑后重新拼接（按帧区间指定，工具自行换算分块）
PYTHONPATH=. python -m tools.rerun_svor_partial --output-dir outputs/001_sam31_slam \
    --config configs/vggt_slam.yaml --ranges-json '[[630,1030],[1440,1799]]'

# 几何：三条路线
python -m tools.rebuild_geometry_prior   --out outputs/geometry_prior_a  --config configs/vggt_slam.yaml
python -m tools.rebuild_geometry_hybrid  --out outputs/geometry_hybrid_b --config configs/vggt_slam.yaml
python -m tools.rebuild_geometry_from_background --output-dir outputs/001_sam31_slam   # redepth（路线 P）

# 几何后处理（去噪 / 平面正则化，对任意网格可用）
python -m tools.denoise_baseline_mesh --run-dir outputs/vggt_slam_baseline --min-component-triangles 300
python -m tools.regularize_planes \
    --mesh outputs/vggt_slam_baseline/background_mesh_denoised_decimated.ply \
    --npz  outputs/vggt_slam_baseline/slam/points_background.npz \
    --out  outputs/vggt_slam_baseline/background_mesh_denoised_reg.ply \
    --band 0.10 --max-shift 0.04 --normal-pull-weight 0.95 --iterations 4

# 轨迹漫游视频（任意网格 + 任意重建位姿）
python -m tools.render_trajectory_video --run-dir outputs/geometry_hybrid_b \
    --mesh outputs/geometry_hybrid_b/background_mesh.ply \
    --out  outputs/geometry_hybrid_b/trajectory_video.mp4

# 评估 / 发布
python -m tools.evaluate_inpainting --output-dir outputs/001_sam31_slam --windows-json outputs/001_sam31_slam/miss_windows.json
python -m tools.publish_deliverables          # 提交 + push（另加 --no-push 仅提交）
```

### 6.4 日志与产物位置约定
- SLAM/SAM/SVOR 日志：`outputs/<run>/logs/`。
- 一次性诊断请放 `/tmp`，不要污染 `outputs/`（项目惯例：`outputs/<run>/` 只放流水线契约产物）。

---

## 7. 已知遗留问题 / 风险

1. ~~**分支未合并 + 生产几何仍带旧 bug（重要）**~~ → **已修复（2026-09-22）**。A/B 的 `geometry.py` 已合并进 `svor-trial`（`7336469`），P0 修复在 `ec177e4`，生产 redepth 已重跑。复核结论与更正：
   - 原记录的"地板先验面 0.7 m²、A 路线 89.7 m²"**两个数字均精确复现**（0.70 / 89.70 m²），缺陷判断成立；
   - 但**生产中丢失的是全部结构先验面**（地板+天花板+墙），不只是地板大 quad——生产 `structural_planes.ply` 的 19 个连通分量**全部**小于阈值 100，被 `clean_mesh` 整体删除；
   - **A 路线的 89.7 m² 是重复计数**：44.59 m² 的真实先验被"双对角线"发射两遍（4 个三角仅 4 个唯一角点，比值恰 2.00），会引发 z-fighting。诚实的先验地板面积是 **44.59 m²**，真实修复倍数为 ~64× 而非 ~128×；
   - 第二个缺陷是**共面重复发射**：改"后追加先验"后，合并层的 `fill_small_boundary_holes` 把先验 quad 自身的 4 边边界环当孔洞耳切封盖。修法＝`protected_vertices`（整环皆先验顶点则跳过）；
   - 新增 `geometry_report.json` 的 `prior_survival` 字段（面积/比值/铰孔填充新增面积），使"先验是否到达最终网格"可回归——此前该缺陷只能靠手工量网格发现（`structural_vertices` 77→107 的差异太隐蔽）；
   - 修复后生产几何地板先验 53.71 m²、天花板 53.71 m²、`ratio = 1.0`。效果对比见 `deliverables/reports/p0_prior_fix/`。
2. **`main` 本地领先 origin 7 个提交未推送**（`25f3c99` 起）。交接前建议确认是否要推送。
3. **`CURRENT_STATE.md` 过期**（停在 9/18）：测试数（51→59/63/68）、掩膜第二轮修复、几何优化、路线 A/B、基线视频都未写入。README 在 A/B 分支已更新到 9/21 状态。
4. **B 路线遮挡区纹理仍有生成痕迹**（模糊/糊块），是"纹理取自扩散"的固有代价；肉眼验收建议直接看 `outputs/geometry_hybrid_b/trajectory_video.mp4`。
5. **剩余墙面波浪**：位置/法线正则化后仍留较大尺度低频起伏（选的是保守参数 band 10cm / 位移 4cm）；如不满意可用 `--band 0.15 --max-shift 0.06 --iterations 6` 再榨一版。
6. **路线 P 的几何质量根因未彻底解决**：redepth 依赖生成视频，几何仍受扩散纹理污染；A/B 是替代方案但尚未转正（未接入 `cli.py` 主链，仍是独立工具）。
7. **子图尺度漂移未归一化**：redepth 路线 6 子图尺度 0.32–1.0（`geometry_redepth_report.json` 的 `submap_scales`），早期 18 子图口径曾达 0.23–1.0。目前靠 2.6m 公制锚定 + 结构先验压住（可能表现为 GLB 地面局部叠影），未做逐子图尺度求解。
8. **历史遗留窗口 764–815（沙发漏检）**：旧文档记载文本 prompt 对该沙发免疫；9/19 修复后未专门复验，需要时用 `refine_late_masks` 同款流程处理。
9. **ProPainter 及权重为非商用许可**（仅 `backend: propainter` 时使用；SVOR 是 Apache-2.0）。商用前需逐个核对权重许可（SAM3.1/SAM2/VGGT/RAFT 等）。
10. **PAT 安全**：`~/.git-credentials` 中的 GitHub PAT 曾过聊天/镜像，建议吊销换新；`origin` 走 ghfast 镜像，直连不稳。
11. **egress 依赖**：ghfast 镜像当前可用；若失效，换直连或换镜像（`git remote set-url`）。

---

## 8. 接下来的工作方向（按建议优先级）

### P0：确认并固化几何路线（最关键的决策）——**1、2 已完成 2026-09-22**
1. ~~肉眼验收三条视频~~ → 已渲染并量化对比（见下 2 与 `deliverables/reports/p0_prior_fix/`）。
2. ~~**合并 A/B 的 `geometry.py` 到 `svor-trial` 并重跑 redepth**~~ → **已完成**（`7336469` 合并、`ec177e4` 修复、redepth 重跑）。生产几何已拿到完整先验面（地板 53.71 m²，`ratio = 1.0`）。注意修复的实质比原描述更大：原记录只说"地板先验面 0.7 m²"，实测是**全部先验面被删**外加**共面重复发射**，两者都已修掉。
3. 若 B 达标：让 redepth 工具支持"几何用原始帧 masked SLAM、纹理用生成视频"的混合模式（现在 redepth 是"二者都用生成视频"）。
   - 入口：`build_mesh(..., texture_frames=...)` 已实现（`geometry-hybrid-b`），只需在 redepth 工具里把 `slam/`（原始帧掩膜版）与 `frames_video/` 组合。
4. 顺带把 `denoise_baseline_mesh` / `regularize_planes` 作为通用后处理应用到生产几何（目前只用在基线上）。**仍未做**。

### P1：工程收尾
4. 推送 `main`（7 个提交）或明确其归属；把 `CURRENT_STATE.md` 更新到 9/22 状态（或直接把 HANDOVER 中指出的变更补进去）。
5. 决定 `geometry-prior-a` 是否保留（A 的独立价值：无生成纹理的纯几何验收；否则可作为 B 的上游分支保留）。

### P2：质量继续榨
6. 更强墙面正则化参数（见 7.5），或自适应 band（按平面大小/观测密度）。
7. 遮挡区纹理：B 路线的糊块可尝试"生成视频 + 时序中值"或限制生成纹理的使用范围（只用于低观测区）。
8. 子图尺度归一化（若继续用红线子图 pipeline 出图）。

### P3：可选探索（已有分支；worktree 已清理）
9. `effecterase-trial` / `videopainter-trial` 两个替代后端试验分支保留在本地与远端；本地 worktree 已于 2026-09-22 删除，产物归档在 `deliverables/trial_archive/`。若要复现试验，`git worktree add <dir> <branch>` + 复原 `external/<backend>` 权重（已随 worktree 一并删除，约 20–22G）。
10. `effecterase` conda 环境仍在 `/home/test/miniconda3/envs/effecterase`（6.7G，9/13 后未用）——如确认不再需要可 `conda env remove -n effecterase` 再释放。
11. 参考文档 `参考.txt` 与 `offline_bundle/`：早期调研材料，未清理。

---

## 9. 接手检查单（Day 1）

1. `git status`：确认分支与工作树状态；`git log --oneline -5` 看最近提交。**2026-09-22 后生产分支 `svor-trial` 已包含几何改进（`7336469`），几何工作可直接在生产分支进行。**
2. `conda activate vbr && python -m vbr.cli doctor --config configs/vggt_slam.yaml` → 13/13 过。
3. `python -m pytest tests/ -q` → **83 passed**。
4. 看产物：
   - 生产视频 `outputs/001_sam31_slam/background_video.mp4`
   - B 路线漫游视频 `outputs/geometry_hybrid_b/trajectory_video.mp4`
   - 基线三版视频（Release 或本地 `outputs/vggt_slam_baseline/`）
   - 对比图 `deliverables/reports/geometry_routes_3way.jpg`
5. 读 `README.md`（三路线说明）+ 本文档第 3、4 节。
6. 如需重跑全链：先备份 `outputs/001_sam31_slam/`（约 2G），再 `vbr.cli run --force`；SVOR 全片约 2.5 小时，VGGT-SLAM 约 10–15 秒/次，几何阶段约 3 分钟。
7. 动手前先确认要动的分支——**生产在 `svor-trial`，最新几何在 A/B 分支**，两边 `geometry.py` 不同。

---

## 附：关键文件索引

| 文件 | 作用 |
| --- | --- |
| `vbr/cli.py` | 主链调度（frames→seg→recon→geometry→video），`_load_opening_hints`、`_refine_miss_windows` 等 |
| `vbr/geometry.py` | 几何核心（build_mesh / fit_wall_lines / _emit_wall / TSDF / 正则化辅助）。**A/B 分支版本更强** |
| `vbr/models/svor.py` | SVOR 适配器（分块/淡化/透传合成/EMA；`_predict_chunk`/`_finalize` 支持部分重跑） |
| `vbr/models/slam.py` + `vbr/vggt_slam_backend.py` | VGGT-SLAM 导出（NPZ 契约：位姿/深度/置信/掩膜） |
| `vbr/models/segmentation.py` + `vbr/sam31_keyframes.py` + `vbr/sam2_propagate.py` | 分割链 |
| `vbr/video.py` | 掩膜后处理（封闭孔洞/时序稳定/修复掩膜/overlay/视频封装） |
| `tools/refine_late_masks.py` | 定向掩膜修补（`--no-preserve` 等参数，是掩膜修复的主力工具） |
| `tools/rerun_svor_partial.py` / `tools/clip_repair_masks.py` / `tools/subtract_walls.py` | SVOR 分块重跑 / 掩膜裁回 / 减墙 |
| `tools/rebuild_geometry_{prior,hybrid,from_background}.py` | 三条几何路线 |
| `tools/{denoise_baseline_mesh,regularize_planes}.py` | 几何后处理（去噪 / 平面+法线正则化） |
| `tools/subtract_foreground.py` | 网格去前景 v1（事后雕刻；实测只去掉 47%，**已不推荐**，见 4.7.1） |
| `tools/refuse_with_masks.py` | 网格去前景 v2（v3 的融合底座；逐帧独立，**残留 23.89%**，见 4.7.2） |
| `tools/consensus_masks.py` | 网格去前景 **v3（推荐）**：多视角 mask 一致性，残留 **12.33%**（见 4.7.2 / REPORT_V3.md） |
| `tools/publish_release.py` | **发布约定工具**：`render` 渲染漫游视频（自动 faststart）、`push` 建/更新 Release。**强制每个产物必须有说明**，并自动生成产物表；上传走 `uploads.github.com`；发布后用服务端 sha256 校验（见 4.10） |
| `tools/render_trajectory_video.py` | 轨迹漫游视频（任意网格 + 位姿；`--ceiling-clearance` 防黑屏） |
| `tools/evaluate_inpainting.py` / `flow_metrics.py` | 视频质量评估（残留/闪烁/glitch/warping error） |
| `tools/publish_deliverables.py` | 交付发布（复制→manifest→commit→push） |