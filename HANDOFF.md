# Video Background Reconstruction 项目交接文档

更新日期：2026-09-08（第七次更新：前景扣除的五个优化方向已依次落地，见第 2.8 节）

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

### 2.1 2026-09-05 新增：完整 VGGT-SLAM backend 已接入

- 新增 `vbr/vggt_slam_backend.py`：在 `vbr-slam` 环境中复用上游 `vggt_slam.Solver`，实现光流关键帧选择 → 子图 VGGT 推理 → SL(4) 位姿图优化 → SALAD 图像检索回环。
- 无头运行：stub 掉 viser Viewer；离线回环：通过 `SALAD_CHECKPOINT` / `DINOV2_REPO` / `DINOV2_CHECKPOINT` 指向 `checkpoints/salad/` 与 `checkpoints/dinov2/`。
- 尺度处理（关键设计）：每个子图的 VGGT 深度有独立单目尺度；优化后的 SL(4) 节点是**射影矩阵**，必须先按 `H / H[3,3]` 归一化再从 `det(H[:3,:3])^(1/3)` 提取子图尺度，把尺度折入深度图（`depth *= scale`），导出刚体外参，TSDF 无需改动即可融合。
- `--model-mode square`（默认）：与 `vggt_direct` 相同的 518×518 信箱式预处理，分辨率对齐；`crop` 模式为上游默认（518×294）。
- overlap 帧去重（每帧只导出一次）、TUM 格式轨迹 `slam/trajectory_tum.txt`、子图尺度与回环写入 `points_background.json`。
- `vbr/models/slam.py` 双 backend 分发；`configs/vggt_slam.yaml` 为对比运行配置（独立输出目录，不覆盖基线）。
- 两种 backend 在 001.mp4 上的对比结论见 `outputs/001_sam31_slam/comparison_report.json`：SLAM 覆盖 101 帧（7 子图、2 次回环），点数 18.3k 与 direct（32 帧 19.4k）相当，轨迹更平滑，几何结构一致；TSDF 表面略小（4455 vs 5780 顶点）。
- 新增 `tools/smoke_test_slam.sh`（约 2 分钟的 SLAM 端到端冒烟测试）和 `tools/evaluate_inpainting.py`（ProPainter 残留/闪烁量化评估）。
- mask 缓存指纹加入分割相关源码 hash（`_SEGMENTATION_SOURCE_MODULES`）；改分割代码后缓存自动失效，其余代码改动不再误伤 mask 缓存。
- `--stop-after` 新增 `geometry` 选项，跳过耗时的视频修复阶段，便于重建/几何对比迭代。
- 单元测试从 6 项增至 13 项（新增 SL(4) 尺度提取/射影归一化、mask 模型空间映射、adapter 命令构建等）。
- 项目根目录已 `git init`（main 分支，见第 14 节）。

### 2.2 2026-09-05 第二轮：mask 感知匹配、墙开口雕刻、prompt 扩充

- **mask 感知 SLAM 匹配**（`--mask-aware-matching`，默认开）：前景像素的深度置信度置零后再进入 Solver，使子图尺度估计（`add_edge` 的 good_mask）、子图点过滤与 SALAD 回环检索嵌入都排除前景干扰。实测所有子图尺度估计发生变化、可检平面 5→6（与 direct 持平）。
- **墙开口雕刻**（`geometry.carve_openings: true`）：新增 `masks_openings` 阶段——对重建帧（NPZ frame_ids）跑 preserve-only SAM3.1（door/window/staircase/stairs/kitchen cabinet/refrigerator/sink，带 manifest 缓存）；`_emit_wall` 把墙格网中心投影到各 hint 视图，深度关联（结构在墙面或墙前）后多数投票雕刻格网，另有“覆盖空洞+穿透证据”机制。001.mp4 上雕刻 9 格（楼梯/门区域），73 个 hint 帧。
- **plan_wall_detection**（脚印平面 2D RANSAC 竖直墙检测，含支撑连续性约束）：已实现并有单测，但**默认关闭**——本视频是多房间/多层级场景，中高层带混入非墙结构，需要先做房间分割才有意义（见 TODO）。
- **prompt 扩充**：探针验证后新增 `cup`、`bottle`、`table lamp`、`doormat`（此前漏检的桌上水杯、落地灯、地垫）；重跑分割后覆盖率 0.3393→0.3398，人工核对 `mask_overlay` 确认命中且无误伤。`tools/probe` 式验证思路：先用少量帧测 prompt 命中，再全量重跑。
- **工具**：`tools/rebuild_geometry.py`（只重跑几何阶段，秒级迭代墙/雕刻参数）。
- 代码梳理修复的 bug 清单：`_emit_wall` 内 floor/ceiling 比较反向、room_height 符号（两处）、fallback 墙切向坐标域错误（两面墙 quad 落在房间外）、hint 投影 mask(原始分辨率)/depth(模型空间) 坐标系混用、可见性条件反向、多数票基准错用 in_bounds 而非深度关联视角数、`sam31_keyframes` 空 prompt 校验顺序、`_emit_wall` 早退分支跳过 hint 雕刻。
- 测试 17 项全通过（新增 hint 投影雕刻含双坐标系映射与深度可见性用例）。

### 2.3 2026-09-06：SLAM 路径完整视频产物与评估

- 完整流水线（含 ProPainter）在 `outputs/001_sam31_slam/` 跑通：**9.2 分钟**（远低于 30-40 分钟预估），产出 `background_video.mp4`（960×540/1799 帧/29.97fps/H.264+AAC 音轨，22MB）。重跑的重建/几何与此前完全一致（17,932 点/9 格开口/4,406 TSDF 顶点），确定性良好。
- 评估（`outputs/001_sam31_slam/video_evaluation.json`，已并入 comparison_report.json）：残留均值 48.96、闪烁比 1.087，与基线（48.93/1.086）持平——**视频修复阶段只依赖 mask 与原帧，与 3D 后端无关**，指标一致即预期行为。
- 新旧背景视频同帧差分（帧 100/500/1200/1500）：落地灯、地垫、茶几杯瓶在 SLAM 版全部被移除（帧 1500 差异 0.3% 集中于杯瓶区域），其余区域差异 0.005-0.008%（无误伤）——**探针验证的 4 个新 prompt（cup/bottle/table lamp/doormat）的收益在最终视频中闭环确认**。
- 新工具：`tools/video_contact_sheet.py`（视频抽帧接触表，可叠加 overlay 视频逐帧对照）。
- ⚠️ 交接文档事故记录：`HANDOFF.md` 曾被会话报告内容覆盖、会话报告文件被删（疑似接管方误操作，已由 git 历史 `a045fcb`/`c46c92b` 恢复并补齐 2.3 节）。两份文档职责：**HANDOFF.md = 项目完整交接文档；SESSION_HANDOFF_2026-09-06.md = 代理会话进度报告**，请勿相互覆盖。

### 2.4 2026-09-06：mask 跳帧突变（关键帧边界硬切）

观测：独立区间 SAM2 在 stride=30 关键帧处硬切。`outputs/001_sam31_slam` 上关键帧边界 XOR 均值 0.065，区间内部 0.019，比值 3.51×；最大单帧 XOR 0.299（帧 1020）。1 帧闪烁约 0.0008。

修复（已接入默认配置，复用已有 SAM3.1 关键帧，未重跑检测）：

1. **双锚点传播**（`vbr/sam2_propagate.py`）：每个区间同时把两端 SAM3.1 种子登记为 conditioning frame，正向传播时 memory 同时看见下一关键帧；关键帧仍用精确种子覆盖。
2. **1 帧闪烁抑制**（`stabilize_mask_sequence`）：像素在 t±1 同为前景/背景而 t 相反时翻转；SAM3.1 关键帧钉住。
3. **大跳变距离变换过渡**（`blend_mask_jumps`）：XOR≥0.06 的跳变用 signed-distance 在最多 8 帧窗口内插值；关键帧钉住，不改 SAM3.1 语义。

`001_sam31_slam` 指标（`mask_temporal_before.json` / `mask_temporal_after.json`）：

| 指标 | 独立区间 | 双锚点+稳定 |
| --- | --- | --- |
| 平均覆盖率 | 0.340 | 0.338 |
| 平均帧间 XOR | 0.0202 | 0.0170 |
| 最大帧间 XOR | 0.299 | 0.088 |
| 关键帧边界 XOR | 0.0654 | 0.0269 |
| 边界/内部 XOR 比 | 3.51 | 1.61 |
| 1 帧闪烁（关/开） | 0.00080 / 0.00070 | 0.00017 / 0.00016 |

旧独立区间 mask 备份在 `outputs/001_sam31_slam/masks_independent_backup/`；旧背景视频备份在 `background_video_independent_masks.mp4`。`outputs/001_sam31/` 回归基线未改。测试 22 项。配置项：`sam2_dual_anchor`、`temporal_stabilize`、`temporal_xor_threshold`、`temporal_blend_window`。

用新 mask 重跑 ProPainter 后：残留均值 48.96→46.64，闪烁比 1.087→1.115。mask 层 1 帧闪烁降了约 5 倍。随后发现视频里物体仍会闪，根因是邻帧未遮住的物体被光流拷回，见 2.5。

### 2.5 2026-09-06：背景视频伪影与前景闪烁

根因：ProPainter 把未遮住像素当已知背景，从邻帧拷回原物体。重建 mask 在 750–930 覆盖率掉到 0.06–0.18（沙发漏检约 6 秒），拷贝率最高 0.93（帧 960）。

修复（只扩 inpaint mask，不改 SLAM 用的紧 mask）：

1. `prepare_inpainting_masks`：形态学闭运算 + 大连通域保守凸包 + 8 帧无门控并集（去闪）+ 24 帧重叠门控并集（跨秒漏检，避免不同视角家具叠成整帧）。
2. ProPainter：`mask_dilation` 8，`neighbor_length` 20，`subvideo_length` 80，`ref_stride` 5。

`001_sam31_slam` 当前视频：

| 指标 | 双锚点紧 mask | 门控 inpaint mask |
| --- | --- | --- |
| 拷贝率均值（mask 内 absdiff&lt;12） | 0.190 | 0.080 |
| 拷贝率&gt;0.2 的抽帧 | 101/360 | 36/360 |
| 闪烁比 | 1.13 | 0.82 |
| 残留均值 | 46.6 | 55.4 |

残留升高是预期：原物体更少被原样留下。仍偏高的 780–900 是 SAM 长时间漏检沙发。产物：`background_video.mp4`，overlay `mask_overlay_inpaint.mp4`。旧视频备份 `background_video_dual_anchor.mp4`。

### 2.6 2026-09-07：inpaint mask 精确优先

用户观测正确：2.5 的长时并集 / 凸包策略过度覆盖了前景外的背景。紧 reconstruction mask 平均覆盖率 0.338，而门控 inpaint mask 为 0.533，凸包单独新增约 1,732 万像素。

当前默认策略：`close_px=9`、`temporal_radius=2`、`expand_px=1`，禁用凸包和长时并集；ProPainter 仍保留内部 `mask_dilation=8`。新增指标 `mean_extra_coverage`、`p90_coverage`、`max_coverage`、`extra_coverage_ratio`。

精确 inpaint mask：均值覆盖率 0.371（额外 0.034），P90 0.556，最高 0.605；旧门控版均值 0.533。视频评估：闪烁比 0.967（仍低于 1），拷贝率 0.158（旧门控版 0.080）。按用户优先级，当前 `background_video.mp4` 采用**精确版**，而不再以大面积背景误遮罩来降低残影；旧门控视频保留为 `background_video_gated_inpaint.mp4`，旧门控 mask 为 `masks_inpaint_gated_backup/`。测试 25 项。

### 2.7 2026-09-07：物体初次出现时 mask 不及时 → 首现精修（onset refinement）

问题：SAM3.1 只在每 30 帧关键帧上做语义分割，物体在区间中途入画时，精确种子最多晚约 1 秒；`masks_inpaint` 的短时并集只补几帧，弥补不了整段延迟，而长时并集又会重新过度覆盖背景。

方案（默认开启，`segmentation.onset_refinement`）：

1. `detect_mask_onsets`：在基础 SAM2 mask 上检测“持续存在的新前景连通域”（`min_new_area_px=3500`、持续 3 帧、9px 膨胀关联、15 帧冷却、最多 20 个事件），单帧闪烁不会触发。
2. `run_refinement_masks`：对每个事件只取 `[onset-12, onset+6]` 的局部稠密窗口重跑 SAM3.1 全部 remove/preserve prompts。
3. 将得到 376 个精确 seed 并入关键帧 seed（共 423 个 pin），用 SAM2 双锚点重传播全部 1799 帧（422 个区间）。
4. `onset_latency_metrics`：对比基线，**平均提前 8.65 帧，20 个事件中 18 个提前**；结果写入 `onset_events.json`。
5. inpaint mask 仍保持 2.6 的精确策略，没有被扩大。

视频评估（当前 vs 首现前精确版）：拷贝率 0.149 vs 0.175，拷贝>0.2 帧 64 vs 87，时序比 0.984 vs 1.031，残留 49.9，闪烁比 0.933。inpaint 覆盖率 0.383（源 0.343），未突破精确上限。重建/SLAM 已用新 mask 同步重跑。产物：`background_video.mp4`（当前）、`mask_overlay.mp4`、`mask_overlay_inpaint.mp4`；回退快照 `background_video_preonset.mp4`、`masks_preonset_backup/`。CLI 增加 `--refine-onsets`，可在已有分割结果上单独重跑首现精修；测试 27 项。

### 2.8 2026-09-08：前景扣除不干净 → 五个优化方向依次落地

对“视频伪影闪烁”按五个方向依次实施并量化（全部保留可回退快照）：

1. **prompt 审计**（方向 2）：在漏检关键帧 840–1020 上探测 130+ 候选 prompt。新增 `dresser`、`chest of drawers`、`boxes`（在 990/1020 检出约 4.6 万/2.5 万新像素）；`cupboard/cabinet` 命中主要来自固定厨房柜体，与 preserve 冲突，未加入。
2. **选择性 pinning**（方向 4）：只 pin 带来首现提前的种子（342 个，另 26 个无效种子放开），减少 mask 边界抖动。重跑后首现平均提前 **9.4 帧**（18/20 事件）。
3. **两遍残差反哺**（方向 1，工具 `tools/refine_inpainting.py`）：找出输出与原图几乎相同的掩膜内连通域，并入 inpaint mask 重跑 ProPainter。两轮共补 26 万像素，拷贝率小幅下降；透传源主要在其它未遮帧，所以收益有限。
4. **ProPainter 参数搜索**（方向 3，工具 `tools/search_propainter_params.py`）：在 760–990 片段上网格搜索，`neighbor_length=80` 最优但全片 OOM（138GB），采用次优的 `neighbor_length=40, ref_stride=10`（拷贝 0.524→0.497）。
5. **时序中值平滑**（方向 5，工具 `tools/post_temporal_smooth.py`）：对掩膜内部做 3 帧中值滤波，再 h264+aac 封装。时序比 0.992→0.945。

最终视频指标（`video_evaluation.json` `vggt_slam_final_20260908`）：残留均值 50.4、闪烁比 0.949、拷贝率 0.136（路线起点 0.149）、mask 内时序比 0.945。inpaint mask 覆盖率保持 0.384+两遍反哺补丁，未回退到长时全局并集。测试 31 项。新工具：`tools/refine_inpainting.py`、`tools/search_propainter_params.py`、`tools/post_temporal_smooth.py`。回退：`background_video_n40_unsmoothed.mp4`（平滑前）、`masks_inpaint_pass1_backup/`（反哺前）。`outputs/001_sam31/` 回归基线始终未动。

### 2.9 2026-09-09：视频伪影残余 + GLB 破碎/空洞 → 漏检窗口精修、光流平滑、背景视频重估深度

用户复评发现：背景视频仍有伪影（沙发段残留最重），GLB 破碎且穿洞。方案分两线，全部落地并量化：

**视频线**

1. **持续漏检窗口重探测**（`detect_persistent_misses` + `_refine_miss_windows`，配置 `segmentation.miss_refinement`）：按逐帧覆盖率检测"≥30 帧持续低于 0.20"的窗口（命中 0–82、745–840、858–989），窗口内稠密重跑 SAM3.1。① 文本 prompt 增补（sofa bed/loveseat/futon/armchair/…）对**沙发失效**（该沙发对 SAM3.1 文本探测结构性免疫）；② **box 种子兜底**：`sam31_keyframes` 新增 `--box-prompts-json`（归一化 xywh，SAM3.1 原生支持 bounding_boxes），box 由现有 mask 团块按 25 帧子窗口自动推导并外扩 40%；③ **证据感知接受**（`select_miss_seeds` + `copy_through_evidence`）：覆盖率增幅上限 0.15 对"新增像素 ≥50% 落在拷贝残留区"的候选豁免——只扩在视频中确实可见的透传区，守住精准原则。命中帧（820/940）新增覆盖与拷贝残留重叠 89%/64%；最终 mask：全片均值 0.3485（上限未破），沙发窗 745–840 0.175、858–989 0.160。
2. **光流对齐时序平滑**（`vbr/temporal_smooth.py`，配置 `video_completion.temporal_smooth`）：RAFT（ProPainter 权重，vbr-seg 环境）把 t±1 warp 到 t 后再做 3 帧中值，取代无对齐中值（相机运动下的糊化/重影）。1799 帧全片，已接入 `ProPainterAdapter.run` 流水线；`tools/apply_temporal_smooth.py` 可对既有视频单独应用。
3. **两遍残差反哺 ×3 轮**（`tools/refine_inpainting.py --max-rounds 3`）：每轮 ~13 万像素入 mask。
4. **V3 分块 n=80 决定**：机制已实现（`chunk_ranges`/`_stitch_chunks`，单测通过），但主导残余是 mask 缺失而非参数（clip 上 n=80 仅 0.497→0.449），跳过全片运行，工具保留。
5. 最终视频（`background_video.mp4`，V1.c+V2+两遍×3）：残留 50.86、闪烁比 0.955、copy 全片 0.175、窗口 745–792 0.409（基线 0.466 的窗口内显著改善）、**866–989 仍 0.577 —— 本线唯一未根治项**，需人工归一化 box（`miss_refinement.box_seeds`，如 `[{"start":866,"end":989,"box":[0.2,0.3,0.95,0.9],"prompt":"sofa"}]`）才能由 SAM3.1 box 种子补完。

**几何线（GLB）**

6. **背景视频重估深度**（`tools/rebuild_geometry_from_background.py`）：从最终背景视频抽 1799 帧 + 全零 mask 重跑 vggt_slam（`vbr.vggt_slam_backend` 增加"空 mask 跳过护栏"判断）→ 前景区获得深度观测。点云 17,829→22,089，原始深度留存 21.4M→27.6M。
7. **真墙替代假墙**：plan 视图墙线检测（`fit_wall_lines`）在重估后生效——`wall_source` 从 `robust_footprint_fallback`（4 面 AABB 假墙、9 个乱刻开口）变为 **`plan_ransac`（8 面真墙、开口 9→3）**；`plan_wall_detection: true` 默认开启。假墙兜底偏移改为近邻点中位数。
8. **网格后处理**（`clean_mesh`：连通域过滤 81→3 组件、主件占 90%、耳切补洞 23 处）；修复 open3d `mesh +=` 与 numpy 视图别名的**指数级复制 bug**（16 次叠加后 6.17 亿顶点 → 全部改为一次性构建）。新 GLB：surface 4670 顶点、合并 4746 顶点/7970 三角、`background_scene.glb` 2026-09-09 版；旧版存 `background_scene_pre_redepth.glb`。子图尺度异常（0.23–1.0）已记录到报告（待后续归一化处理，未在本轮修复）。

指标与产物：`video_evaluation.json` `vggt_slam_final_20260909`、`comparison_report.json` `video_iterations`/`glb_redepth_20260909`、`geometry_redepth_report.json`、`miss_windows.json`（含 evidence 统计）、`pipeline_status.json` `final_20260909`。回退快照：`background_video_unsmoothed.mp4`、`background_video_premiss.mp4`、`masks_premiss_backup/`、`background_mesh_pre_redepth.ply`、`background_scene_pre_redepth.glb`。测试 48 项（新增：漏检窗口检测、box 种子推导、证据豁免、耳切、补洞、chunk 拼接、光流 warp/中值）。`outputs/001_sam31/` 回归基线始终未动。

**剩余可做**：① 用户提供沙发归一化 box 后重跑 `tools/refine_misses.py`（≈50 分钟）；② 子图尺度归一化（修复 0.23–1.0 尺度漂移造成的地面叠影）；③ vggt_slam 重估深度的 `--reuse-slam` 缓存选项（当前每次全重跑）。

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
- 22 个核心单元测试。
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

当前没有正在运行的训练、推理或长时间后台进程；项目处于“两种重建后端均可运行、等待质量继续提升”的交接状态。

当前配置仍然使用：

```yaml
slam:
  backend: vggt_direct   # 可选 vggt_slam（见 configs/vggt_slam.yaml）
```

`vggt_direct` 是单次全局 VGGT 预测（最多 32 帧，受 VGGT 上下文限制）；`vggt_slam` 是完整 VGGT-SLAM（光流关键帧 + 子图 + SL(4) 优化 + 回环，可扩展到全视频长度）。两者输出相同的 NPZ/PLY 接口，几何与视频阶段无需区分。

## 4. 下一步应该做什么

建议按以下顺序推进：

### 第一优先级：先验证当前结果

1. 查看 `outputs/001_sam31/mask_overlay.mp4`，确认红色区域是否真的对应可移动前景。
2. 查看 `outputs/001_sam31/background_video.mp4`，重点观察前景边缘、遮挡区域和时间闪烁。
3. 打开 `outputs/001_sam31/interactive.html`，确认点云、结构网格和相机轨迹是否合理。
4. 检查 `pipeline_status.json`、`geometry_report.json` 和 `slam/points_background.json`。

### 第二优先级：解决完整 VGGT-SLAM 接入问题（已完成，2026-09-05）

以上 1–7 全部完成：mask 按帧号对齐子图帧；帧选择 = `frame_stride` 预过滤 + 光流 `min_disparity` 关键帧（`max_keyframes` 上限）；封装 `vggt_slam.solver.Solver`（不调用 `main.py`）；导出优化后刚体外参、折算尺度后的深度、帧索引、子图尺度与回环数；foreground mask 在点云采样与 TSDF 中过滤（姿态估计尚未使用 mask，见 TODO）；NPZ 接口与 `vggt_direct` 完全兼容（含非正方形模型空间支持）；对比报告见 `outputs/001_sam31_slam/comparison_report.json`。剩余后续项：把 mask 引入 VGGT/子图匹配以降低前景对姿态的干扰。

### 第三优先级：提高背景质量

- 完善墙面、门窗、楼梯开口的结构建模。
- 增加多视角 3D mask 融合后再重投影的流程。
- 将 VLM 自动 prompt 发现接入实际 provider。
- mask 跳帧突变已压低；inpaint mask 已收紧为精确优先（见 2.4–2.6）；首现延迟已通过局部稠密 SAM3.1 种子平均提前 8.65 帧（见 2.7）。下一步是 780–900 的沙发漏检，优先改 prompt/阈值。
- 对 ProPainter 输出做前景残留和闪烁评估（`tools/evaluate_inpainting.py`；当前首现精修版：残留 49.9，闪烁比 0.93，拷贝率 0.149）。
- 增加缓存版本号或代码 hash（已完成：mask 缓存指纹含分割源码 hash）。
- 处理 VGGT 相对尺度（已部分完成：子图间尺度已折算统一；跨运行的真实米制标定接口仍缺）。

## 5. 项目目录结构

```text
video_background_reconstruction/
├── video/
│   └── 001.mp4                         # 输入视频
├── configs/
│   ├── default.yaml                    # 主配置（backend: vggt_direct）
│   └── vggt_slam.yaml                  # 完整 SLAM 对比配置（独立输出目录）
├── vbr/                                # 主项目 Python 包
│   ├── cli.py                          # Pipeline 总调度和 CLI
│   ├── config.py                       # YAML 配置读取
│   ├── video.py                        # 视频、帧、mask 和回退修复
│   ├── sam31_keyframes.py              # SAM3.1 关键帧分割
│   ├── sam2_propagate.py               # SAM2 时序传播
│   ├── vggt_direct.py                  # 直接 VGGT 重建（vbr-slam 环境内运行）
│   ├── vggt_slam_backend.py            # 完整 VGGT-SLAM backend（vbr-slam 环境内运行）
│   ├── geometry.py                     # 平面、TSDF、Poisson、网格（支持非正方形模型空间）
│   ├── interactive.py                  # Plotly 交互式 HTML
│   ├── prompts.py                      # GPT/VLM prompt 接口
│   ├── sfm.py                          # 旧/预留 SfM 相关代码
│   └── models/
│       ├── segmentation.py             # vbr-seg 子进程适配器
│       ├── slam.py                     # vbr-slam 子进程适配器（vggt_direct/vggt_slam 分发）
│       └── inpainting.py               # ProPainter + FFmpeg 适配器
├── external/
│   ├── VGGT-SLAM/                      # VGGT-SLAM 上游源码
│   ├── ProPainter/                     # ProPainter 上游源码
│   └── sam2/                           # SAM2 上游源码
├── checkpoints/
│   ├── sam3.1/sam3.1_multiplex.pt
│   ├── sam2/sam2.1_hiera_large.pt
│   ├── vggt/model.pt
│   ├── propainter/*.pth
│   ├── dinov2/dinov2_vitb14_pretrain.pth   # vggt_slam 回环依赖
│   └── salad/dino_salad.ckpt               # vggt_slam 回环依赖
├── outputs/
│   ├── 001_sam31/                      # 001.mp4 vggt_direct 基线产物（回归基线，勿覆盖）
│   └── 001_sam31_slam/                 # 001.mp4 vggt_slam 产物 + comparison_report.json
├── tests/test_core.py                  # 核心单元测试（13 项）
├── requirements*.txt                   # 环境约束
├── tools/                              # 下载/评估/冒烟测试脚本
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

1. ~~当前不是完整 VGGT-SLAM~~（已解决）：`vggt_slam` backend 已接入 submap、SL(4) 优化和回环。剩余问题：foreground mask 尚未参与姿态估计/子图匹配，前景仍可能干扰 VGGT 预测。
2. 单目视频中被家具全程遮挡的真实纹理不可观测，ProPainter 只能生成合理纹理，不能保证是真实纹理。
3. 两种 backend 当前运行的墙壁来源都是 `robust_footprint_fallback`，而不是稳定的 RANSAC 竖直墙。
4. fallback 生成的是四面完整矩形墙，不会自动开门、窗、楼梯和柜体洞口。
5. VGGT 坐标是相对尺度，`room_height` 不是实际米制高度；子图间尺度已在 SLAM 内折算统一，但跨视频的真实尺度标定接口仍缺。
6. `segmentation_stride: 30` 对快速运动物体可能过稀；提高关键帧密度会显著增加 SAM3.1 时间和显存占用。
7. SAM2 当前按独立区间传播，可减少长序列漂移，但区间边界可能产生 mask 变化。
8. 当前 mask 平均覆盖率约 33.93%，提示词较宽泛；应人工检查 `mask_overlay.mp4`，不能只看“Pipeline complete”。
9. `gpt.enabled` 当前为 `false`，VLM 自动发现前景只预留了接口，没有接入实际 provider。
10. cache fingerprint 已包含分割相关源码 hash；但重建/几何阶段没有缓存，改配置重跑会重复计算（当前可接受）。
11. 最近一次状态中的 `filled_enclosed_hole_pixels: 0` 表示最后一次运行时 mask 已经没有新孔洞，不代表算法没有实现孔洞填充。
12. `interactive.html` 是点云/网格浏览器，不是带完整语义层级、碰撞和导航的引擎场景。
13. `vggt_slam` 的 TSDF 表面（4406 顶点）小于 `vggt_direct`（5780），原因是子图残余失配与更保守的置信度过滤；可通过调大 `max_keyframes`/降低 `min_disparity` 或降低 `confidence_percentile` 改善。
14. VGGT-SLAM 的 SL(4) 节点是射影矩阵，任何从 `graph.get_homography()` 提取位姿/尺度的新代码都必须先做 `H / H[3,3]` 归一化（`vbr/vggt_slam_backend.py` 的 `rigid_from_similarity` 已处理，勿删）。
15. `plan_wall_detection` 在多房间/多层场景会把楼梯平台等误当墙（中高层带不纯净）；启用前需先实现房间分割。footprint fallback 仍是生产默认。
16. `outputs/001_sam31` 基线保留旧版 mask（与其输出一致）；`outputs/001_sam31_slam` 使用扩充 prompt 后的新 mask。两边 mask 版本不同，跨目录对比时注意（见 comparison_report.json 的 mask_version 字段）。
14. VGGT-SLAM 的 SL(4) 节点是射影矩阵，任何从 `graph.get_homography()` 提取位姿/尺度的新代码都必须先做 `H / H[3,3]` 归一化（`vbr/vggt_slam_backend.py` 的 `rigid_from_similarity` 已处理，勿删）。

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

- [x] 在保留 `vggt_direct` 的情况下，实现可选的 `vggt_slam` backend。
- [x] 设计完整 VGGT-SLAM 的 mask-aware 接口（点云/TSDF 过滤已做；姿态估计引入 mask 仍待做）。
- [x] 输出优化后每帧 extrinsics、frame ids、submap/loop closure 报告（`trajectory_tum.txt` + `points_background.json`）。
- [x] 对比 direct VGGT 与完整 VGGT-SLAM 的轨迹和网格质量（`outputs/001_sam31_slam/comparison_report.json`）。
- [x] 增加完整运行的回归测试或小视频 smoke test（`tools/smoke_test_slam.sh` + 13 项单元测试）。
- [x] 把 foreground mask 传入 VGGT/子图匹配过程（mask 感知匹配：尺度估计/点过滤/回环检索排除前景）。
- [x] 墙面加入门窗、楼梯、固定柜体的开口/遮挡建模（preserve mask 投影投票 + 穿透证据；001.mp4 雕刻 9 格）。

### 质量提升

- [ ] 调整 SAM3.1 prompt，减少宽泛的 `furniture` / `decoration` 误删。
- [ ] 对 mask 做边界评估、时序一致性评估和固定物体保护评估。
- [x] 墙面加入门窗、楼梯、固定柜体的开口/遮挡建模（见 2.2 节）。
- [ ] 将多视角 3D mask 融合和重投影作为可选增强阶段。
- [ ] 接入 GPT/VLM provider，并限制其输出为稳定的短名词 prompt。
- [ ] 增加真实尺度输入接口，例如已知门高、房间尺寸或相机高度。
- [x] 给缓存添加源码/模型版本号（mask 缓存指纹已含分割源码 hash）。
- [ ] 基于评估指标（残留/闪烁）迭代 ProPainter 与 mask 参数。

### 工程维护

- [x] 将项目根目录正式纳入 Git（已完成：main 分支初始提交 a5acff7，`.gitignore` 排除 checkpoints/outputs/external/video 等大目录）。
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

当前测试共 17 项，覆盖：

- 相机重力方向；四墙 fallback；
- mask 映射到 VGGT 模型空间（square/crop 两种模式）；
- SL(4) 射影归一化后的子图尺度提取与刚体位姿恢复（含 gauge 不变性）；
- SLAMAdapter 双 backend 命令构建与未知 backend 拒绝；
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

项目根目录已于 2026-09-05 初始化为 Git 仓库（分支 `main`，初始提交 `a5acff7`）：

- 纳入版本控制：`vbr/`、`configs/`、`tests/`、`tools/`、`requirements*.txt`、`README.md`、`HANDOFF.md`、`.gitignore`。
- `.gitignore` 排除：`checkpoints/`、`outputs/`、`video/`、`external/`、`source/`、`offline_bundle/`、`__pycache__/`、`.pytest_cache/`、`*.log`。

外部仓库状态：

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

