# 用位姿 + mask 在网格上去除前景（家具）

日期：2026-09-23 · 工具：`tools/subtract_foreground.py` · 分支：`svor-trial`

## 做了什么

基线网格（`outputs/vggt_slam_baseline/`）是**故意用空掩膜**建的（`tools/rebuild_vggt_baseline.py`），
所以原相机看到的家具全部被融进了网格。本工具在网格上事后把家具挖掉：用 NPZ 里的
SLAM 位姿把每个顶点投到 71 个关键帧，读该像素的 mask 与深度，投票后删除。

产物（`deliverables/reports/foreground_removal/`）：

| 文件 | 说明 |
| --- | --- |
| `compare_with_vs_without_furniture.mp4` | 左：有家具 ｜ 右：已挖除（同机位逐帧对应，带标注）**主验收片** |
| `walkthrough_no_furniture.mp4` | 去家具后单画面漫游（全长 1799 帧 + 原音轨） |
| `walkthrough_with_furniture.mp4` | 去家具前单画面漫游（对照组，同参数） |
| `foreground_removal_stills.jpg` | 4 个时间点 × 上下两排静帧对比 |

## 判据（这是本工作的核心）

**不要用 mask 占比阈值。** "某顶点在 ≥70% 的视角里是前景就删掉"听起来合理，在基线上也确实
会标出 ~25% 的顶点——但它同时会打穿真墙：在**控制组**（`outputs/geometry_prior_a/`，用掩膜帧
建的、本来就没有家具的网格）上它照样标出 9.8% 的顶点。那些是家具**背后**的墙和地板：家具挡在
前面，墙就"继承"了家具的 mask。占比判据分不清"这个面就是家具"和"家具大部分时间挡在这个面前面"。

真正的区分点是**真表面终究会被看见**：71 个关键帧的走位里，沙发背后的墙至少有一帧是直接
可见的背景；而沙发本体在任何看到它的帧里都是前景。所以判据是：

```
background_views == 0  且  foreground_views >= 3
```

## 数据（全部实测，非估计）

| 指标 | 基线（有家具） | 控制组 route A（无家具） |
| --- | --- | --- |
| 顶点数 | 232422 | 266481 |
| 判据命中 | **37985（16.34%）** | **1065（0.40%）** |
| 分离度 | — | **41×** |

控制组的 0.40% 就是该判据诚实的**误报率**。作为对比，占比阈值最好的分离度只有 ~10×。

**挖除结果**

| 指标 | 前 | 后 |
| --- | --- | --- |
| 三角面 | 444059 | 382093（−61966，−14.0%） |
| 顶点 | 232422 | 205376 |
| 边界边（洞） | 27537（4.05%） | 34525（5.85%） |

判据敏感度（基线，命中顶点数）：

| | fg≥2 | fg≥3（采用） | fg≥5 |
| --- | --- | --- | --- |
| bg≤0（采用） | 39793 | **37985** | 35440 |
| bg≤1 | 52619 | 50424 | 46985 |
| bg≤2 | 59448 | 56833 | 52839 |

`bg≤0` 是最严格的一档；放松到 `bg≤1` 命中数跳升 ~33%，会把大量只被瞥见过一次的真结构也划进来，
因此不放松。

## 诚实的局限

1. **挖掉后会留下空洞**（边界边 +1.4 个百分点）。这不是 bug：相机从没看到沙发／桌子**背后**
   的表面，那里本来就没有几何。要补只能靠先验或扩散生成——这正是结构先验（墙/地/顶 quad）
   存在的理由。当前基线网格是 TSDF-only，**不含**先验面（先验面会生成幽灵墙挡相机）。
2. **部分家具存活**。厨房柜体（第 4 帧）没被完全挖掉：它在某些视角下被看到了背景（bg>0），
   按判据保留。这是判据保守性的必然代价，宁可留也不误删结构。
3. **阈值本身是经验值**。`fg≥3` / `bg≤0` 是在这一条 71 帧走位上标定的。换序列需重新标定，
   标定方法就是本文的控制组做法——拿一条**已知无家具**的网格跑一遍，看误报率。
4. 教程/文档口径提醒：`masks/`（1799 帧）是被 `tools/refine_late_masks.py` **原地覆盖**过的
   权威版本；`masks_inpaint/` 是它的膨胀超集（只用于视频阶段），不要用来做网格雕刻。

## 坐标约定（踩过的坑，已写进工具 docstring）

深度/置信度是**模型空间**（本例 518×294），mask 是**原图空间**（960×540，由
`original_coords` 的信箱框给出）。所以：深度用模型投影索引，mask 用原图信箱投影索引。
第一版把 mask resize 到模型尺度却继续用原图坐标索引，投票被静默错配，命中率从 16.34% 变成
0.33%。已有单测锁死该约定（见下）。

## 回归测试

`tests/test_core.py::SubtractForegroundTests`（5 项，全库 **80 passed**）：

- `test_never_background_rule_separates_furniture_from_wall_behind_it` — 核心主张，合成投票；
  并断言占比判据确实会误收"背后的墙"
- `test_min_foreground_views_rejects_flicker_and_max_background_stays_strict`
- `test_vote_uses_model_space_depth_and_original_space_masks` — 锁死坐标约定
- `test_wrong_sized_mask_is_reported_not_silently_zeroed`
- `test_carve_requires_all_three_corners_and_keeps_the_rim`

**变异验证**（确保测试不是摆设）：去掉 `background` 条件 → 2 项失败；
用模型坐标索引 mask → 坐标测试失败。恢复后 80 项全绿。

## 复现

```bash
# 只看投票，不写网格
python -m tools.subtract_foreground \
    --mesh outputs/vggt_slam_baseline/background_mesh_denoised_reg.ply --dry-run

# 挖除
python -m tools.subtract_foreground \
    --mesh outputs/vggt_slam_baseline/background_mesh_denoised_reg.ply \
    --out outputs/vggt_slam_baseline/background_mesh_nofurniture.ply

# 验收（同机位对比）
python -m tools.render_trajectory_video --run-dir outputs/vggt_slam_baseline \
    --mesh outputs/vggt_slam_baseline/background_mesh_nofurniture.ply \
    --out deliverables/reports/foreground_removal/walkthrough_no_furniture.mp4
```