# P0 修复效果对比：结构先验面恢复

生成日期：2026-09-22 · 分支 `svor-trial` @ `7336469` · 修复提交 `ec177e4`

## 一、缺陷是什么

生产交付网格 `outputs/001_sam31_slam/background_mesh.ply` 中，**所有结构先验面
（地板 / 天花板 / 墙）都被删除**，墙地顶只能靠 TSDF 表面支撑；而 TSDF 在家具
永久遮挡区没有观测——先验面正是为此存在。

两个缺陷叠加（详见提交信息）：

1. **删除**：`svor-trial` 的 `build_mesh` 在合并后才跑 `clean_mesh`，
   small-component 过滤器（阈值 `mesh_min_component_triangles: 100`）把每个
   先验 quad（2 三角/分量）当碎片整体删除。实测生产 `structural_planes.ply`
   的 19 个分量**全部**小于 100。
2. **重复**：A/B 分支改为"先清理稠密表面、后追加先验"，修掉了删除，但合并层的
   `fill_small_boundary_holes` 又把每个先验 quad 自身的 4 边边界环当成 artifact
   孔洞耳切封盖，在被盖面之上叠了一层共面副本——地板 44.59 → 89.18 m²
   （4 个三角只有 4 个唯一角点，比值恰 2.00），渲染表现为 z-fighting。

## 二、修复内容

`vbr/geometry.py`：

- `fill_small_boundary_holes(..., protected_vertices=...)`：整环皆为先验顶点的
  边界不是孔洞，跳过；真实 TSDF 孔洞的环含表面顶点，仍照常填充。
- `build_mesh` 新增 `prior_survival` 报告字段（先验面积 / 进入合并网格的先验面积 /
  比值 / 铰孔填充新增面积），让"先验是否到达最终网格"成为**可见、可回归**的数字。
- 新增 `tests/test_core.py` 3 项回归测试；全量 68 → **71 passed**。

## 三、实测对比

### 3.1 同一份生产重建（slam_bg），只换 geometry.py

| 状态 | 地板先验面 | 大三角数 | 唯一角点 | 判定 |
| --- | --- | --- | --- | --- |
| 修复前（svor-trial 旧版） | **缺失**（无 >1 m² 三角） | 0 | – | 先验被删除 |
| 修复后 | **53.71 m²** | 2 | 4 | 正确 |
| 参考：A 路线 | 89.18 m² | 4 | 4 | 重复发射（比值 2.00） |

`prior_survival.ratio = 1.0`（结构先验面积 167.46 m² 全部进入最终网格）。

### 3.2 交付产物（重跑 redepth 后）

`outputs/001_sam31_slam/`：地板先验 53.71 m²、天花板先验 53.71 m²、
总表面积 193.7 → 361.5 m²、`prior_survival.ratio = 1.0`。

### 3.3 控制实验（隔离先验的影响）

为排除"旧网格 tessellation 不同"的干扰，构造控制网格 = 修复后的生产网格
**仅剥掉先验面**（其余完全一致），用同一条 SLAM 相机轨迹渲染三版漫游视频：

| 对比 | 平均像素差 | 最大差 | 差值 >2 的帧 |
| --- | --- | --- | --- |
| 修复前 vs 修复后 | 24.21 | 120.5 | 91.3% |
| **控制组 vs 修复后（仅先验差异）** | **13.03** | 31.0 | 87.9% |

控制组仍产生 13.0/255 的平均差异，证明**先验面本身**就显著改变了渲染结果，
而不是网格细分的副产物。

## 四、产物

| 文件 | 说明 |
| --- | --- |
| `priors_before_control_after.jpg` | 四行对比图：修复前 / 控制组 / 修复后 / 差异热图（帧 973、1200、1596） |
| `diff_priors_isolated.jpg` | 控制实验差异条（BEFORE / AFTER / DIFF） |
| `walkthrough_before_priors_deleted.mp4` | 修复前（先验被删）漫游 |
| `walkthrough_control_priors_stripped.mp4` | 控制组（同网格、剥掉先验） |
| `walkthrough_after_priors_present.mp4` | 修复后（先验完整）漫游 |
| `p0_diff_stats_before_vs_after.json` | 修复前 vs 修复后的原始指标（含网格细分差异，非纯先验效应） |
| `p0_diff_stats_isolated.json` | 控制实验原始指标（仅先验面为唯一变量） |

三版漫游均由 `tools/render_trajectory_video.py` 从同一 SLAM 位姿渲染
（960×540@29.97，1799 帧，`--no-audio`）。

## 五、附注：交接文档数字口径

`HANDOVER.md` 记录"生产地板先验面只剩 0.7 m²、A 路线 89.7 m²"，本次复核
**两个数字均精确复现**（0.70 / 89.70 m²），缺陷判断成立。需修正的是：

- 生产中丢失的是**全部先验面**（不只是地板大 quad）；
- A 路线的 89.7 m² 是**重复计数**（44.59 m² 的真实先验被发射两遍），
  诚实的先验地板面积是 **44.59 m²**，真实修复倍数是 ~64× 而非 ~128×。

## 六、复现

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate vbr
export CUDA_VISIBLE_DEVICES=7
python -m pytest tests/ -q                                   # 71 passed
python -m tools.rebuild_geometry_from_background \
    --output-dir outputs/001_sam31_slam --config configs/vggt_slam.yaml
python -c "import json;print(json.load(open('outputs/001_sam31_slam/geometry_report.json'))['prior_survival'])"
```