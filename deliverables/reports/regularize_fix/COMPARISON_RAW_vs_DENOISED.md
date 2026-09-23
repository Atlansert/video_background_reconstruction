# 当前正则化在两套网格上的对比视频（RAW / DENOISED）

生成日期：2026-09-23 · 工具 `tools/regularize_planes.py` @ `f82ebd3`（当前版本）

## 一、两套链路

| 链路 | 输入网格 | 正则化输出 |
| --- | --- | --- |
| **RAW** | `background_mesh_tsdf_decimated.ply`（TSDF 抽稀，未去噪，45 万三角） | `/tmp/reg_on_raw.ply` |
| **DENOISED** | `background_mesh_denoised_decimated.ply`（去噪后抽稀，44.4 万三角） | `background_mesh_denoised_reg.ply` |

两套都用**当前默认参数**：`--band 0.10 --max-shift 0.04 --iterations 4`
（`--field-smooth-rounds` 默认 1），`--npz outputs/vggt_slam_baseline/slam/points_background.npz`。

## 二、产物

| 文件 | 说明 |
| --- | --- |
| `compare_regularize_on_RAW.mp4` | 左：RAW（未正则化）｜右：RAW + 当前正则化 |
| `compare_regularize_on_DENOISED.mp4` | 左：DENOISED（未正则化）｜右：DENOISED + 当前正则化 |
| `walkthrough_RAW_regularized.mp4` | RAW + 当前正则化的完整单画面漫游 |
| `walkthrough_DENOISED_regularized.mp4` | DENOISED + 当前正则化的完整单画面漫游 |

> 早期本目录还放过 `compare_regularize_2panel.mp4` / `compare_regularize_3panel.mp4` /
> `reg_shipped_vs_fixed.jpg`。它们由**过时代码**产生（旧 ramp，或只修法线的中间版本），
> 两栏都与当前工具不一致，已删除；GitHub Release 上的同名 asset 也已一并替换。

全部 1280×422（对比版）/ 960×540（单画面）、1799 帧 60s、h264+aac 含原音轨、
faststart（`moov` 在前，网页可边下边播）。由同一条 SLAM 相机轨迹渲染，
均启用天花板钳制（377 帧），故两版渲染口径一致。

## 三、实测效果（16 帧、未压缩渲染、按基线平滑度分组）

| 链路 | 墙面 RMS | 平整区域 | 粗糙区域 | 判定 |
| --- | --- | --- | --- | --- |
| **RAW** | 17.25 → **12.39 mm** | 0.9521 → 0.9539（**+0.0018**） | 13.5382 → 13.4311（−0.1071） | 粗糙改善；平整 +0.0018 在噪声量级 |
| **DENOISED** | 18.02 → **11.82 mm** | 0.8421 → **0.8184**（−0.0236） | 13.3077 → 13.1690（−0.1387） | 两者都改善 |

- 墙面 RMS：最大平面内点到拟合平面的 RMS 残差（越低越平），两套都降约 **28–34%**。
- 平整/粗糙：按**未正则化渲染**的分块局部对比度分成两组。平整区域是用户投诉的对象，
  要求不得变差；粗糙区域是正则化本来要处理的对象。

**DENOISED 链路的平整区域明确改善（−0.0236）**；**RAW 链路的平整区域 +0.0018**，
仍在噪声量级、未达到严格不变差。

## 四、看视频时的注意点

1. 两侧是**同一条相机轨迹、同一光照**的渲染，唯一变量是顶点位置（渲染器会重算法线，
   所以工具写的法线字段不影响画面）。
2. 差异集中在**大面积墙面的掠射角区域**（帧约 1450–1650 的高机位段最明显）；
   家具与门窗边缘两侧一致。
3. 逐帧放大看得出墙面"颗粒感"变化；整片缩略图不易分辨，建议全屏、必要时抽帧对比。
4. **不要用整帧高频能量等全局指标判断**：该指标奖励过度平滑，压低真实表面起伏即可
   刷低数值（发布版正是如此）。

## 五、复现

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate vbr
export CUDA_VISIBLE_DEVICES=7

# RAW + 正则化
python -m tools.regularize_planes \
  --mesh outputs/vggt_slam_baseline/background_mesh_tsdf_decimated.ply \
  --npz  outputs/vggt_slam_baseline/slam/points_background.npz \
  --out  outputs/vggt_slam_baseline/background_mesh_raw_reg.ply \
  --band 0.10 --max-shift 0.04 --iterations 4

# 渲染
python -m tools.render_trajectory_video --run-dir outputs/vggt_slam_baseline \
  --mesh outputs/vggt_slam_baseline/background_mesh_raw_reg.ply \
  --out  outputs/vggt_slam_baseline/trajectory_video_raw_reg.mp4
```

注：`fit_planes` 现已播种（`seed=0`），因此同一输入的重跑结果可复现。