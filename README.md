# Video Background Reconstruction

从单目室内视频恢复接近空房间的背景。当前生产路径按"纯背景"语义移除全部家具——包括冰箱、厨房柜体/橱柜、水槽等固定家具——只保留建筑结构：墙、地板、天花板、门窗和楼梯。

## Pipeline

1. 按 `sampling.segmentation_stride` 抽取语义关键帧，同时保留全部视频帧。
2. SAM 3.1 使用开放词汇提示在关键帧检测前景，并用 `preserve_prompts` 从移除掩码中扣回固定结构。
3. SAM2.1 将关键帧联合掩码传播到全部视频帧；封闭孔洞后处理用于覆盖靠垫等家具内部物体。
4. VGGT 预测相机位姿、内参、深度和置信度，在像素反投影之前排除 SAM 前景，生成彩色背景点云。
5. 从相机姿态估计重力方向，RANSAC 拟合水平/竖直平面；墙面补到地板和天花板之间，并补全地板、天花板。若单目深度的墙面不满足稳定 3D RANSAC，则使用鲁棒水平足迹生成四面贯通墙作为显式后备先验，并在报告中标记来源。
6. 掩码深度通过 TSDF 融合（失败时回退 Poisson），与结构先验网格合并。
7. ProPainter 根据全部逐帧掩码完成背景视频，最终恢复原分辨率、帧率和音轨。

该流程不会把“生成了文件”当作成功：全零掩码、缺失帧、未过滤任何 VGGT 前景点或空点云都会直接终止并写入 `pipeline_status.json`。

## Environments

项目使用三个环境是为了解开上游版本冲突：

- `vbr`：主调度、OpenCV、Open3D、TSDF/Poisson、Plotly。
- `vbr-seg`：Python 3.12、PyTorch 2.10 + CUDA 12.8、SAM 3.1、SAM2 CUDA 扩展、ProPainter。
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

## Outputs

- `mask_overlay.mp4`：红色叠加的全帧前景掩码，用于质量验收。
- `pointcloud_background.ply`：前景过滤后的彩色背景点云。
- `background_mesh.ply`：TSDF/Poisson 表面与房间结构先验合并网格。
- `structural_planes.ply`：单独的墙、地板和天花板先验网格。
- `background_scene.glb`：通用三维场景文件。
- `interactive.html`：内嵌 Plotly、无需联网的交互场景和相机轨迹。
- `background_video.mp4`：ProPainter 完成的视频，包含原音轨。
- `pipeline_status.json`：各阶段状态和关键质量指标。
- `logs/`：SAM3.1、SAM2、VGGT 和 ProPainter 完整日志。

## Prompt Interface

当前没有可用 GPT/VLM 端点，因此使用 `configs/default.yaml` 中经视频抽样验证的提示词。`gpt` 配置段保留给后续“每 4/8 帧拼图后自动描述前景”的提供器；更换描述来源只需要更新 `segmentation.prompts`，不会改变 SAM3.1、SAM2 或三维重建接口。

## WorldAct Reference

[WorldAct (arXiv:2605.15843)](https://arxiv.org/abs/2605.15843) 的输入是已经生成的整体 3DGS 场景，与本项目的单目视频输入不同，因此不替换当前 Pipeline。可以借鉴的部分有：从稀疏轨迹帧由多模态模型生成对象级提示、把多视角对象 mask 融合进 3D 后再投影以提高完整性，以及用视频扩散修复后将新内容按预测深度提升回 3D。当前项目已预留第一项的 GPT/VLM 接口；后两项可作为后续质量升级，现版本继续使用 SAM2 时序传播、ProPainter 和墙/地/顶平面先验。

## Known Limitation

单目视频中被家具全程遮挡的真实纹理没有观测值，任何方法都只能推断而不能精确还原。当前 `background_video.mp4` 会优先保证前景被移除和时间连续性，永久遮挡的大区域可能呈现平滑或模糊的生成纹理；`background_scene.glb` 与 `structural_planes.ply` 则用显式墙/地/顶先验保证结构完整性。若后续需要照片级未知区域，可在保留本 Pipeline 的前提下，把 WorldAct 使用的 3D mask 重投影和视频扩散修复作为可选增强阶段。

ProPainter 代码及模型采用其上游的非商业许可；商业使用前需单独核对该许可。
