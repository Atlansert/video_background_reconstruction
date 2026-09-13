"""VideoPainter inference driver (runs inside the dedicated `videopainter` conda env).

Loads TencentARC/VideoPainter's CogVideoXI2VDualInpaintAnyLPipeline (CogVideoX-5b-I2V
+ context-control branch + ID-resample LoRA) and inpaints the masked regions of a
frame directory. The pipeline windows the clip internally (window_frames windows,
window_stride step, latent-space averaging + previous-window conditioning), so the
whole downsampled clip goes through in ONE call — the model loads once for the
entire video, unlike the SVOR per-chunk wrapper.

Input/output conventions mirror vbr.models.svor: frames are `NNNNNN.jpg`, masks are
`NNNNNN.png` grayscale with white = region to inpaint. Frames are temporally
downsampled by `down_sample_stride` before the model (CogVideoX is an 8fps model);
the output video is written at `source_fps / down_sample_stride` so its duration
matches the source. The caller (VideoPainterAdapter) rescales to source fps/size.

Run via: conda run -p <vp-env> python -m vbr.videopainter_driver ... (cwd = project root)
"""

import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from vbr.models.videopainter import window_pad_count


def main():
    parser = argparse.ArgumentParser(description="VideoPainter inpainting driver")
    parser.add_argument("--frames_dir", required=True)
    parser.add_argument("--masks_dir", required=True)
    parser.add_argument("--output", required=True, help="output mp4 (mp4v, model fps)")
    parser.add_argument("--report", required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--branch_path", required=True)
    parser.add_argument("--id_adapter_path", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--source_fps", type=float, required=True)
    parser.add_argument("--down_sample_stride", type=int, default=3)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=720)
    parser.add_argument("--window_frames", type=int, default=49)
    parser.add_argument("--window_stride", type=int, default=49)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--guidance_scale", type=float, default=6.0)
    parser.add_argument("--prev_clip_weight", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mask_dilate_px", type=int, default=16, help="dilation at source resolution")
    parser.add_argument(
        "--first_frame_fill",
        default="telea",
        choices=["telea", "none"],
        help="fill the first frame's mask before conditioning (the repo's pipeline uses FLUX.1-Fill "
        "here; a black-holed I2V conditioning frame anchors black content in the generation)",
    )
    parser.add_argument(
        "--fill_mode",
        default="telea",
        choices=["telea", "black"],
        help="content of the `video` kwarg inside the fill region. black = repo default; telea = "
        "classical pre-fill, which keeps the branch conditioning identical (the pipeline re-blacks "
        "the fill internally) while giving replace_gt uncontaminated latents — encoding black "
        "holes darkens latents around the fill and replace_gt keeps them as a black boundary ring",
    )
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16"])
    parser.add_argument("--no_dynamic_cfg", action="store_true", help="disable dynamic CFG")
    args = parser.parse_args()

    frame_paths = sorted(Path(args.frames_dir).glob("*.jpg"), key=lambda p: int(p.stem))
    if not frame_paths:
        raise RuntimeError(f"No frames found in {args.frames_dir}")
    mask_paths = sorted(Path(args.masks_dir).glob("*.png"), key=lambda p: int(p.stem))
    if len(mask_paths) != len(frame_paths):
        raise RuntimeError(
            f"Mask/frame count mismatch: {len(mask_paths)} masks vs {len(frame_paths)} frames"
        )
    total = len(frame_paths)
    stride = max(1, int(args.down_sample_stride))
    indices = list(range(0, total, stride))

    def load_mask(path):
        mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise RuntimeError(f"Unreadable mask {path}")
        return mask

    frames_ds = []
    masks_ds = []
    for index in indices:
        frame = cv2.imread(str(frame_paths[index]))
        if frame is None:
            raise RuntimeError(f"Unreadable frame {frame_paths[index]}")
        frames_ds.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        masks_ds.append(load_mask(mask_paths[index]))
    height, width = frames_ds[0].shape[:2]

    if args.mask_dilate_px > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (2 * args.mask_dilate_px + 1, 2 * args.mask_dilate_px + 1),
        )
        masks_ds = [cv2.dilate(mask, kernel) for mask in masks_ds]

    pad = window_pad_count(len(frames_ds), args.window_frames, args.window_stride)
    if pad:
        frames_ds += [frames_ds[-1]] * pad
        masks_ds += [masks_ds[-1]] * pad

    masked_video = []
    binary_masks = []
    for frame, mask in zip(frames_ds, masks_ds):
        binary = (mask > 127).astype(np.uint8) * 255
        binary_3 = np.repeat(binary[:, :, None], 3, axis=2)
        masked = np.where(binary_3 > 0, 0, frame)
        if args.fill_mode == "telea" and (binary > 0).any():
            masked = cv2.inpaint(frame, binary, 7, cv2.INPAINT_TELEA)
        masked_video.append(Image.fromarray(masked))
        binary_masks.append(Image.fromarray(binary_3))
    if args.first_frame_fill == "telea":
        first_mask = np.array(binary_masks[0])[:, :, 0]
        if first_mask.sum() > 0:
            # first_frame_gt semantics: frame 0 becomes fully-conditioned ground
            # truth (mask zeroed) using a cheap classical fill in place of the
            # repo's FLUX.1-Fill step
            filled = cv2.inpaint(frames_ds[0], (first_mask > 0).astype(np.uint8) * 255, 7, cv2.INPAINT_TELEA)
            frames_ds[0] = filled
            masked_video[0] = Image.fromarray(filled)
            binary_masks[0] = Image.fromarray(np.zeros_like(np.array(binary_masks[0])))
    model_frames = len(frames_ds)
    fps_out = args.source_fps / stride
    print(
        f"total={total} stride={stride} model_frames={model_frames} pad={pad} "
        f"fps_out={fps_out:.3f} size={width}x{height}",
        flush=True,
    )

    from diffusers import (
        CogVideoXDPMScheduler,
        CogVideoXI2VDualInpaintAnyLPipeline,
        CogvideoXBranchModel,
        CogVideoXTransformer3DModel,
    )

    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    started = time.time()
    branch = CogvideoXBranchModel.from_pretrained(args.branch_path, torch_dtype=dtype)
    branch = branch.to(dtype=dtype).cuda()
    transformer = CogVideoXTransformer3DModel.from_pretrained(
        args.model_path,
        subfolder="transformer",
        torch_dtype=dtype,
        id_pool_resample_learnable=True,
    ).to(dtype=dtype).cuda()
    pipe = CogVideoXI2VDualInpaintAnyLPipeline.from_pretrained(
        args.model_path,
        branch=branch,
        transformer=transformer,
        torch_dtype=dtype,
    )
    pipe.load_lora_weights(
        args.id_adapter_path,
        weight_name="pytorch_lora_weights.safetensors",
        adapter_name="test_1",
        target_modules=["transformer"],
    )
    pipe.scheduler = CogVideoXDPMScheduler.from_config(
        pipe.scheduler.config, timestep_spacing="trailing"
    )
    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()
    pipe.to("cuda")
    print(f"pipeline loaded in {time.time() - started:.1f}s", flush=True)

    generator = torch.Generator().manual_seed(args.seed)
    started = time.time()
    result = pipe(
        prompt=args.prompt,
        image=masked_video[0],
        height=args.height,
        width=args.width,
        num_frames=args.window_frames,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        use_dynamic_cfg=not args.no_dynamic_cfg,
        num_videos_per_prompt=1,
        generator=generator,
        video=masked_video,
        masks=binary_masks,
        strength=1.0,
        replace_gt=True,
        mask_add=True,
        stride=args.window_stride,
        prev_clip_weight=args.prev_clip_weight,
        id_pool_resample_learnable=True,
        output_type="np",
    )
    generated = result.frames[0]  # (T, H, W, C) float [0, 1]
    elapsed = time.time() - started
    print(f"generation done: {generated.shape} in {elapsed:.1f}s", flush=True)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(fps_out),
        (generated.shape[2], generated.shape[1]),
    )
    for frame in generated:
        writer.write((np.clip(frame, 0.0, 1.0) * 255.0).astype(np.uint8)[:, :, ::-1])
    writer.release()

    Path(args.report).write_text(
        json.dumps(
            {
                "method": "videopainter",
                "source_frames": total,
                "down_sample_stride": stride,
                "model_frames": model_frames,
                "pad_frames": pad,
                "output_frames": int(generated.shape[0]),
                "fps_out": fps_out,
                "window_frames": args.window_frames,
                "window_stride": args.window_stride,
                "num_inference_steps": args.num_inference_steps,
                "guidance_scale": args.guidance_scale,
                "seed": args.seed,
                "load_seconds": None,
                "generate_seconds": elapsed,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
