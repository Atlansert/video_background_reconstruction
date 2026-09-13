"""Inference wrapper for EffectErase (FudanCVL/EffectErase, CC BY-NC 4.0).

Runs inside the `effecterase` conda env (the vendored upstream repo at
external/effecterase is pip-installed there, providing `diffsynth`). This
mirrors upstream examples/remove_wan/infer_remove_wan.py with the operational
fixes the VBR chunked pipeline needs:

- frames are read sequentially instead of per-frame seeking (chunk videos are
  mp4v-written by cv2 and exactly `--num_frames` long);
- the object reference crop is taken from the first mask frame that actually
  contains foreground — upstream crashes on an empty first-frame mask, which
  routinely happens on chunk boundaries;
- model paths, resolution, frame count and sampler settings come from the
  adapter instead of being hardcoded in script/test_remove.sh.

Heavy imports stay inside main() so the pure helpers can be unit-tested in
the `vbr` environment, where diffsynth is not installed.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def build_parser():
    parser = argparse.ArgumentParser(description="EffectErase object-removal inference (VBR wrapper).")
    parser.add_argument("--fg_bg_path", type=str, required=True, help="Input (foreground) video.")
    parser.add_argument("--mask_path", type=str, required=True, help="Mask video, white = remove.")
    parser.add_argument("--output_path", type=str, required=True, help="Output background video.")
    parser.add_argument("--model_dir", type=str, required=True, help="Wan-AI/Wan2.1-Fun-1.3B-InP directory.")
    parser.add_argument("--lora_path", type=str, required=True, help="EffectErase.ckpt path.")
    parser.add_argument("--num_frames", type=int, default=81, help="Number of frames to process.")
    parser.add_argument("--height", type=int, default=544, help="Model frame height (multiple of 16).")
    parser.add_argument("--width", type=int, default=960, help="Model frame width (multiple of 16).")
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--cfg", type=float, default=1.0)
    parser.add_argument("--lora_alpha", type=float, default=1.0)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--tiled", action="store_true", help="Enable VAE tiling.")
    return parser


def read_video_frames_sequential(video_path, num_frames, height, width):
    """Read the first `num_frames` frames sequentially; resize + normalize to [-1,1]."""
    import torchvision

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    frames = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        if len(frames) == num_frames:
            break
    capture.release()
    if len(frames) < num_frames:
        raise RuntimeError(
            f"Video {video_path} has only {len(frames)} frames, expected {num_frames}"
        )
    tensors = []
    for frame in frames:
        frame = torchvision.transforms.functional.resize(
            frame,
            (height, width),
            interpolation=torchvision.transforms.InterpolationMode.BILINEAR,
        )
        tensor = torchvision.transforms.functional.to_tensor(frame)  # [0,255] -> [0,1]
        tensor = torchvision.transforms.functional.normalize(
            tensor, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]
        )  # -> [-1,1]
        tensors.append(tensor)
    return frames, tensors


def first_visible_mask_frame(mask_frames):
    """Return the index of the first frame whose mask has foreground pixels."""
    for index, frame in enumerate(mask_frames):
        array = np.array(frame)
        if array.ndim == 3:
            array = array.max(axis=-1)
        if (array > 0).any():
            return index
    return -1


def crop_square_from_pil(mask_img, fg_bg_img, target_size=224):
    """Square crop around the mask bbox, frame content zeroed outside the mask
    (upstream `crop_square_from_pil`, which hard-fails on empty masks)."""
    mask_np = np.array(mask_img)
    if mask_np.ndim == 3:
        mask_np = mask_np.max(axis=-1)
    mask_np = (mask_np > 0).astype(np.uint8)
    img_np = np.array(fg_bg_img.convert("RGB"))
    height, width = mask_np.shape
    ys, xs = np.where(mask_np > 0)
    if len(xs) == 0:
        raise ValueError("mask frame has no foreground pixels")
    x0, x1 = xs.min(), xs.max() + 1
    y0, y1 = ys.min(), ys.max() + 1
    side = max(x1 - x0, y1 - y0)
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    sx0, sy0 = int(np.floor(cx - side / 2)), int(np.floor(cy - side / 2))
    sx1, sy1 = sx0 + side, sy0 + side
    pad_left, pad_top = max(0, -sx0), max(0, -sy0)
    pad_right, pad_bottom = max(0, sx1 - width), max(0, sy1 - height)
    if pad_left or pad_top or pad_right or pad_bottom:
        img_np = np.pad(
            img_np, ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
            mode="constant", constant_values=0,
        )
        mask_np = np.pad(
            mask_np, ((pad_top, pad_bottom), (pad_left, pad_right)),
            mode="constant", constant_values=0,
        )
        sx0, sx1 = sx0 + pad_left, sx1 + pad_left
        sy0, sy1 = sy0 + pad_top, sy1 + pad_top
    crop_img = img_np[sy0:sy1, sx0:sx1] * mask_np[sy0:sy1, sx0:sx1][..., None]
    crop = torch_compat_crop(crop_img, target_size)
    return crop


def torch_compat_crop(crop_img, target_size):
    """Bilinear-resize the crop to target_size and normalize to [-1,1] ([C,S,S])."""
    import torch
    import torch.nn.functional as F

    tensor = torch.from_numpy(crop_img).permute(2, 0, 1).float().unsqueeze(0)
    tensor = F.interpolate(
        tensor, size=(target_size, target_size), mode="bilinear", align_corners=False
    )[0]
    tensor = tensor / 255.0 * 2.0 - 1.0
    return tensor


def dummy_reference_crop(fg_bg_img, target_size=224):
    """Empty-chunk fallback: small center patch as the (no-op) object reference."""
    img_np = np.array(fg_bg_img.convert("RGB"))
    height, width = img_np.shape[:2]
    side = max(16, min(height, width) // 16)
    cy, cx = height // 2, width // 2
    patch = img_np[cy - side // 2 : cy + side // 2, cx - side // 2 : cx + side // 2]
    return torch_compat_crop(patch, target_size)


def save_video(frames, output_path, fps):
    import imageio

    frames = np.asarray(frames)
    if frames.ndim == 3:
        frames = frames[..., None]
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(
        output_path.as_posix(), fps=fps, codec="libx264", format="FFMPEG",
        ffmpeg_params=["-pix_fmt", "yuv420p", "-movflags", "+faststart"], quality=8,
    )
    try:
        for frame in frames:
            writer.append_data(np.clip(frame, 0, 255).astype(np.uint8))
    finally:
        writer.close()
    return str(output_path)


def main():
    args = build_parser().parse_args()

    import torch
    from einops import rearrange

    from diffsynth import ModelManager, WanRemovePipeline

    fg_frames, fg_tensors = read_video_frames_sequential(
        args.fg_bg_path, args.num_frames, args.height, args.width
    )
    mask_frames, mask_tensors = read_video_frames_sequential(
        args.mask_path, args.num_frames, args.height, args.width
    )

    visible = first_visible_mask_frame(mask_frames)
    if visible >= 0:
        reference = crop_square_from_pil(mask_frames[visible], fg_frames[visible])
    else:
        reference = dummy_reference_crop(fg_frames[0])

    model_dir = Path(args.model_dir)
    print("[INFO] Building model...")
    model_manager = ModelManager(device="cuda")
    model_manager.load_models(
        [
            str(model_dir / "diffusion_pytorch_model.safetensors"),
            str(model_dir / "models_t5_umt5-xxl-enc-bf16.pth"),
            str(model_dir / "Wan2.1_VAE.pth"),
            str(model_dir / "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"),
        ],
        torch_dtype=torch.bfloat16,
    )
    model_manager.load_lora_v2(args.lora_path, lora_alpha=args.lora_alpha)
    pipe = WanRemovePipeline.from_model_manager(model_manager, torch_dtype=torch.bfloat16, device="cuda")
    pipe.enable_vram_management(num_persistent_param_in_dit=6 * 10**9)

    mask_video = rearrange(torch.stack(mask_tensors), "T C H W -> C T H W").to("cuda")
    fg_video = rearrange(torch.stack(fg_tensors), "T C H W -> C T H W").to("cuda")
    reference = reference.to("cuda")

    remove_prompt = (
        "Remove the specified object and all related effects, then restore a clean background."
    )
    negative_prompt = (
        "细节模糊不清，字幕，作品，画作，画面，静止，最差质量，低质量，JPEG压缩残留，"
        "丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，"
        "形态畸形的肢体，手指融合，杂乱的背景，三条腿，背景人很多，倒着走"
    )

    print("[INFO] Running inference...")
    with torch.inference_mode(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
        remove_video, _ = pipe(
            video_mask=mask_video,
            video_fg_bg=fg_video,
            video_bg=None,
            task="remove",
            fg_first_img=reference,
            prompt_remove=remove_prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=args.num_inference_steps,
            cfg_scale=args.cfg,
            seed=args.seed,
            tiled=args.tiled,
            height=args.height,
            width=args.width,
        )

    if hasattr(remove_video, "detach"):
        remove_video = remove_video.detach().cpu().numpy()

    capture = cv2.VideoCapture(str(args.fg_bg_path))
    fps = capture.get(cv2.CAP_PROP_FPS) or 24.0
    capture.release()
    save_video(remove_video, args.output_path, fps)
    print("[INFO] All done!")


if __name__ == "__main__":
    main()
