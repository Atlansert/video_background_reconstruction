"""Extension point for VLM-generated foreground prompt lists."""

from __future__ import annotations

import importlib
from pathlib import Path


def resolve_prompts(segmentation_cfg, gpt_cfg, keyframes):
    """Return configured prompts or call a user-supplied VLM provider.

    A provider is a dotted callable such as ``package.module:function``. It
    receives numeric keyframe paths, the collage group size, and the GPT
    configuration, then returns a list of foreground noun phrases.
    """
    configured = list(segmentation_cfg.get("prompts", []))
    if not gpt_cfg or not gpt_cfg.get("enabled", False):
        return configured
    provider_name = gpt_cfg.get("provider")
    if not provider_name or ":" not in provider_name:
        raise ValueError("gpt.provider must be a 'module:function' callable")
    module_name, function_name = provider_name.split(":", 1)
    provider = getattr(importlib.import_module(module_name), function_name)
    paths = sorted((Path(path) for path in keyframes), key=lambda path: int(path.stem))
    generated = provider(paths, int(gpt_cfg.get("group_size", 8)), dict(gpt_cfg))
    if not isinstance(generated, list) or not all(isinstance(item, str) for item in generated):
        raise TypeError("The GPT/VLM prompt provider must return list[str]")
    return list(dict.fromkeys(configured + generated))
