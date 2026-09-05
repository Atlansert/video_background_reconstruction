from pathlib import Path
import yaml

def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["input_video"] = str(Path(cfg["input_video"]))
    return cfg
