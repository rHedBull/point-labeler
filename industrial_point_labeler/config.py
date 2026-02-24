"""Load and validate class configuration."""

from pathlib import Path

import yaml

DEFAULT_CONFIG = Path(__file__).resolve().parent.parent / "config" / "classes.yaml"


def load_classes(config_path=None):
    """Load class definitions from YAML.

    Returns (class_map, class_colors, class_keys, process_classes).

    class_map: {"pipe": 0, "tank": 1, ...}
    class_colors: {"pipe": [0.15, 0.8, 0.15], ...}
    class_keys: {"1": "pipe", "2": "tank", ...}
    process_classes: {"pipe", "tank", "equipment"}
    """
    path = Path(config_path) if config_path else DEFAULT_CONFIG
    if not path.exists():
        raise FileNotFoundError(f"Classes config not found: {path}")

    with open(path) as f:
        cfg = yaml.safe_load(f)

    classes = cfg.get("classes", {})
    class_map = {}
    class_colors = {}
    class_keys = {}

    for i, (name, props) in enumerate(classes.items()):
        class_map[name] = i
        class_colors[name] = props.get("color", [0.5, 0.5, 0.5])
        key = props.get("key")
        if key:
            class_keys[str(key)] = name

    process_classes = set(cfg.get("process_classes", []))

    return class_map, class_colors, class_keys, process_classes
