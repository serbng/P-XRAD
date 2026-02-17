from __future__ import annotations
from typing import Dict, Any

def dump_yaml(data: Dict[str, Any], path: str) -> None:
    try:
        import yaml
    except ImportError as e:
        raise ImportError("PyYAML is required to write YAML. Install with `pip install pyyaml`.") from e

    class _SmartDumper(yaml.SafeDumper):
        pass

    def is_scalar(x: Any) -> bool:
        return isinstance(x, (int, float, bool, str))

    def list_representer(dumper: yaml.Dumper, seq: list):
        # 1) Vec2 / Vec3: [a, b] or [a, b, c] -> flow
        if len(seq) in (2, 3) and all(is_scalar(v) for v in seq):
            return dumper.represent_sequence("tag:yaml.org,2002:seq", seq, flow_style=True)

        # 2) 2D matrices: list of lists of scalars -> outer block, inner flow
        if seq and all(isinstance(row, list) and all(is_scalar(v) for v in row) for row in seq):
            node = yaml.SequenceNode(tag="tag:yaml.org,2002:seq", value=[], flow_style=False)
            for row in seq:
                row_node = dumper.represent_sequence("tag:yaml.org,2002:seq", row, flow_style=True)
                node.value.append(row_node)
            return node

        # 3) Default: block (PyYAML style standard for lists)
        return dumper.represent_sequence("tag:yaml.org,2002:seq", seq, flow_style=False)

    _SmartDumper.add_representer(list, list_representer)

    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(
            data,
            f,
            Dumper=_SmartDumper,
            sort_keys=False,
            default_flow_style=False,
            width=120,
            indent=2,
        )


def load_yaml(path: str) -> Dict[str, Any]:
    try:
        import yaml
    except ImportError as e:
        raise ImportError("PyYAML is required to read YAML. Install with `pip install pyyaml`.") from e
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError("YAML root must be a mapping (dict).")
    return data
