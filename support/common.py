import json
import math
import os
import platform
import sys
import tempfile
from datetime import date, datetime, timezone
from pathlib import Path


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def resolve_path(path):
    # Windows Python 3.7 的 resolve 对尚不存在的相对路径可能仍返回相对路径。
    return Path(path).expanduser().absolute().resolve()


def clean_json(value):
    """输出标准 JSON，未定义的统计量写为 null。"""
    if isinstance(value, dict):
        return {str(k): clean_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(v) for v in value]
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if hasattr(value, "item"):
        return clean_json(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(resolve_path(value))
    return value


def write_json(path, value, overwrite=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(clean_json(value), ensure_ascii=False, indent=2, allow_nan=False)
    if not overwrite:
        with path.open("x", encoding="utf-8") as handle:
            handle.write(payload + "\n")
        return
    fd, tmp = tempfile.mkstemp(prefix=".beer-release-", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(payload + "\n")
        os.replace(tmp, str(path))
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def runtime_info():
    from importlib import import_module
    versions = {}
    for name in ("numpy", "pandas", "PIL", "openpyxl", "pyarrow"):
        try:
            versions[name] = import_module(name).__version__
        except (ImportError, OSError, ValueError) as exc:
            versions[name] = {"unavailable": "{}: {}".format(type(exc).__name__, exc)}
    return {"python": sys.version, "platform": platform.platform(), "libraries": versions}


def protect_output(output, inputs):
    """避免报告或实验目录落入输入数据目录，或覆盖输入文件。"""
    output = resolve_path(output)
    for source in inputs:
        source = resolve_path(source)
        if output == source or (source.is_dir() and source in output.parents):
            raise ValueError("输出路径必须与原始输入分开: {}".format(source))
