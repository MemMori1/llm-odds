"""模型注册表：`~/.config/llm-odds/models.json`（可用 env / `--registry` 覆盖）。

三层参数覆盖里的"profile 层"就是这个文件的一条条目；
BUILTIN 是没有任何注册表时的兜底（llama-server @ 127.0.0.1:8090，
与 hsbg-helper 的现状一致）。
"""
import json
import os
from pathlib import Path
from typing import Optional

from .calibrate import default_temperature
from .errors import BadRequest, UnknownProfile
from .protocol import LETTERS
from .server import DEFAULT_PORT, size_key

DEFAULT_REGISTRY_PATH = Path.home() / ".config" / "llm-odds" / "models.json"
ENV_VAR = "LLM_ODDS_REGISTRY"

#: 没有任何注册表时的内建默认档位
BUILTIN = {
    "backend": "llama-server",
    # url 留空：由 api 层按 port 拼（profile 只写 port 时不会被打回 8090）
    "url": "",
    "port": DEFAULT_PORT,
    "context": 8192,
    "top_k": 40,
    "timeout_s": 60.0,
    "retries": 1,
    "max_options": len(LETTERS),
    "option_prefix": True,
    "include_raw": False,
    "temperature": None,
    "template": "qwen3.5-think-off",
    "system": None,
    "letter_floor_margin": 2.0,
    "cache_ttl_s": 0.0,
    "cache_prompt": False,
}


def registry_path(explicit: Optional[str] = None) -> Path:
    if explicit:
        return Path(explicit).expanduser()
    env = os.environ.get(ENV_VAR)
    if env:
        return Path(env).expanduser()
    return DEFAULT_REGISTRY_PATH


def load_registry(explicit: Optional[str] = None) -> dict:
    """读注册表；文件不存在 → {}（用内建默认档位）。坏 JSON 要炸（别静默）。"""
    p = registry_path(explicit)
    if not p.exists():
        return {}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except ValueError as e:
        raise BadRequest("注册表不是合法 JSON（%s）：%s" % (p, e))
    if not isinstance(data, dict):
        raise BadRequest("注册表必须是 JSON 对象（%s）" % p)
    models = data.get("models")
    if models is not None and not isinstance(models, dict):
        raise BadRequest("注册表 models 必须是对象（%s）" % p)
    return data


def default_name(reg: dict) -> Optional[str]:
    name = reg.get("default")
    return str(name) if name else None


def get_profile(reg: dict, name: Optional[str]) -> dict:
    """返回 BUILTIN 合并后的 profile dict；name=None 时给内建默认。"""
    if not name:
        return dict(BUILTIN)
    models = reg.get("models") or {}
    entry = models.get(name)
    if entry is None:
        raise UnknownProfile("注册表里没有模型档位 %r（%s）"
                             % (name, registry_path()))
    if not isinstance(entry, dict):
        raise BadRequest("注册表条目 %r 必须是对象" % name)
    merged = dict(BUILTIN)
    merged.update({k: v for k, v in entry.items() if v is not None})
    return merged


def resolve_temperature(prof: dict) -> float:
    """profile 里没写 temperature 时，按 GGUF/档位名查校准表。"""
    t = prof.get("temperature")
    if t is not None:
        return float(t)
    size = str(prof.get("size") or "")
    if not size and prof.get("gguf"):
        size = size_key(str(prof.get("gguf")))
    return default_temperature(size)
