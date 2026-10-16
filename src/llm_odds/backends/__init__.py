"""后端工厂：core 通过 `make_backend(params)` 拿到实现。

`core/` 不许 import 具体后端之外的任何网络细节 —— 这里就是唯一的装配点。
"""
from ..errors import BadRequest
from ..types import Params
from .base import Backend, LetterResult
from .llama_server import LlamaServerBackend
from .mock import MockBackend
from .openai_compat import OpenAICompatBackend

#: 后端名 → 类
BACKENDS = {
    "llama-server": LlamaServerBackend,
    "openai": OpenAICompatBackend,
    "mock": MockBackend,
}


def make_backend(params: Params) -> Backend:
    cls = BACKENDS.get(params.backend)
    if cls is None:
        raise BadRequest("未知后端：%r（可选 %s）"
                         % (params.backend, "、".join(sorted(BACKENDS))))
    return cls(params)


__all__ = ["Backend", "LetterResult", "BACKENDS", "make_backend",
           "LlamaServerBackend", "OpenAICompatBackend", "MockBackend"]
