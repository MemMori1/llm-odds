"""stdio MCP server（手写 JSON-RPC 2.0 子集，零依赖）。

覆盖宿主需要的一切：`initialize` / `notifications/initialized`（忽略）/
`ping` / `tools/list` / `tools/call`；暂不做 resources / prompts / 采样。
**stdout 只出协议行**（逐行 JSON）—— 任何日志都走 stderr。

tools：`ask` / `ask_many` / `models` / `serve`（见 DESIGN.md §9.3）。
"""
import json
import sys
from typing import Optional

from . import __version__
from .api import ask_item, ask_many, models_info
from .errors import BadRequest, OddsError
from .registry import default_name, get_profile, load_registry
from .server import DEFAULT_CONTEXT, DEFAULT_PORT
from .server import start as serve_start
from .server import status as serve_status
from .server import stop as serve_stop
from .session import Session

PROTOCOL_VERSION = "2024-11-05"

TOOLS = [
    {
        "name": "ask",
        "description": "单题：给上下文（evidence）和一列选项，返回校准概率分布"
                       "（契约 v1，见 schemas/ask.response.v1.schema.json）。",
        "inputSchema": {
            "type": "object",
            "properties": {
                "v": {"const": 1},
                "id": {"type": "string"},
                "task": {"enum": ["choice", "noul", "score"]},
                "evidence": {"type": "string"},
                "criterion": {"type": "string"},
                "options": {"type": "array"},
                "params": {"type": "object"},
            },
            "required": ["task"],
        },
    },
    {
        "name": "ask_many",
        "description": "批量（一次进程问一串）：入参 {items, params?}，返回 "
                       "{\"v\": 1, \"results\": [...]}。",
        "inputSchema": {
            "type": "object",
            "properties": {
                "items": {"type": "array"},
                "params": {"type": "object"},
            },
            "required": ["items"],
        },
    },
    {
        "name": "models",
        "description": "列出注册表里的模型档位与可用性。",
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "serve",
        "description": "本地推理服务生命周期：start / stop / status。",
        "inputSchema": {
            "type": "object",
            "properties": {
                "action": {"enum": ["start", "stop", "status"]},
                "profile": {"type": "string"},
            },
            "required": ["action"],
        },
    },
]


def _tool_ask(args, reg, session):
    return ask_item(dict(args or {}), registry=reg, session=session)


def _tool_ask_many(args, reg, session):
    args = args or {}
    return ask_many(args.get("items"), params=args.get("params"),
                    registry=reg, session=session)


def _tool_models(args, reg, session):
    return models_info(registry=reg)


def _tool_serve(args, reg, session):
    args = args or {}
    action = args.get("action")
    name = args.get("profile") or default_name(reg) or None
    prof = get_profile(reg, name)
    url = str(prof.get("url") or "")
    if action == "start":
        rc = serve_start(
            profile=name, server=prof.get("server_path") or None,
            model=prof.get("gguf") or None,
            port=int(prof.get("port") or DEFAULT_PORT),
            context=int(prof.get("context") or DEFAULT_CONTEXT),
            url=url or None)
        st = serve_status(profile=name, url=url or None)
        st["start_rc"] = rc
        return st
    if action == "stop":
        rc = serve_stop(profile=name)
        st = serve_status(profile=name, url=url or None)
        st["stop_rc"] = rc
        return st
    if action == "status":
        return serve_status(profile=name, url=url or None)
    raise BadRequest("serve.action 必须是 start / stop / status")


CALL = {"ask": _tool_ask, "ask_many": _tool_ask_many,
        "models": _tool_models, "serve": _tool_serve}


def _result(mid, payload):
    return {"jsonrpc": "2.0", "id": mid, "result": payload}


def _error(mid, code, message):
    return {"jsonrpc": "2.0", "id": mid,
            "error": {"code": code, "message": message}}


def handle(msg, reg: dict, session: Session) -> Optional[dict]:
    """处理一条 JSON-RPC 消息；通知返回 None，其余返回要写回的对象。"""
    if not isinstance(msg, dict) or msg.get("jsonrpc") != "2.0":
        mid = msg.get("id") if isinstance(msg, dict) else None
        return _error(mid, -32600, "invalid request（需要 jsonrpc: \"2.0\"）")
    mid = msg.get("id")
    method = msg.get("method")
    if method == "initialize":
        return _result(mid, {
            "protocolVersion": PROTOCOL_VERSION,
            "capabilities": {"tools": {}},
            "serverInfo": {"name": "llm-odds", "version": __version__},
        })
    if method in ("notifications/initialized", "initialized"):
        return None
    if method == "ping":
        return _result(mid, {})
    if method == "tools/list":
        return _result(mid, {"tools": TOOLS})
    if method == "tools/call":
        params = msg.get("params") or {}
        name = params.get("name")
        fn = CALL.get(name)
        if fn is None:
            return _error(mid, -32602, "未知工具：%r" % name)
        try:
            payload = fn(params.get("arguments") or {}, reg, session)
            is_err = isinstance(payload, dict) and payload.get("ok") is False
        except OddsError as e:
            payload = {"v": 1, "ok": False,
                       "error": {"code": e.code, "message": str(e)}}
            is_err = True
        except Exception as e:
            payload = {"v": 1, "ok": False,
                       "error": {"code": "internal",
                                 "message": "%s: %s" % (type(e).__name__, e)}}
            is_err = True
        if not isinstance(payload, dict):
            payload = {"v": 1, "ok": True, "result": payload}
        return _result(mid, {
            "content": [{"type": "text",
                         "text": json.dumps(payload, ensure_ascii=False)}],
            "isError": is_err,
        })
    if "id" not in msg:
        return None  # 其它通知一律忽略
    return _error(mid, -32601, "未知方法：%r" % method)


def main(argv=None) -> int:
    reg = load_registry()
    session = Session(default_ttl=12.0)  # MCP 常驻：12s 内同题去重
    out = sys.stdout
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except ValueError:
            out.write(json.dumps(_error(None, -32700, "JSON 解析失败"),
                                 ensure_ascii=False) + "\n")
            out.flush()
            continue
        resp = handle(msg, reg, session)
        if resp is not None:
            out.write(json.dumps(resp, ensure_ascii=False) + "\n")
            out.flush()
    return 0
