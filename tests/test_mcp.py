"""mcp：子进程管道喂 JSON-RPC，逐行断言（离线，mock 后端）。"""
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NOPE_REG = str(ROOT / "tests" / "_no_such_registry.json")


def _env():
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src")
    env["LLM_ODDS_REGISTRY"] = NOPE_REG
    return env


def _mcp_raw(text):
    p = subprocess.run([sys.executable, "-m", "llm_odds", "mcp"],
                       input=text, capture_output=True, text=True,
                       env=_env(), timeout=60, cwd=str(ROOT))
    lines = [json.loads(x) for x in p.stdout.splitlines() if x.strip()]
    return p, lines


def _mcp(msgs):
    return _mcp_raw("".join(json.dumps(m) + "\n" for m in msgs))


def test_mcp_handshake_list_ping():
    p, out = _mcp([
        {"jsonrpc": "2.0", "id": 1, "method": "initialize",
         "params": {"protocolVersion": "2024-11-05", "capabilities": {},
                    "clientInfo": {"name": "t", "version": "0"}}},
        {"jsonrpc": "2.0", "method": "notifications/initialized"},
        {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
        {"jsonrpc": "2.0", "id": 3, "method": "ping"},
    ])
    assert p.returncode == 0, p.stderr
    assert len(out) == 3  # 通知不回
    init, tools, pong = out
    assert init["id"] == 1
    assert init["result"]["serverInfo"]["name"] == "llm-odds"
    assert init["result"]["protocolVersion"] == "2024-11-05"
    assert "tools" in init["result"]["capabilities"]
    assert {t["name"] for t in tools["result"]["tools"]} == \
        {"ask", "ask_many", "models", "serve"}
    assert pong["id"] == 3 and pong["result"] == {}


def test_mcp_tools_call_ask():
    p, out = _mcp([
        {"jsonrpc": "2.0", "id": 1, "method": "tools/call",
         "params": {"name": "ask", "arguments": {
             "v": 1, "task": "choice", "evidence": "e",
             "options": ["a", "b"], "params": {"backend": "mock"}}}},
    ])
    assert p.returncode == 0, p.stderr
    res = out[0]["result"]
    assert res["isError"] is False
    payload = json.loads(res["content"][0]["text"])
    assert payload["ok"] is True
    assert payload["model"]["backend"] == "mock"


def test_mcp_tools_call_ask_many():
    p, out = _mcp([
        {"jsonrpc": "2.0", "id": 1, "method": "tools/call",
         "params": {"name": "ask_many", "arguments": {
             "items": [{"v": 1, "task": "noul", "evidence": "e"}],
             "params": {"backend": "mock",
                        "mock_probs": {"true": 1, "false": 0}}}}},
    ])
    payload = json.loads(out[0]["result"]["content"][0]["text"])
    assert payload["results"][0]["ok"] is True
    assert payload["results"][0]["noul"] == 1.0


def test_mcp_tools_call_error_sets_iserror():
    p, out = _mcp([
        {"jsonrpc": "2.0", "id": 1, "method": "tools/call",
         "params": {"name": "ask", "arguments": {"v": 1, "task": "nope"}}},
    ])
    res = out[0]["result"]
    assert res["isError"] is True
    payload = json.loads(res["content"][0]["text"])
    assert payload["error"]["code"] == "bad_request"


def test_mcp_unknown_method_and_tool():
    p, out = _mcp([
        {"jsonrpc": "2.0", "id": 1, "method": "no/such"},
        {"jsonrpc": "2.0", "id": 2, "method": "tools/call",
         "params": {"name": "nope", "arguments": {}}},
        {"jsonrpc": "2.0", "id": 3, "method": "tools/call",
         "params": {"name": "serve", "arguments": {"action": "dance"}}},
    ])
    assert out[0]["error"]["code"] == -32601
    assert out[1]["error"]["code"] == -32602
    res3 = out[2]["result"]  # 工具内部错误 → isError，而不是 JSON-RPC error
    assert res3["isError"] is True
    assert json.loads(res3["content"][0]["text"])["error"]["code"] == \
        "bad_request"


def test_mcp_malformed_line():
    p, out = _mcp_raw("{oops\n"
                      '{"jsonrpc": "2.0", "id": 1, "method": "ping"}\n')
    assert out[0]["error"]["code"] == -32700
    assert out[1]["id"] == 1 and out[1]["result"] == {}


def test_mcp_stdout_only_protocol_lines():
    # 任何一行 stdout 都必须是合法 JSON-RPC（人话必须走 stderr）
    p, out = _mcp([{"jsonrpc": "2.0", "id": 1, "method": "ping"}])
    for obj in out:
        assert obj.get("jsonrpc") == "2.0"
