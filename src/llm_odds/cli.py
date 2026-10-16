"""命令行：ask / ask-many / models / serve / mcp / selftest。

规矩（DESIGN.md §9.2）：

- **stdout 只出 JSON**（ask / ask-many / serve status / models --json）；
  人话全走 stderr —— 宿主可以放心 `json.loads(stdout)`。
- 退出码：`0` 成功 / `2` 契约或参数错 / `3` 模型不可用（含 logprobs 不支持）/
  `4` 超时 / `1` 内部错。
"""
import argparse
import json
import sys
from pathlib import Path

from . import __version__
from .api import ask_item, ask_many, models_info
from .errors import ERROR_EXIT, BadRequest, OddsError
from .registry import default_name, get_profile, load_registry
from .server import DEFAULT_CONTEXT, DEFAULT_PORT
from .server import start as serve_start
from .server import status as serve_status
from .server import stop as serve_stop
from .session import Session


def _read_json(path: str) -> object:
    try:
        raw = sys.stdin.read() if path == "-" else Path(path).read_text(
            encoding="utf-8")
    except OSError as e:
        raise BadRequest("读不到请求文件：%s" % e)
    try:
        return json.loads(raw)
    except ValueError as e:
        raise BadRequest("请求不是合法 JSON：%s" % e)


def _print_json(obj) -> None:
    print(json.dumps(obj, ensure_ascii=False, indent=2))


def _exit_code(resp: dict) -> int:
    if resp.get("ok"):
        return 0
    err = resp.get("error") or {}
    return ERROR_EXIT.get(err.get("code"), 1)


def _int_of(v, default: int) -> int:
    try:
        return int(v)
    except (TypeError, ValueError):
        return default


# -- 各子命令 ---------------------------------------------------------------

def _cmd_ask(args) -> int:
    req = _read_json(args.req)
    if not isinstance(req, dict):
        raise BadRequest("请求必须是 JSON 对象")
    resp = ask_item(req, registry=load_registry(args.registry),
                    session=Session())
    _print_json(resp)
    return _exit_code(resp)


def _cmd_ask_many(args) -> int:
    req = _read_json(args.req)
    if isinstance(req, list):
        items, params = req, None
    elif isinstance(req, dict):
        items, params = req.get("items"), req.get("params")
    else:
        raise BadRequest("ask-many 请求必须是 {\"items\": […]} 或数组")
    resp = ask_many(items, params=params, registry=load_registry(args.registry),
                    session=Session())
    _print_json(resp)
    for r in resp["results"]:
        if not r.get("ok"):
            return _exit_code(r)
    return 0


def _cmd_models(args) -> int:
    reg = load_registry(args.registry)
    info = models_info(registry=reg, check=not args.no_check)
    if args.json:
        _print_json(info)
        return 0
    print("注册表  %s" % info["registry"])
    default = info["default"] or "（内建）"
    print("默认    %s  %s" % (default, "✅" if info["default_available"] else "❌"))
    print("-" * 60)
    for name in sorted(info["models"]):
        m = info["models"][name]
        state = m.get("available")
        mark = "✅" if state else ("❌" if state is False else "·")
        print("%-22s %-14s %s  %s" % (name, m.get("backend", "?"),
                                      m.get("url", ""), mark))
    b = info["builtin"]
    print("%-22s %-14s %s  %s" % ("（内建默认）", b.get("backend", "?"),
                                  b.get("url", ""),
                                  "✅" if b.get("available") else "❌"))
    return 0


def _cmd_serve(args) -> int:
    reg = load_registry(args.registry)
    name = args.profile or default_name(reg) or None
    prof = get_profile(reg, name)
    port = _int_of(prof.get("port"), DEFAULT_PORT)
    context = _int_of(prof.get("context"), DEFAULT_CONTEXT)
    url = str(prof.get("url") or "")
    if args.action == "start":
        return serve_start(profile=name, server=prof.get("server_path") or None,
                           model=prof.get("gguf") or None, port=port,
                           context=context, url=url or None)
    if args.action == "stop":
        return serve_stop(profile=name)
    _print_json(serve_status(profile=name, url=url or None))
    return 0


def _cmd_selftest(args) -> int:
    from .calibrate import default_temperature
    reg = {} if args.mock else load_registry(args.registry)
    results = []  # (名字, ok, 详情)

    def check(name, fn):
        try:
            detail = fn() or ""
            results.append((name, True, str(detail)))
        except AssertionError as e:
            results.append((name, False, str(e) or "断言失败"))
        except OddsError as e:
            results.append((name, False, "[%s] %s" % (e.code, e)))
        except Exception as e:
            results.append((name, False, "%s: %s" % (type(e).__name__, e)))

    def near(a, b, eps=1e-9):
        assert abs(a - b) < eps, "期望 %r，得到 %r" % (b, a)

    def c_mock_choice():
        r = ask_item({"v": 1, "task": "choice", "evidence": "Self-test.",
                      "criterion": "Pick one option.",
                      "options": ["alpha", "beta", "gamma"],
                      "params": {"backend": "mock"}}, registry={})
        assert r["ok"], r.get("error")
        near(sum(r["probs"].values()), 1.0)
        assert len(r["ranked"]) == 3
        return "choice=%s p=%.3f" % (r["choice"], r["confidence"])

    def c_mock_probs():
        r = ask_item({"v": 1, "task": "choice", "evidence": "Self-test.",
                      "options": ["alpha", "beta", "gamma"],
                      "params": {"backend": "mock",
                                 "mock_probs": {"0": 0.5, "1": 0.3,
                                                "2": 0.2}}}, registry={})
        assert r["ok"], r.get("error")
        near(r["probs"]["0"], 0.5)
        near(r["probs"]["1"], 0.3)
        return "0:0.5 → %s" % ("精确" if r["choice"] == "0" else "不对!")

    def c_noul():
        r = ask_item({"v": 1, "task": "noul", "evidence": "Self-test.",
                      "params": {"backend": "mock",
                                 "mock_probs": {"true": 0.75,
                                                "false": 0.25}}}, registry={})
        assert r["ok"], r.get("error")
        near(r["noul"], 0.75)
        return "P(true)=0.75"

    def c_score():
        r = ask_item({"v": 1, "task": "score", "evidence": "Self-test.",
                      "options": ["lvl0", "lvl1", "lvl2", "lvl3"],
                      "params": {"backend": "mock",
                                 "mock_probs": {"0": 0.25, "1": 0.25,
                                                "2": 0.25, "3": 0.25}}},
                     registry={})
        assert r["ok"], r.get("error")
        near(r["score"], 1.5)
        return "score=1.5"

    def c_temperature():
        near(default_temperature("4b-v0.2"), 1.532)
        near(default_temperature(""), 1.42)
        return "4b-v0.2 → 1.532"

    def c_truncation():
        opts = ["o%d" % i for i in range(17)]
        r = ask_item({"v": 1, "task": "choice", "evidence": "Self-test.",
                      "options": opts,
                      "params": {"backend": "mock",
                                 "mock_probs": {str(i): 1.0
                                                for i in range(17)}}},
                     registry={})
        assert r["ok"], r.get("error")
        near(sum(r["probs"].values()), 1.0)
        assert r["metrics"]["truncated"] == 1
        assert "truncated_options" in r["warnings"]
        return "17 → 16+份额"

    check("mock 后端 choice", c_mock_choice)
    check("mock_probs 精确复现", c_mock_probs)
    check("noul", c_noul)
    check("score", c_score)
    check("校准温度表", c_temperature)
    check("截断份额 ∑p=1", c_truncation)

    if not args.mock:
        def c_real():
            r = ask_item({"v": 1, "task": "choice",
                          "evidence": "Self-test: choose the sunnier option.",
                          "criterion": "Which option is more likely to be sunny?",
                          "options": ["Sunny.", "Rainy."],
                          "params": {"include_raw": True}}, registry=reg)
            assert r["ok"], "[%s] %s" % ((r.get("error") or {}).get("code"),
                                         (r.get("error") or {}).get("message"))
            m = r["model"]
            return "%s @ %s，P(top)=%.3f" % (m["name"], m["backend"],
                                             r["confidence"])
        check("真模型一问（注册表默认档位）", c_real)

    failed = [r for r in results if not r[1]]
    for name, ok, detail in results:
        print("%s %s%s" % ("✅" if ok else "❌", name,
                           " —— %s" % detail if detail else ""))
    if failed:
        print("selftest：%d/%d 通过" % (len(results) - len(failed), len(results)))
        return 1 if args.mock else 3
    print("selftest：%d/%d 通过%s" % (len(results), len(results),
                                     "（--mock，未碰真模型）" if args.mock else ""))
    return 0


# -- 入口 -------------------------------------------------------------------

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="llm-odds",
        description="本地模型的校准概率顾问（上下文 + 选项 → 概率分布）")
    ap.add_argument("--version", action="version",
                    version="llm-odds %s" % __version__)
    sub = ap.add_subparsers(dest="cmd")

    def common(p):
        p.add_argument("--registry", metavar="PATH",
                       help="模型注册表路径（默认 $LLM_ODDS_REGISTRY 或 "
                            "~/.config/llm-odds/models.json）")
        return p

    p = common(sub.add_parser("ask", help="单题：请求 JSON 文件，或 - 从 stdin"))
    p.add_argument("req", nargs="?", default="-")
    p = common(sub.add_parser("ask-many", help="批量：{\"items\": […]}"))
    p.add_argument("req", nargs="?", default="-")
    p = common(sub.add_parser("models", help="列出注册表里的模型与可用性"))
    p.add_argument("--json", action="store_true", help="输出 JSON（机器用）")
    p.add_argument("--no-check", action="store_true", help="不探测服务可用性")
    p = common(sub.add_parser("serve", help="本地推理服务生命周期"))
    p.add_argument("action", choices=["start", "stop", "status"])
    p.add_argument("--profile", help="注册表档位名（默认注册表 default）")
    common(sub.add_parser("mcp", help="stdio MCP server"))
    p = common(sub.add_parser("selftest", help="全链路冒烟（CI 入口）"))
    p.add_argument("--mock", action="store_true", help="只跑离线部分")

    args = ap.parse_args(argv)
    if not args.cmd:
        ap.print_help(sys.stderr)
        return 2
    try:
        if args.cmd == "ask":
            return _cmd_ask(args)
        if args.cmd == "ask-many":
            return _cmd_ask_many(args)
        if args.cmd == "models":
            return _cmd_models(args)
        if args.cmd == "serve":
            return _cmd_serve(args)
        if args.cmd == "mcp":
            from .mcp import main as mcp_main
            return mcp_main()
        return _cmd_selftest(args)
    except OddsError as e:
        _print_json({"v": 1, "ok": False,
                     "error": {"code": e.code, "message": str(e)}})
        return e.exit_code
    except KeyboardInterrupt:
        return 130
    except Exception as e:
        _print_json({"v": 1, "ok": False,
                     "error": {"code": "internal",
                               "message": "%s: %s" % (type(e).__name__, e)}})
        return 1
