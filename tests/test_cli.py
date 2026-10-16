"""cli：子进程级（stdout 只出 JSON / 退出码约定 / bash 包装器）。"""
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BIN = ROOT / "bin" / "llm-odds"
EX = ROOT / "examples"
NOPE_REG = str(ROOT / "tests" / "_no_such_registry.json")


def _env():
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src")
    env["LLM_ODDS_REGISTRY"] = NOPE_REG  # 绝不读用户真实的 models.json
    return env


def _run(args, stdin=None, use_bin=False):
    cmd = ([str(BIN)] if use_bin
           else [sys.executable, "-m", "llm_odds"]) + args
    return subprocess.run(cmd, input=stdin, capture_output=True, text=True,
                          env=_env(), timeout=60, cwd=str(ROOT))


_MOCK_REQ = {"v": 1, "task": "choice", "evidence": "e",
             "options": ["a", "b"], "params": {"backend": "mock"}}


def test_ask_stdin_mock():
    p = _run(["ask", "-"], stdin=json.dumps(_MOCK_REQ))
    assert p.returncode == 0, p.stderr
    got = json.loads(p.stdout)
    assert got["ok"] is True
    assert got["model"]["backend"] == "mock"


def test_ask_error_exit_codes():
    cases = [
        ({"v": 1, "task": "vibes", "evidence": "e"}, "bad_request"),
        ({"v": 1, "task": "choice", "evidence": "e"}, "no_options"),
        ({"v": 1, "task": "noul", "evidence": "e",
          "params": {"profile": "nope"}}, "unknown_profile"),
    ]
    for req, code in cases:
        p = _run(["ask", "-"], stdin=json.dumps(req))
        assert p.returncode == 2, (req, p.returncode, p.stdout)
        assert json.loads(p.stdout)["error"]["code"] == code


def test_ask_missing_file_is_bad_request():
    p = _run(["ask", "/nonexistent/nope.json"])
    assert p.returncode == 2
    assert json.loads(p.stdout)["error"]["code"] == "bad_request"


def test_ask_many_from_file():
    p = _run(["ask-many", str(EX / "ask-many.request.json")])
    assert p.returncode == 0, p.stderr
    got = json.loads(p.stdout)
    assert [r["ok"] for r in got["results"]] == [True, True]
    assert got["results"][1]["noul"] == 0.35


def test_models_json_offline():
    p = _run(["models", "--json", "--no-check"])
    assert p.returncode == 0, p.stderr
    got = json.loads(p.stdout)
    assert got["default"] == ""
    assert got["models"] == {}
    assert got["builtin"]["backend"] == "llama-server"


def test_selftest_mock():
    p = _run(["selftest", "--mock"])
    assert p.returncode == 0, (p.stdout, p.stderr)
    assert "通过" in p.stdout


def test_bin_wrapper_runs():
    p = _run(["selftest", "--mock"], use_bin=True)
    assert p.returncode == 0, (p.stdout, p.stderr)


def test_version():
    p = _run(["--version"])
    assert p.returncode == 0
    assert "llm-odds" in p.stdout


def test_no_command_prints_help():
    p = _run([])
    assert p.returncode == 2
