"""server.py：size_key / 档位偏好 / 找文件 / PID 路径与状态（全部离线）。"""
import os
import tempfile
from pathlib import Path

from llm_odds import server as sv


def test_size_key_variants():
    assert sv.size_key("/x/jevk5-4b-v0.2-Q4_K_M.gguf") == "4b-v0.2"
    assert sv.size_key("jevk5-4b-Q4_K_M.gguf") == "4b"
    assert sv.size_key("JevK5-9B-v0.3-Q8_0.gguf") == "9b-v0.3"
    assert sv.size_key("no-size-here.gguf") == ""
    assert sv.size_key("") == ""


def test_size_preference_4b_over_9b():
    # 官方实测：4B > 9B（9B 校准更差）；2B 只是 4B 缺失时的退路
    assert sv._size_rank("jevk5-4b-q4_k_m.gguf") > \
        sv._size_rank("jevk5-9b-q8_0.gguf")
    assert sv._size_rank("jevk5-2b-q8_0.gguf") > \
        sv._size_rank("jevk5-9b-q8_0.gguf")


def test_find_gguf_prefers_size_then_quant():
    with tempfile.TemporaryDirectory() as d:
        names = ["jevk5-9b-q8_0.gguf", "jevk5-4b-q4_k_m.gguf",
                 "jevk5-4b-q8_0.gguf"]
        for n in names:
            (Path(d) / n).write_bytes(b"x")
        got = sv.find_gguf(str(Path(d) / "*.gguf"))
        assert got is not None
        assert Path(got).name == "jevk5-4b-q8_0.gguf"


def test_find_gguf_none_when_empty():
    with tempfile.TemporaryDirectory() as d:
        assert sv.find_gguf(str(Path(d) / "*.gguf")) is None


def test_find_llama_server_explicit_missing():
    assert sv.find_llama_server("/nonexistent/llama-server") is None


def test_find_llama_server_version_sort():
    with tempfile.TemporaryDirectory() as d:
        for ver in ("2.9.0", "2.10.0", "2.8.0"):
            p = Path(d) / ("llama.cpp-x-%s" % ver) / "llama-server"
            p.parent.mkdir(parents=True)
            p.write_text("#!/bin/sh\n", encoding="utf-8")
            os.chmod(str(p), 0o755)
        got = sv.find_llama_server(None, str(Path(d) / "*" / "llama-server"))
        assert got is not None
        assert "2.10.0" in got  # 版本号按数字段比，2.10 > 2.9


def test_serve_command_shape():
    cmd = sv.serve_command(server="/s/llama-server", model="/m/x.gguf",
                           port=8099, context=4096)
    assert cmd == ('"/s/llama-server" -m "/m/x.gguf" -c 4096 -ngl 99 '
                   '--port 8099 --host 127.0.0.1')


def test_pid_and_log_paths():
    assert str(sv.pid_file()) == "/tmp/llm-odds-default.pid"
    assert str(sv.pid_file("jevk5-4b")) == "/tmp/llm-odds-jevk5-4b.pid"
    assert str(sv.log_file("a b/c")) == "/tmp/llm-odds-a_b_c.log"


def test_read_pid_missing():
    assert sv.read_pid("no-such-profile-xyz") is None


def test_status_dict_offline():
    orig = sv.health
    sv.health = lambda *a, **k: False
    try:
        st = sv.status(profile="zz-test-profile-123",
                       url="http://127.0.0.1:8099")
        assert st == {"profile": "zz-test-profile-123",
                      "url": "http://127.0.0.1:8099", "port": 8099,
                      "pid": None, "alive": False, "health": False}
    finally:
        sv.health = orig
