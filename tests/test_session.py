"""session.py：缓存键 / TTL / 拷贝隔离 / 淘汰。"""
import time

from llm_odds.session import Session, cache_key


def test_cache_key_strips_id():
    a = cache_key({"id": "x", "task": "choice", "params": {"b": 1}})
    b = cache_key({"task": "choice", "params": {"b": 1}, "id": "y"})
    assert a == b
    c = cache_key({"task": "choice", "params": {"b": 2}})
    assert a != c


def test_cache_key_order_insensitive():
    a = cache_key({"task": "choice", "evidence": "e", "options": ["a"]})
    b = cache_key({"options": ["a"], "evidence": "e", "task": "choice"})
    assert a == b


def test_session_ttl_hit_then_expire():
    s = Session(default_ttl=0.05)
    s.put("k", {"v": 1})
    assert s.get("k") == {"v": 1}
    time.sleep(0.08)
    assert s.get("k") is None


def test_session_off_by_default():
    s = Session()  # default_ttl=0 → put 是空操作
    s.put("k", {"v": 1})
    assert s.get("k") is None


def test_session_returns_copies():
    s = Session(default_ttl=10)
    s.put("k", {"metrics": {"cache": "miss"}, "n": 1})
    got = s.get("k")
    got["metrics"]["cache"] = "hit"
    got["n"] = 999
    again = s.get("k")
    assert again["metrics"]["cache"] == "miss"
    assert again["n"] == 1


def test_session_put_ttl_override():
    s = Session()  # 默认关
    s.put("k", {"v": 1}, ttl=5.0)
    assert s.get("k") == {"v": 1}


def test_session_eviction():
    s = Session(default_ttl=10, max_items=2)
    s.put("a", {"v": 1})
    s.put("b", {"v": 2})
    s.put("c", {"v": 3})
    assert s.get("a") is None
    assert s.get("b") is not None
    assert s.get("c") is not None
