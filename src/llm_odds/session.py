"""常驻进程里的 TTL 缓存（CLI 单发模式下等价于直通）。"""
import json
import threading
import time
from typing import Dict, Optional, Tuple


def cache_key(item: dict) -> str:
    """请求内容 → 缓存键（剥掉 id；其余字段规范化成排序 JSON）。"""
    payload = {k: v for k, v in item.items() if k != "id"}
    return json.dumps(payload, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"))


class Session:
    """线程安全的 TTL 缓存：key → 上次的完整响应。

    `default_ttl <= 0` 表示默认关闭（请求里 `cache_ttl_s > 0` 可单独开）；
    MCP 常驻进程用 `Session(default_ttl=12)`。
    """

    def __init__(self, default_ttl: float = 0.0, max_items: int = 256) -> None:
        self.default_ttl = float(default_ttl)
        self.max_items = int(max_items)
        self._lock = threading.Lock()
        self._store = {}  # type: Dict[str, Tuple[float, dict]]

    def get(self, key: str) -> Optional[dict]:
        now = time.time()
        with self._lock:
            hit = self._store.get(key)
            if hit is None:
                return None
            expires, payload = hit
            if now >= expires:
                self._store.pop(key, None)
                return None
            # 深拷贝：调用方可能改 metrics.cache，不能污染缓存
            return json.loads(json.dumps(payload))

    def put(self, key: str, payload: dict, ttl: Optional[float] = None) -> None:
        ttl = self.default_ttl if ttl is None else float(ttl)
        if ttl <= 0:
            return
        with self._lock:
            if len(self._store) >= self.max_items:
                now = time.time()
                for k in [k for k, (exp, _) in self._store.items() if now >= exp]:
                    self._store.pop(k, None)
                while len(self._store) >= self.max_items:
                    self._store.pop(next(iter(self._store)))
            self._store[key] = (time.time() + ttl, payload)
