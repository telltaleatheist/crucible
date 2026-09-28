from __future__ import annotations

from collections import OrderedDict


class VisionFeatureCache:
    def __init__(self, max_size: int = 20) -> None:
        self.max_size = max_size
        self._cache: OrderedDict[str, object] = OrderedDict()

    def get(self, key: str) -> object | None:
        return self._cache.get(key)

    def put(self, key: str, features: object) -> None:
        self._cache[key] = features
        while len(self._cache) > self.max_size:
            self._cache.popitem(last=False)
