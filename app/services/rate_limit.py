import time
from collections import defaultdict

from app.core.config import get_settings


class TokenBucketRateLimiter:
    def __init__(self) -> None:
        self._buckets: dict[str, dict[str, float]] = defaultdict(
            lambda: {"tokens": get_settings().rate_limit_rps, "last": time.monotonic()}
        )

    def allow(self, key: str) -> bool:
        rate = get_settings().rate_limit_rps
        bucket = self._buckets[key]
        now = time.monotonic()
        elapsed = now - bucket["last"]
        bucket["tokens"] = min(rate, bucket["tokens"] + elapsed * rate)
        bucket["last"] = now
        if bucket["tokens"] >= 1:
            bucket["tokens"] -= 1
            return True
        return False


rate_limiter = TokenBucketRateLimiter()
