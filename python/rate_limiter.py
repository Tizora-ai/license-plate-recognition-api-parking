"""
In-memory daily IP rate limiter for demo endpoints.
Thread-safe, zero external dependencies, automatic RAM cleanup on day transitions.
"""

from __future__ import annotations

import datetime
import logging
import os
import threading
from typing import Any

from fastapi import Request

logger = logging.getLogger("alpr.limiter")


class IPRateLimiter:
    """Thread-safe in-memory daily rate limiter per IP."""

    def __init__(self, daily_limit: int = 10, trust_proxy_headers: bool = False):
        self.daily_limit = daily_limit
        self.trust_proxy_headers = trust_proxy_headers
        self._lock = threading.Lock()
        # Storage: {ip: {"date": "YYYY-MM-DD", "count": int}}
        self._records: dict[str, dict[str, Any]] = {}
        self._last_cleanup_date = self._current_date()

    @staticmethod
    def _current_date() -> str:
        """Current date string (UTC) used for daily resets."""
        return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d")

    @staticmethod
    def _seconds_until_midnight() -> int:
        """Seconds remaining until the next UTC midnight."""
        now = datetime.datetime.now(datetime.timezone.utc)
        tomorrow = (now + datetime.timedelta(days=1)).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        return max(1, int((tomorrow - now).total_seconds()))

    def _cleanup_old_records(self, today: str) -> None:
        """Purge entries from prior days to keep memory usage minimal."""
        if today != self._last_cleanup_date:
            keys_to_delete = [
                ip for ip, data in self._records.items() if data.get("date") != today
            ]
            for ip in keys_to_delete:
                del self._records[ip]
            self._last_cleanup_date = today
            logger.info("Cleaned up %d expired rate-limit IP records", len(keys_to_delete))

    def get_client_ip(self, request: Request) -> str:
        """Extract the real client IP address safely."""
        if self.trust_proxy_headers:
            # Check Cloudflare connecting IP
            cf_ip = request.headers.get("cf-connecting-ip")
            if cf_ip:
                return cf_ip.strip()

            # Check standard X-Forwarded-For header (first address is client)
            forwarded_for = request.headers.get("x-forwarded-for")
            if forwarded_for:
                return forwarded_for.split(",")[0].strip()

            # Check X-Real-IP
            real_ip = request.headers.get("x-real-ip")
            if real_ip:
                return real_ip.strip()

        if request.client and request.client.host:
            return request.client.host

        return "127.0.0.1"

    def check_and_increment(self, ip: str) -> tuple[bool, int, int, int]:
        """
        Check if IP is within daily limit and increment usage if allowed.
        Returns:
            (allowed: bool, limit: int, remaining: int, reset_seconds: int)
        """
        today = self._current_date()
        reset_seconds = self._seconds_until_midnight()

        with self._lock:
            self._cleanup_old_records(today)

            record = self._records.setdefault(ip, {"date": today, "count": 0})

            # If record is from a previous day, reset count
            if record["date"] != today:
                record["date"] = today
                record["count"] = 0

            current_count = record["count"]

            if current_count >= self.daily_limit:
                # Limit reached
                return False, self.daily_limit, 0, reset_seconds

            # Increment count
            record["count"] += 1
            remaining = self.daily_limit - record["count"]
            return True, self.daily_limit, remaining, reset_seconds

    def get_quota(self, ip: str) -> dict[str, Any]:
        """Query remaining quota for an IP without incrementing."""
        today = self._current_date()
        reset_seconds = self._seconds_until_midnight()

        with self._lock:
            self._cleanup_old_records(today)
            record = self._records.get(ip)

            if not record or record.get("date") != today:
                used = 0
            else:
                used = record.get("count", 0)

            remaining = max(0, self.daily_limit - used)

        return {
            "ip": ip,
            "limit": self.daily_limit,
            "used": used,
            "remaining": remaining,
            "reset_in_seconds": reset_seconds,
        }


# Singleton instance configured from environment variables
_DEMO_DAILY_LIMIT = int(os.getenv("DEMO_DAILY_LIMIT", "10"))
_TRUST_PROXY_HEADERS = os.getenv("TRUST_PROXY_HEADERS", "false").strip().lower() in (
    "true",
    "1",
    "yes",
)

limiter = IPRateLimiter(
    daily_limit=_DEMO_DAILY_LIMIT,
    trust_proxy_headers=_TRUST_PROXY_HEADERS,
)
