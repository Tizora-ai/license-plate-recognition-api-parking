"""
Tests for daily IP rate limiter and /demo endpoints (pure Python unittest).
"""

from __future__ import annotations

import io
import json
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi import Request, UploadFile

from python.api import demo_quota, demo_recognize
from python.rate_limiter import IPRateLimiter, limiter


class TestIPRateLimiter(unittest.TestCase):
    def setUp(self):
        self.test_limiter = IPRateLimiter(daily_limit=3, trust_proxy_headers=True)

    def test_rate_limit_enforcement(self):
        ip = "192.168.1.100"

        # Request 1: allowed
        allowed, limit, remaining, _ = self.test_limiter.check_and_increment(ip)
        self.assertTrue(allowed)
        self.assertEqual(limit, 3)
        self.assertEqual(remaining, 2)

        # Request 2: allowed
        allowed, limit, remaining, _ = self.test_limiter.check_and_increment(ip)
        self.assertTrue(allowed)
        self.assertEqual(remaining, 1)

        # Request 3: allowed (reaches limit)
        allowed, limit, remaining, _ = self.test_limiter.check_and_increment(ip)
        self.assertTrue(allowed)
        self.assertEqual(remaining, 0)

        # Request 4: blocked (exceeds limit)
        allowed, limit, remaining, retry_after = self.test_limiter.check_and_increment(ip)
        self.assertFalse(allowed)
        self.assertEqual(remaining, 0)
        self.assertGreater(retry_after, 0)

    def test_different_ips_isolated(self):
        ip1 = "10.0.0.1"
        ip2 = "10.0.0.2"

        # Exhaust ip1
        for _ in range(3):
            self.test_limiter.check_and_increment(ip1)

        self.assertFalse(self.test_limiter.check_and_increment(ip1)[0])

        # ip2 should still have full quota
        allowed, _, remaining, _ = self.test_limiter.check_and_increment(ip2)
        self.assertTrue(allowed)
        self.assertEqual(remaining, 2)

    def test_quota_does_not_consume(self):
        ip = "10.0.0.3"
        quota1 = self.test_limiter.get_quota(ip)
        self.assertEqual(quota1["remaining"], 3)
        self.assertEqual(quota1["used"], 0)

        # Increment once
        self.test_limiter.check_and_increment(ip)

        quota2 = self.test_limiter.get_quota(ip)
        self.assertEqual(quota2["remaining"], 2)
        self.assertEqual(quota2["used"], 1)

    def test_daily_reset_and_cleanup(self):
        ip = "10.0.0.4"
        # Exhaust quota for "today"
        for _ in range(3):
            self.test_limiter.check_and_increment(ip)
        self.assertFalse(self.test_limiter.check_and_increment(ip)[0])

        # Simulate next day by mocking _current_date
        with patch.object(self.test_limiter, "_current_date", return_value="2026-09-23"):
            allowed, _, remaining, _ = self.test_limiter.check_and_increment(ip)
            self.assertTrue(allowed)
            self.assertEqual(remaining, 2)

    def test_ip_extraction_direct_and_proxy(self):
        # Direct connection (no proxy header)
        mock_req = MagicMock(spec=Request)
        mock_req.client.host = "1.2.3.4"
        mock_req.headers = {}
        self.assertEqual(self.test_limiter.get_client_ip(mock_req), "1.2.3.4")

        # Cloudflare header
        mock_req.headers = {"cf-connecting-ip": "8.8.8.8"}
        self.assertEqual(self.test_limiter.get_client_ip(mock_req), "8.8.8.8")

        # X-Forwarded-For header with multiple IPs
        mock_req.headers = {"x-forwarded-for": "9.9.9.9, 10.0.0.1"}
        self.assertEqual(self.test_limiter.get_client_ip(mock_req), "9.9.9.9")


class TestDemoAPIEndpoints(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        limiter._records.clear()
        limiter.daily_limit = 2

    def tearDown(self):
        limiter.daily_limit = 10
        limiter._records.clear()

    def _make_mock_request(self, ip: str = "127.0.0.1") -> MagicMock:
        req = MagicMock(spec=Request)
        req.client.host = ip
        req.headers = {}
        return req

    def _make_mock_image(self) -> MagicMock:
        img = MagicMock(spec=UploadFile)
        img.content_type = "image/jpeg"
        img.filename = "car.jpg"
        img.read = AsyncMock(return_value=b"fake_jpeg_content")
        return img

    def test_demo_quota_endpoint(self):
        req = self._make_mock_request("192.168.1.55")
        result = demo_quota(req)
        self.assertTrue(result["ok"])
        self.assertEqual(result["ip"], "192.168.1.55")
        self.assertEqual(result["limit"], 2)
        self.assertEqual(result["remaining"], 2)

    @patch("python.api.predict_with_annotation")
    @patch("python.api.decode_image_bytes")
    async def test_demo_recognize_rate_limiting(self, mock_decode, mock_predict_annotated):
        mock_decode.return_value = MagicMock()
        mock_predict_annotated.return_value = (
            [{"plate": "XYZ9876", "box": {"x1": 10, "y1": 10, "x2": 50, "y2": 50}}],
            15.0,
            b"fake_annotated_jpeg",
        )

        req = self._make_mock_request("192.168.1.77")

        # Request 1: success (remaining: 1)
        res1 = await demo_recognize(req, self._make_mock_image())
        self.assertEqual(res1.status_code, 200)
        self.assertEqual(res1.headers.get("X-RateLimit-Remaining"), "1")
        self.assertEqual(res1.headers.get("X-RateLimit-Limit"), "2")

        body1 = json.loads(res1.body.decode())
        self.assertTrue(body1["ok"])
        self.assertIn("annotated_image", body1)
        self.assertTrue(body1["annotated_image"].startswith("data:image/jpeg;base64,"))

        # Request 2: success (remaining: 0)
        res2 = await demo_recognize(req, self._make_mock_image())
        self.assertEqual(res2.status_code, 200)
        self.assertEqual(res2.headers.get("X-RateLimit-Remaining"), "0")

        # Request 3: blocked (HTTP 429)
        res3 = await demo_recognize(req, self._make_mock_image())
        self.assertEqual(res3.status_code, 429)
        self.assertEqual(res3.headers.get("X-RateLimit-Remaining"), "0")
        self.assertIn("Retry-After", res3.headers)

        body3 = json.loads(res3.body.decode())
        self.assertFalse(body3["ok"])
        self.assertEqual(body3["error"], "Rate limit exceeded")
        self.assertIn("Daily demo limit of 2 requests reached", body3["message"])

    @patch("python.api.predict_with_annotation")
    @patch("python.api.decode_image_bytes")
    async def test_demo_recognize_image_shares_quota(self, mock_decode, mock_annotate):
        from python.api import demo_recognize_image

        mock_decode.return_value = MagicMock()
        mock_annotate.return_value = ([], 10.0, b"annotated_jpeg")

        req = self._make_mock_request("192.168.1.88")

        # Request 1 via demo_recognize: success
        with patch("python.api.predict_image", return_value=([], 10.0)):
            res1 = await demo_recognize(req, self._make_mock_image())
            self.assertEqual(res1.status_code, 200)

        # Request 2 via demo_recognize_image: success
        res2 = await demo_recognize_image(req, self._make_mock_image())
        self.assertEqual(res2.status_code, 200)
        self.assertEqual(res2.body, b"annotated_jpeg")

        # Request 3 via demo_recognize_image: blocked (HTTP 429)
        res3 = await demo_recognize_image(req, self._make_mock_image())
        self.assertEqual(res3.status_code, 429)


if __name__ == "__main__":
    unittest.main()
