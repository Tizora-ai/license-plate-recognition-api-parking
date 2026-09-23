"""
Low-latency ALPR engine: shared ONNX session, DirectML/CUDA-first providers.

Modes (env ALPR_MODE):
  fast     — t-384 + cct-xs  (default, low latency)
  balanced — t-512 + cct-xs
  accurate — s-608 + cct-s   (HF-like recall, slower)

Per-field overrides still work: ALPR_DETECTOR_MODEL, ALPR_OCR_MODEL, ALPR_DET_CONF
"""

from __future__ import annotations

import logging
import os
import statistics
import time
from dataclasses import asdict, is_dataclass
from typing import Any

import cv2
import numpy as np
import onnxruntime as ort
from fast_alpr import ALPR, ALPRResult

logger = logging.getLogger(__name__)

_MODE_PRESETS: dict[str, dict[str, Any]] = {
    "fast": {
        "detector": "yolo-v9-t-384-license-plate-end2end",
        "ocr": "cct-xs-v2-global-model",
        "conf": 0.4,
    },
    "balanced": {
        "detector": "yolo-v9-t-512-license-plate-end2end",
        "ocr": "cct-xs-v2-global-model",
        "conf": 0.35,
    },
    "accurate": {
        "detector": "yolo-v9-s-608-license-plate-end2end",
        "ocr": "cct-s-v2-global-model",
        "conf": 0.3,
    },
}


def _resolve_mode() -> str:
    mode = os.getenv("ALPR_MODE", "fast").strip().lower()
    if mode not in _MODE_PRESETS:
        logger.warning("Unknown ALPR_MODE=%r, falling back to 'fast'", mode)
        return "fast"
    return mode


ALPR_MODE = _resolve_mode()
_preset = _MODE_PRESETS[ALPR_MODE]

DETECTOR_MODEL = os.getenv("ALPR_DETECTOR_MODEL", _preset["detector"])
OCR_MODEL = os.getenv("ALPR_OCR_MODEL", _preset["ocr"])
DETECTOR_CONF_THRESH = float(os.getenv("ALPR_DET_CONF", str(_preset["conf"])))

_alpr: ALPR | None = None
_providers: list[str] = []
_warmed_up = False


def _session_options() -> ort.SessionOptions:
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    # More CPU threads help on Linux CPU hosts; keep low on GPU.
    so.intra_op_num_threads = int(os.getenv("ALPR_INTRA_THREADS", "4"))
    so.inter_op_num_threads = int(os.getenv("ALPR_INTER_THREADS", "1"))
    return so


def _pick_providers() -> list[str]:
    """
    Provider order:
      ALPR_PROVIDERS=cpu          → CPU only (Linux CPU EC2)
      ALPR_PROVIDERS=cuda         → CUDA then CPU
      ALPR_PROVIDERS=directml     → DirectML then CPU
      unset                       → auto: DirectML → CUDA → CPU
    """
    available = set(ort.get_available_providers())
    forced = os.getenv("ALPR_PROVIDERS", "").strip().lower()

    if forced in ("cpu", "cpuonly", "cpu_only"):
        return ["CPUExecutionProvider"]

    ordered: list[str] = []
    if forced in ("directml", "dml"):
        if "DmlExecutionProvider" in available:
            ordered.append("DmlExecutionProvider")
    elif forced in ("cuda", "gpu"):
        if "CUDAExecutionProvider" in available:
            ordered.append("CUDAExecutionProvider")
    else:
        if "DmlExecutionProvider" in available:
            ordered.append("DmlExecutionProvider")
        if "CUDAExecutionProvider" in available:
            ordered.append("CUDAExecutionProvider")

    ordered.append("CPUExecutionProvider")
    return ordered


def _ocr_device(providers: list[str]) -> str:
    if providers == ["CPUExecutionProvider"]:
        return "cpu"
    if "CUDAExecutionProvider" in providers or "DmlExecutionProvider" in providers:
        return "cuda"
    return "cpu"


def _input_size() -> int:
    for size in (640, 608, 512, 416, 384, 256):
        if str(size) in DETECTOR_MODEL:
            return size
    return 384


def get_engine() -> ALPR:
    """Return the process-wide ALPR singleton (created once)."""
    global _alpr, _providers
    if _alpr is not None:
        return _alpr

    _providers = _pick_providers()
    so = _session_options()
    logger.info(
        "Initializing ALPR mode=%s detector=%s ocr=%s conf=%.2f providers=%s",
        ALPR_MODE,
        DETECTOR_MODEL,
        OCR_MODEL,
        DETECTOR_CONF_THRESH,
        _providers,
    )
    _alpr = ALPR(
        detector_model=DETECTOR_MODEL,
        ocr_model=OCR_MODEL,
        detector_conf_thresh=DETECTOR_CONF_THRESH,
        detector_providers=_providers,
        ocr_providers=_providers,
        detector_sess_options=so,
        ocr_sess_options=so,
        ocr_device=_ocr_device(_providers),
    )
    return _alpr


def engine_info() -> dict[str, Any]:
    get_engine()
    return {
        "mode": ALPR_MODE,
        "detector_model": DETECTOR_MODEL,
        "ocr_model": OCR_MODEL,
        "detector_conf_thresh": DETECTOR_CONF_THRESH,
        "providers": list(_providers),
        "providers_env": os.getenv("ALPR_PROVIDERS", "auto"),
        "available_providers": ort.get_available_providers(),
        "warmed_up": _warmed_up,
    }


def warmup(runs: int = 2) -> float:
    """Run dummy inferences so first real request is not cold."""
    global _warmed_up
    alpr = get_engine()
    size = _input_size()
    dummy = np.zeros((size, size, 3), dtype=np.uint8)
    t0 = time.perf_counter()
    for _ in range(max(1, runs)):
        alpr.predict(dummy)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    _warmed_up = True
    logger.info("Warmup done: %d runs in %.1f ms", runs, elapsed_ms)
    return elapsed_ms


def _ocr_confidence(value: float | list[float] | None) -> float | None:
    if value is None:
        return None
    if isinstance(value, list):
        return float(statistics.fmean(value)) if value else None
    return float(value)


def _serialize_result(result: ALPRResult) -> dict[str, Any]:
    det = result.detection
    ocr = result.ocr
    plate: dict[str, Any] = {
        "text": ocr.text if ocr else None,
        "confidence": _ocr_confidence(ocr.confidence) if ocr else None,
        "detection_confidence": float(det.confidence) if det.confidence is not None else None,
    }
    box = getattr(det, "bounding_box", None)
    if box is not None and is_dataclass(box):
        plate["box"] = asdict(box)
    return plate


def predict_image(image: np.ndarray | str) -> tuple[list[dict[str, Any]], float]:
    """Run ALPR and return (plates, latency_ms)."""
    alpr = get_engine()
    t0 = time.perf_counter()
    results = alpr.predict(image)
    latency_ms = (time.perf_counter() - t0) * 1000.0
    plates = [_serialize_result(r) for r in results]
    return plates, latency_ms


def predict_with_annotation(
    image: np.ndarray,
    *,
    jpeg_quality: int = 85,
) -> tuple[list[dict[str, Any]], float, bytes]:
    """
    Run ALPR once via draw_predictions; return plates, latency_ms, JPEG bytes of annotated image.
    """
    alpr = get_engine()
    # draw_predictions draws in-place; copy so caller frame stays clean
    frame = image.copy()
    t0 = time.perf_counter()
    drawn = alpr.draw_predictions(frame)
    latency_ms = (time.perf_counter() - t0) * 1000.0
    plates = [_serialize_result(r) for r in drawn.results]
    ok, buf = cv2.imencode(
        ".jpg",
        drawn.image,
        [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)],
    )
    if not ok:
        raise ValueError("Failed to encode annotated image")
    return plates, latency_ms, buf.tobytes()


def encode_image_base64(image_bytes: bytes, mime: str = "image/jpeg") -> str:
    """Return a data-URL (or raw base64 if mime empty)."""
    import base64

    b64 = base64.b64encode(image_bytes).decode("ascii")
    if mime:
        return f"data:{mime};base64,{b64}"
    return b64


def decode_image_bytes(data: bytes) -> np.ndarray:
    arr = np.frombuffer(data, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("Could not decode image bytes")
    return frame
