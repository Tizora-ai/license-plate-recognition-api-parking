"""
FastAPI service for alpr (in-process DirectML/CUDA engine).
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from typing import Any

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.responses import JSONResponse, Response

from python.alpr_engine import (
    decode_image_bytes,
    encode_image_base64,
    engine_info,
    predict_image,
    predict_with_annotation,
    warmup,
)
from python.rate_limiter import limiter

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("alpr.api")

MAX_UPLOAD_BYTES = 15 * 1024 * 1024
ALLOWED_TYPES = {
    "image/jpeg",
    "image/jpg",
    "image/png",
    "image/bmp",
    "image/webp",
    "application/octet-stream",
}


@asynccontextmanager
async def lifespan(_: FastAPI):
    logger.info("Loading ALPR engine...")
    info = engine_info()
    logger.info("Engine ready: %s", info)
    warmup(runs=2)
    yield


app = FastAPI(
    title="alpr",
    description="License plate recognition API",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
def health() -> dict[str, Any]:
    return {"ok": True, **engine_info()}


@app.post("/recognize")
async def recognize(image: UploadFile = File(..., description="Vehicle / plate image")) -> JSONResponse:
    """JSON plates only (no annotated image in body)."""
    content_type = (image.content_type or "application/octet-stream").lower()
    if content_type not in ALLOWED_TYPES and not content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail=f"Unsupported content type: {content_type}")

    data = await image.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty file")
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=400, detail="File too large (max 15MB)")

    try:
        frame = decode_image_bytes(data)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    plates, latency_ms = predict_image(frame)
    return JSONResponse(
        {
            "ok": True,
            "plates": plates,
            "count": len(plates),
            "latency_ms": round(latency_ms, 2),
            "filename": image.filename,
        }
    )


@app.post(
    "/recognize/image",
    responses={200: {"content": {"image/jpeg": {}}}},
)
async def recognize_image_file(
    image: UploadFile = File(..., description="Vehicle / plate image"),
    jpeg_quality: int = Query(85, ge=40, le=95),
) -> Response:
    """Annotated JPEG only (boxes + plate text)."""
    content_type = (image.content_type or "application/octet-stream").lower()
    if content_type not in ALLOWED_TYPES and not content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail=f"Unsupported content type: {content_type}")

    data = await image.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty file")
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=400, detail="File too large (max 15MB)")

    try:
        frame = decode_image_bytes(data)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    _, latency_ms, jpeg_bytes = predict_with_annotation(frame, jpeg_quality=jpeg_quality)
    return Response(
        content=jpeg_bytes,
        media_type="image/jpeg",
        headers={"X-ALPR-Latency-Ms": f"{latency_ms:.2f}"},
    )


@app.get("/demo/quota")
def demo_quota(request: Request) -> dict[str, Any]:
    """Check remaining daily demo requests for the caller's IP."""
    client_ip = limiter.get_client_ip(request)
    return {"ok": True, **limiter.get_quota(client_ip)}


def _crop_plate_base64(frame: np.ndarray, box: dict[str, Any] | None) -> str | None:
    """Crop the bounding box of a detected plate and return base64 Data-URL."""
    if not box:
        return None
    try:
        if not hasattr(frame, "shape") or len(frame.shape) < 2:
            return None
        h, w = frame.shape[:2]
        x1 = max(0, min(int(box.get("x1", 0)), w - 1))
        x2 = max(0, min(int(box.get("x2", w)), w))
        y1 = max(0, min(int(box.get("y1", 0)), h - 1))
        y2 = max(0, min(int(box.get("y2", h)), h))
        if x2 > x1 and y2 > y1:
            crop = frame[y1:y2, x1:x2]
            ok, buf = cv2.imencode(".jpg", crop, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
            if ok:
                return encode_image_base64(buf.tobytes())
    except Exception as exc:
        logger.warning("Could not crop plate image: %s", exc)
    return None


@app.post("/demo/recognize")
async def demo_recognize(
    request: Request,
    image: UploadFile = File(..., description="Vehicle / plate image for demo"),
    include_annotated_image: bool = Query(
        True, description="Include full base64 annotated image in response"
    ),
    include_plate_crop: bool = Query(
        True, description="Include base64 cropped plate in each plate item"
    ),
) -> JSONResponse:
    """Demo endpoint: JSON plates with daily rate limit and extracted images."""
    client_ip = limiter.get_client_ip(request)
    allowed, limit, remaining, reset_seconds = limiter.check_and_increment(client_ip)

    rate_headers = {
        "X-RateLimit-Limit": str(limit),
        "X-RateLimit-Remaining": str(remaining),
        "X-RateLimit-Reset": str(reset_seconds),
    }

    if not allowed:
        return JSONResponse(
            status_code=429,
            content={
                "ok": False,
                "error": "Rate limit exceeded",
                "message": (
                    f"Daily demo limit of {limit} requests reached for your IP ({client_ip}). "
                    "Please try again tomorrow."
                ),
                "limit": limit,
                "remaining": 0,
                "retry_after_seconds": reset_seconds,
            },
            headers={"Retry-After": str(reset_seconds), **rate_headers},
        )

    content_type = (image.content_type or "application/octet-stream").lower()
    if content_type not in ALLOWED_TYPES and not content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail=f"Unsupported content type: {content_type}")

    data = await image.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty file")
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=400, detail="File too large (max 15MB)")

    try:
        frame = decode_image_bytes(data)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    annotated_image_b64: str | None = None
    if include_annotated_image:
        plates, latency_ms, jpeg_bytes = predict_with_annotation(frame)
        annotated_image_b64 = encode_image_base64(jpeg_bytes)
    else:
        plates, latency_ms = predict_image(frame)

    if include_plate_crop:
        for plate in plates:
            crop_b64 = _crop_plate_base64(frame, plate.get("box"))
            if crop_b64:
                plate["crop_image"] = crop_b64

    response_content: dict[str, Any] = {
        "ok": True,
        "plates": plates,
        "count": len(plates),
        "latency_ms": round(latency_ms, 2),
        "filename": image.filename,
    }
    if annotated_image_b64:
        response_content["annotated_image"] = annotated_image_b64

    return JSONResponse(response_content, headers=rate_headers)


@app.post(
    "/demo/recognize/image",
    responses={200: {"content": {"image/jpeg": {}}}},
)
async def demo_recognize_image(
    request: Request,
    image: UploadFile = File(..., description="Vehicle / plate image for demo"),
    jpeg_quality: int = Query(85, ge=40, le=95),
) -> Response:
    """Demo endpoint: Annotated JPEG with daily rate limit (shares same quota)."""
    client_ip = limiter.get_client_ip(request)
    allowed, limit, remaining, reset_seconds = limiter.check_and_increment(client_ip)

    rate_headers = {
        "X-RateLimit-Limit": str(limit),
        "X-RateLimit-Remaining": str(remaining),
        "X-RateLimit-Reset": str(reset_seconds),
    }

    if not allowed:
        return JSONResponse(
            status_code=429,
            content={
                "ok": False,
                "error": "Rate limit exceeded",
                "message": (
                    f"Daily demo limit of {limit} requests reached for your IP ({client_ip}). "
                    "Please try again tomorrow."
                ),
                "limit": limit,
                "remaining": 0,
                "retry_after_seconds": reset_seconds,
            },
            headers={"Retry-After": str(reset_seconds), **rate_headers},
        )

    response = await recognize_image_file(image, jpeg_quality=jpeg_quality)
    for key, value in rate_headers.items():
        response.headers[key] = value
    return response


def main() -> None:
    import uvicorn

    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("python.api:app", host=host, port=port, reload=False, workers=1)


if __name__ == "__main__":
    main()
