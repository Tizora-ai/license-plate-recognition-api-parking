# License Plate Recognition API

## Description
An ALPR (Automatic License Plate Recognition) REST API for parking. Send it a photo from an
entry camera, a patrol car, or a phone. It sends back the plate number, how sure it is, and
where the plate is in the photo.

The API only reads plates. Your app does the rest — check the plate against your permit
list, start or end a parking session, open a gate, or mark a car that stayed too long. The
answer is plain JSON, so it is easy to connect to the system you already use.

Built with **FastAPI** and **ONNX Runtime**. It runs on a GPU (DirectML / CUDA) or on a
normal CPU, so you can run it on a laptop, a small box at the site, or a cloud server.

## Use Cases
- **Permit verification** — read the plate at the entrance and check it against your permit
  list before the barrier opens.
- **Parking enforcement** — let a patrol officer photograph a vehicle and get the plate back
  for a permit or payment check, instead of typing it by hand.
- **Digital chalking** — record plate and timestamp on each pass through a timed zone, and
  compare the two to find overstays.
- **Ticketless / pay-by-plate parking** — capture the plate on entry and exit and use it as
  the key for the session and the payment.
- **Gated and residential access** — open a gate for vehicles on a resident or staff list.
- **Occupancy and audit records** — log which vehicles were in a lot and when.

## Features
- Detects multiple license plates in one image.
- Returns plate text, confidence, and bounding box as JSON.
- Returns an annotated image with boxes and plate text.
- Three modes: `fast`, `balanced` and `accurate`.
- CLI for prediction and latency benchmarking.

## Technologies Used
- Python, FastAPI, Uvicorn
- ONNX Runtime
- YOLOv9 plate detector + CCT OCR
- OpenCV, NumPy

## How It Works
```
        Image upload (multipart/form-data)
                │
                ▼
        Validate type and size (max 15 MB)
                │
                ▼
        Decode to BGR frame (OpenCV)
                │
                ▼
┌──────────────────────────────────────────────────┐
│  YOLOv9 detector  ->  crop plate  ->  CCT OCR    │
│  ONNX Runtime: DirectML -> CUDA -> CPU           │
└──────────────────────────────────────────────────┘
                │
                ├──►  JSON: text, confidence, box, latency_ms
                └──►  Annotated JPEG: boxes + plate text drawn
```

The models load once when the server starts and stay in memory, warmed up so the
first request is not slow.

## Model Modes
Set the mode with the `ALPR_MODE` environment variable:

| Mode | Detector | OCR | Confidence | Use case |
|------|----------|-----|------------|----------|
| `fast` (default) | `yolo-v9-t-384-license-plate-end2end` | `cct-xs-v2-global-model` | 0.40 | CPU servers / low latency |
| `balanced` | `yolo-v9-t-512-license-plate-end2end` | `cct-xs-v2-global-model` | 0.35 | Between fast and accurate |
| `accurate` | `yolo-v9-s-608-license-plate-end2end` | `cct-s-v2-global-model` | 0.30 | Best recall (GPU recommended) |

You can also set the models and threshold one by one with `ALPR_DETECTOR_MODEL`, `ALPR_OCR_MODEL` and `ALPR_DET_CONF`.

## Installation

### 1. Clone the repository
```bash
git clone https://github.com/Tizora-ai/license-plate-recognition-api-parking.git
cd license-plate-recognition-api-parking
```

### 2. Windows (local, DirectML GPU)
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m python.api
```

## Configuration
Copy `.env.example` to `.env`, then edit it:

| Variable | Default | Description |
|----------|---------|-------------|
| `HOST` / `PORT` | `0.0.0.0` / `8000` | Address and port the server listens on |
| `ALPR_MODE` | `fast` | `fast`, `balanced` or `accurate` |
| `ALPR_PROVIDERS` | auto | `cpu`, `cuda` or `directml` (auto tries DirectML, then CUDA, then CPU) |
| `ALPR_DETECTOR_MODEL` | from mode | Override the detector model |
| `ALPR_OCR_MODEL` | from mode | Override the OCR model |
| `ALPR_DET_CONF` | from mode | Detector confidence threshold |
| `ALPR_INTRA_THREADS` | `4` | ONNX intra-op threads |
| `ALPR_INTER_THREADS` | `1` | ONNX inter-op threads |

## Usage

### API Endpoints
| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Service status and engine / provider information |
| `POST` | `/recognize` | Detected plates as JSON |
| `POST` | `/recognize/image` | Annotated JPEG (latency in the `X-ALPR-Latency-Ms` header) |

Interactive docs: `http://<host>:8000/docs`

### Examples
```bash
# Recognize plates (JSON)
curl -X POST http://127.0.0.1:8000/recognize -F "image=@car.jpg"

# Get an annotated image
curl -X POST http://127.0.0.1:8000/recognize/image -F "image=@car.jpg" -o result.jpg
```

### Sample Response
```json
{
  "ok": true,
  "plates": [
    {
      "text": "MH12AB1234",
      "confidence": 0.97,
      "detection_confidence": 0.91,
      "box": { "x1": 412, "y1": 388, "x2": 598, "y2": 441 }
    }
  ],
  "count": 1,
  "latency_ms": 38.52,
  "filename": "car.jpg"
}
```