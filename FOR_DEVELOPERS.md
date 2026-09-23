# Sharing this zip

## Included
- `python/` — FastAPI API + ALPR engine + CLI
- `docs/` — short project docs
- `assets/` — sample test image
- `requirements.txt` — Windows (DirectML)
- `requirements-linux-cpu.txt` — Linux CPU (`onnx`)
- `.env.example` — copy to `.env` and edit
- `README.md` — run + deploy docs

## Not included (recreate on each machine)
- `.venv/` — create with `python -m venv .venv`
- `.env` — copy from `.env.example`
- Downloaded ONNX model cache (auto-downloads on first run)

## Quick start (Windows)
```powershell
cd alpr-api
copy .env.example .env
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
.\.venv\Scripts\python.exe -m python.api
```

## Quick start (Linux CPU)
```bash
cd alpr
cp .env.example .env
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements-linux-cpu.txt
set -a; source .env; set +a
python -m python.api
```

API: http://127.0.0.1:8000/docs  
Recognize: `POST /recognize`
