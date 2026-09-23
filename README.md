# alpr

In-process license plate recognition API (FastAPI + ONNX).

## Windows (local)

```powershell
cd C:\Users\ADMIN\Desktop\alpr-api
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
.\.venv\Scripts\python.exe -m python.api
```

## Linux CPU (EC2 / VPS)

Use **`requirements-linux-cpu.txt`**, force CPU, prefer **`ALPR_MODE=fast`**.

### 1. Upload project (from Windows)

```powershell
scp -i your-key.pem -r python deploy docs assets requirements-linux-cpu.txt requirements.txt .env.example README.md ubuntu@<EC2_IP>:~/alpr/
```

### 2. On the server

```bash
cd ~/alpr
bash deploy/linux-cpu/setup.sh
set -a; source .env; set +a
source .venv/bin/activate
python -m python.api
```

### 3. systemd (keeps running)

```bash
sudo bash deploy/linux-cpu/install-systemd.sh
curl http://127.0.0.1:8000/health
```

Security group: open **22** + **8000**.

### Env for Linux CPU

| Variable | Value | Why |
|----------|--------|-----|
| `ALPR_PROVIDERS` | `cpu` | Force CPUExecutionProvider |
| `ALPR_MODE` | `fast` | Tiny models; `accurate` is slow on CPU |
| `HOST` / `PORT` | `0.0.0.0` / `8000` | Bind for remote access |

## API

- `GET /health`
- `POST /recognize` — JSON plates (unrestricted / production)
- `POST /recognize/image` — annotated JPEG (unrestricted / production)
- `POST /demo/recognize` — JSON plates with daily IP rate limit (default 10/day)
- `POST /demo/recognize/image` — annotated JPEG with daily IP rate limit (default 10/day)
- `GET /demo/quota` — check remaining daily demo quota for caller's IP
- Docs: `http://<host>:8000/docs`

```bash
# Standard recognition:
curl -X POST http://127.0.0.1:8000/recognize -F "image=@photo.jpg"

# Demo recognition (capped at 10 requests/day per IP):
curl -X POST http://127.0.0.1:8000/demo/recognize -F "image=@photo.jpg"

# Check demo remaining quota:
curl http://127.0.0.1:8000/demo/quota
```

## ALPR_MODE

| Mode | Detector | OCR | Use |
|------|----------|-----|-----|
| `fast` (default) | t-384 | cct-xs | CPU / low latency |
| `balanced` | t-512 | cct-xs | Middle |
| `accurate` | s-608 | cct-s | Max recall (slow on CPU) |

```bash
export ALPR_MODE=accurate   # only if you accept higher latency
```
