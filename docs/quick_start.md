## Quick start

Start the API:

```shell
python -m python.api
```

Recognize a plate:

```shell
curl -X POST http://127.0.0.1:8000/recognize -F "image=@photo.jpg"
```

CLI:

```shell
python -m python.cli predict path/to/image.jpg
```
