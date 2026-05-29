# CPU image — runs on any OS with Docker, no local Python setup needed.
# faster-whisper's ctranslate2 and PyAV ship self-contained wheels (PyAV bundles
# the ffmpeg libraries), so no apt ffmpeg/system codecs are required.
#
# GPU: build a derived image that also `pip install -r requirements-gpu.txt` and
# run with `--gpus all` on an NVIDIA host (see README).
FROM python:3.14-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install deps first for layer caching.
COPY requirements.txt ./
RUN pip install --upgrade pip && pip install -r requirements.txt

# App code. Runtime state (DBs, captures, logs, model cache) lives on mounted
# volumes — see docker-compose.yml — not baked into the image.
COPY . .

EXPOSE 8000

# `python main.py` runs uvicorn via main's __main__; it's also what the
# cross-platform self-restart (os.execv) re-execs.
CMD ["python", "main.py"]
