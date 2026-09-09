FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake libopenblas-dev libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY configs ./configs
COPY src ./src
COPY app.py ./
RUN pip install --no-cache-dir . \
    && useradd --create-home --shell /usr/sbin/nologin appuser \
    && mkdir -p /app/face_attendance_data \
    && chown -R appuser:appuser /app

USER appuser

EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0"]
