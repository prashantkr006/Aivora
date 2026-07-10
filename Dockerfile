FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app/src

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential curl git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY . .

# Runtime — Streamlit dashboard is the main process; docker-compose
# starts the OAuth sidecar as a second service so cross-service
# restarts stay independent.
EXPOSE 8501 8502

CMD ["python", "-m", "streamlit", "run", "app/multi_user_app.py", \
     "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true"]
