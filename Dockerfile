FROM python:3.9.7-slim-buster

COPY ./requirements.txt /app/requirements.txt

WORKDIR /app

RUN apt-get update && apt-get upgrade -y && \
    apt-get install -y --no-install-recommends \
    python3-pip \
    python3-dev \
    libssl-dev \
    libffi-dev \
    python3-setuptools \
    python3-venv \
    python3-wheel && \
    rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir -r requirements.txt

RUN useradd -m -U -u 1000 appuser
USER appuser

EXPOSE 8000
CMD ["uvicorn", "app:app", "--proxy-headers", "--host", "0.0.0.0", "--port", "8000"]