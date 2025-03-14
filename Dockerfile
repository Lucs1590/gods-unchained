FROM python:3.9.7-slim-buster

COPY ./requirements.txt /app/requirements.txt

ENV PYTHONUNBUFFERED 1
ENV PYTHONDONTWRITEBYTECODE 1

WORKDIR /app

RUN apt-get update && apt-get upgrade -y && \
    apt-get install -y --no-install-recommends \
    curl \
    redis-tools \
    build-essential \
    libssl-dev \
    libffi-dev \
    python3-dev \
    redis-server

RUN pip install --upgrade pip
RUN pip3 install --no-cache-dir -r requirements.txt

COPY . /app

RUN useradd -m -U -u 1000 appuser
USER appuser

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
