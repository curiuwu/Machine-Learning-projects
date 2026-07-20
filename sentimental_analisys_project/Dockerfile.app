FROM python:3.12-slim

WORKDIR /app

COPY requirements.app.txt .
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.13.0 || \
    pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.13.0+cpu
RUN pip install --no-cache-dir -r requirements.app.txt

COPY . .

ENV PYTHONPATH=/app
