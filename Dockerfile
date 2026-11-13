# Stage 1: Install dependencies into a dedicated directory.
FROM python:3.10-slim AS builder
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# Stage 2: Run the application with the installed dependencies.
FROM python:3.10-slim
WORKDIR /app

# pip's --prefix install location is copied into the runtime image's Python prefix.
COPY --from=builder /install /usr/local

COPY app.py train.py ./
COPY models/ ./models/

EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
