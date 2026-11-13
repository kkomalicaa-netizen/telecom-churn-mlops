# Stage 1: Build stage (installs heavy dependencies and packages)
FROM python:3.10-slim AS builder
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Stage 2: Final lightweight runtime stage
FROM python:3.10-slim
WORKDIR /app

# Copy only the installed python packages from the builder stage
COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH

# Copy application code and model artifacts
COPY app.py train.py ./
COPY models/ ./models/

EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]