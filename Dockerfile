FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 MPLCONFIGDIR=/tmp/matplotlib
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY aavail/ ./aavail/
COPY scripts/ ./scripts/
COPY tests/ ./tests/
COPY run_tests.py ./run_tests.py
COPY data/ ./data/
RUN mkdir -p /app/logs /app/models /app/reports && python -m scripts.bootstrap_demo && python run_tests.py
EXPOSE 8080
CMD ["uvicorn", "aavail.api:app", "--host", "0.0.0.0", "--port", "8080"]
