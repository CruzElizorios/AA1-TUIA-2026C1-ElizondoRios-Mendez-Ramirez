FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY inferencia.py .
COPY *.joblib .
COPY threshold.json .

ENTRYPOINT ["python", "inferencia.py"]
