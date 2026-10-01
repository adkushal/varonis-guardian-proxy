FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app.py .

ENV GUARDIAN_MODE=mock
ENV GUARDIAN_MODEL=ibm-granite/granite-guardian-4.1-8b
EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
