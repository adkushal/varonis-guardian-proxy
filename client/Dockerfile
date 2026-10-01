FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY chat_client.py .

ENV PROXY_BASE_URL=https://nginx:8443
ENV SSL_CERT_FILE=/certs/ca.crt

ENTRYPOINT ["python", "chat_client.py"]
