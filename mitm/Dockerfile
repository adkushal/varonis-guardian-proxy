FROM mitmproxy/mitmproxy:10.4.2

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY simple_classifier.py guardian_client.py strategies.py chain.py decorators.py addon.py ./

ENV GUARDIAN_URL=http://guardian:8000
ENV GUARDIAN_CRITERIA=harm,profanity
ENV GUARDIAN_FAIL_CLOSED=1

# Reverse proxy to OpenAI. The listener accepts TLS from NGINX using /certs/mitm.pem
# (signed by the local CA); the upstream connection to OpenAI is verified normally.
EXPOSE 8080
CMD ["mitmdump", "-s", "/app/addon.py", \
     "--mode", "reverse:https://api.openai.com", \
     "--listen-host", "0.0.0.0", "--listen-port", "8080", \
     "--set", "certs=*=/certs/mitm.pem", \
     "--set", "block_global=false"]
