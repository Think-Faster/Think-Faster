# Воронка показаний. Контекст сборки — docs/backend (рядом лежит tfkit):
#   docker build -f docs/backend/funnel/Dockerfile -t tf-funnel docs/backend
# Пароль Kafka (tf-funnel) и техучётки шины — из Vault (secret/tf/kafka, secret/tf/funnel).
FROM python:3.12-slim

RUN pip install --no-cache-dir confluent-kafka==2.15.1 redis==8.1.0 fastapi==0.141.1 uvicorn==0.54.0 \
        PyJWT==2.15.0 cryptography==50.0.1

WORKDIR /app
COPY tfkit/tfkit.py /app/tfkit/tfkit.py
COPY funnel/funnel.py /app/funnel/funnel.py

RUN useradd --uid 10004 --no-create-home --shell /usr/sbin/nologin tffunnel
USER tffunnel

ENV TF_ENV=prod PYTHONUNBUFFERED=1
EXPOSE 8000
WORKDIR /app/funnel
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import sys,urllib.request; sys.exit(urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4).status != 200)"
ENTRYPOINT ["python", "funnel.py"]
