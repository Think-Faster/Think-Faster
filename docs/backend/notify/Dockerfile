# Сервис уведомлений. Контекст сборки — docs/backend (рядом лежит tfkit):
#   docker build -f docs/backend/notify/Dockerfile -t tf-notify docs/backend
# Файл .env в образ не попадает: в контуре секреты — только из Vault (config.py).
FROM python:3.12-slim

WORKDIR /app/notify
COPY notify/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY tfkit/tfkit.py /app/tfkit/tfkit.py
COPY notify/config.py notify/mailer.py notify/main.py notify/rabbit.py notify/telegram_bot.py ./

RUN useradd --uid 10003 --no-create-home --shell /usr/sbin/nologin tfnotify
USER tfnotify

ENV TF_ENV=prod TF_NOTIFY_ENV_FILE= PYTHONUNBUFFERED=1
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import sys,urllib.request; sys.exit(urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4).status != 200)"
ENTRYPOINT ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
