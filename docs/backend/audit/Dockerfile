# Сервис аудита. Контекст сборки — docs/backend (рядом лежит tfkit):
#   docker build -f docs/backend/audit/Dockerfile -t tf-audit docs/backend
# Базу готовит администратор один раз: psql -f docs/backend/audit/schema.sql (пароль роли — из Vault).
FROM python:3.12-slim

RUN pip install --no-cache-dir "psycopg[binary]==3.3.6" redis==8.1.0 fastapi==0.141.1 uvicorn==0.54.0 \
        PyJWT==2.15.0 cryptography==50.0.1

WORKDIR /app
COPY tfkit/tfkit.py /app/tfkit/tfkit.py
COPY audit/audit.py /app/audit/audit.py

RUN useradd --uid 10002 --no-create-home --shell /usr/sbin/nologin tfaudit
USER tfaudit

ENV TF_ENV=prod PYTHONUNBUFFERED=1
EXPOSE 8000
WORKDIR /app/audit
# жив, пока пишущий поток проходит цикл: файл обновляется каждые ≤ 5 с чтения потока
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import os,sys,time; sys.exit(time.time() - os.path.getmtime('/tmp/tf-audit.alive') > 120)"
ENTRYPOINT ["python", "audit.py"]
