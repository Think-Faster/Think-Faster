# Сервис модели tf-model (INTEGRATION §13.1). Сборка из корня репозитория:
#
#   docker build -f ML/service/Dockerfile -t tf-model .
#
# В образе нет ни моделей, ни секретов. Пакет модели (bundle.py, §13.6) монтируется только для
# чтения в /models/current, рабочий том (горячий журнал, история порога, состояние правил, таблицы
# главного диспетчера) — в /work. Секреты — из Vault (§13.5), токен роли — файлом VAULT_TOKEN_FILE.
# Раскладка /app повторяет репозиторий: svc.py находит pipeline/ и docs/backend/tfkit по соседству.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8 \
    PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1

# libgomp — потоки xgboost и catboost; сети считаются на процессоре
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# версии — как в ML/requirements.txt, на которых учились модели выгрузки; torch — сборка для процессора
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch==2.5.1 \
    && pip install numpy==2.5.3 duckdb==1.5.5 polars==1.44.2 pyarrow==25.0.1 scikit-learn==1.9.1 \
       xgboost==3.4.1 catboost==1.2.10 \
       fastapi==0.141.1 uvicorn==0.54.0 confluent-kafka==2.15.1 pika==1.4.4 redis==8.1.0 \
       PyJWT==2.15.0 cryptography==50.0.1

WORKDIR /app
COPY ML/pipeline ML/pipeline
COPY ML/service ML/service
COPY ML/settings ML/settings
COPY docs/backend/tfkit/tfkit.py docs/backend/tfkit/tfkit.py
# справочники объектов и каналов (Н26: пока топик tf.ingest.reference не читается на ходу)
COPY docs/dataset/справочник_объектов_диспетчер.csv docs/dataset/справочник_каналов_датчиков.csv docs/dataset/

RUN useradd --system --uid 10001 --home-dir /work tfmodel \
    && mkdir -p /work /models && chown tfmodel /work

ENV TF_ENV=prod TF_WORK=/work TF_MODEL_BUNDLE=/models/current TF_MODEL_CLOCK=live \
    CUDA_VISIBLE_DEVICES=
USER tfmodel
VOLUME ["/work"]
EXPOSE 8000
WORKDIR /app/ML/service

HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4)"

ENTRYPOINT ["python", "main.py"]
CMD ["--loop"]
