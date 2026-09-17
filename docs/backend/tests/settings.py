# -*- coding: utf-8 -*-
"""Минимальные настройки стенда: без базы, без приложений, один шаблон.

Стенд одноразовый и держит состояние в памяти процесса, поэтому ни модели,
ни миграции ему не нужны.
"""
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

# стенд поднимается только локально, настоящих секретов тут нет и быть не должно
SECRET_KEY = os.environ.get('TF_SECRET_KEY', 'stand-local-only-not-a-secret')
DEBUG = True
ALLOWED_HOSTS = ['*']

ROOT_URLCONF = 'urls'
INSTALLED_APPS = []
MIDDLEWARE = ['django.middleware.common.CommonMiddleware']

TEMPLATES = [{
    'BACKEND': 'django.template.backends.django.DjangoTemplates',
    'DIRS': [BASE_DIR / 'templates'],
    'APP_DIRS': False,
    'OPTIONS': {
        'context_processors': [],
        # с Django 4.1 шаблоны кешируются даже в DEBUG, а стенд поднимается
        # с --noreload, и правка panel.html не подхватывалась бы до перезапуска
        'loaders': ['django.template.loaders.filesystem.Loader'],
    },
}]

DATABASES = {}
USE_TZ = False
LANGUAGE_CODE = 'ru-ru'
DEFAULT_CHARSET = 'utf-8'
