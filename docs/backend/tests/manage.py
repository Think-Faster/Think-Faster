#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Запуск стенда: python manage.py runserver

Настройки движка читаются из переменных окружения: TF_DATA (папка датасета),
TF_SPEED, TF_JITTER, TF_FAULT_INTERVAL, TF_SEED. Ускорение меняется и на ходу,
из панели.
"""
import os
import sys


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, here)
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'settings')
    from django.core.management import execute_from_command_line
    execute_from_command_line(sys.argv)


if __name__ == '__main__':
    main()
