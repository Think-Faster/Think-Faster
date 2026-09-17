# -*- coding: utf-8 -*-
from django.urls import path

import views

urlpatterns = [
    path('', views.panel),                    # окно оператора
    path('events', views.events),             # поток событий по курсору
    path('stream', views.stream),             # тот же поток через SSE
    path('channels', views.channels),
    path('objects', views.objects),
    path('health', views.health),
    path('api/sensors', views.sensors),       # строки для таблички датчиков
    path('api/speed', views.speed),           # ускорение времени
    path('api/override', views.override),     # ручное значение датчика
    path('api/fault', views.fault),           # залипание, отключение, замыкание
    path('api/reset', views.reset),
]
