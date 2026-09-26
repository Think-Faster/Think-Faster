"""Отправка сообщений через бота Telegram (Bot API, метод sendMessage)."""
import html
import time

import httpx

from config import Settings

LIMIT = 4096  # длина одного сообщения в Telegram


class TelegramError(Exception):
    def __init__(self, status: int, message: str, failed: dict[str, str] | None = None):
        super().__init__(message)
        self.status = status
        self.message = message
        self.failed = failed or {}


def render(subject: str, text: str) -> str:
    return f'<b>{html.escape(subject)}</b>\n\n{html.escape(text)}'


def client(settings: Settings) -> httpx.Client:
    token = settings.telegram_bot_token.get_secret_value()
    if not token:
        raise TelegramError(503, 'не задан TELEGRAM_BOT_TOKEN')
    return httpx.Client(base_url=f'{settings.telegram_api_url}/bot{token}', timeout=settings.telegram_timeout)


def send(settings: Settings, chats: list[int | str], subject: str, text: str,
         later: list[str] | None = None) -> tuple[list[str], dict[str, str]]:
    """Шлёт сообщение в каждый чат по очереди. Возвращает, куда ушло и куда нет с причиной.

    `later` — если передан, сюда же попадают чаты, которым не ушло по временной причине (сеть,
    частота, сбой у Telegram): их стоит повторить позже (очередь, rabbit.py).
    """
    payload = {'text': render(subject, text), 'parse_mode': 'HTML', 'link_preview_options': {'is_disabled': True}}
    sent: list[str] = []
    failed: dict[str, str] = {}
    with client(settings) as api:
        for i, chat in enumerate(chats):
            try:
                reply = call(api, 'sendMessage', {**payload, 'chat_id': chat})
            except (httpx.HTTPError, TelegramError) as e:
                # сеть легла или не тот токен: остальным тоже не уйдёт, ждать таймаут на каждого незачем
                if isinstance(e, TelegramError):
                    status, reason = e.status, e.message
                else:
                    status, reason = 503, f'Telegram недоступен: {scrub(e, settings)}'
                failed.update({str(c): reason for c in chats[i:]})
                if later is not None:
                    later.extend(str(c) for c in chats[i:])
                if not sent:
                    raise TelegramError(status, reason, failed) from None
                break
            if reply.get('ok'):
                sent.append(str(chat))
            else:
                failed[str(chat)] = explain(reply)
                if later is not None and (reply.get('error_code') or 0) in (429, 500, 502, 503, 504):
                    later.append(str(chat))
    return sent, failed


def recent_chats(settings: Settings) -> list[dict]:
    """Чаты, которые писали боту за последние сутки: так узнают chat_id получателей."""
    with client(settings) as api:
        try:
            reply = call(api, 'getUpdates', {'allowed_updates': ['message', 'channel_post', 'my_chat_member']})
        except httpx.HTTPError as e:
            raise TelegramError(503, f'Telegram недоступен: {scrub(e, settings)}') from None
    if not reply.get('ok'):
        raise TelegramError(502, explain(reply))
    chats: dict[int, dict] = {}
    for update in reply['result']:
        for kind in ('message', 'channel_post', 'my_chat_member'):
            chat = update.get(kind, {}).get('chat')
            if chat:
                name = chat.get('title') or ' '.join(filter(None, (chat.get('first_name'), chat.get('last_name'))))
                chats[chat['id']] = {'chat_id': chat['id'], 'type': chat['type'],
                                     'name': name, 'username': chat.get('username')}
    return list(chats.values())


def call(api: httpx.Client, method: str, body: dict) -> dict:
    data = parse(api.post(f'/{method}', json=body))
    retry = data.get('parameters', {}).get('retry_after')
    if data.get('error_code') == 429 and retry is not None and retry <= 5:
        time.sleep(retry)
        data = parse(api.post(f'/{method}', json=body))
    return data


def parse(reply: httpx.Response) -> dict:
    if reply.status_code in (401, 404):
        # 401 — токен отозван или с ошибкой, 404 — токен не похож на токен
        raise TelegramError(502, 'Telegram не принял токен бота: проверьте TELEGRAM_BOT_TOKEN')
    try:
        return reply.json()
    except ValueError:
        raise TelegramError(502, f'Telegram ответил не по протоколу Bot API: HTTP {reply.status_code}') from None


def explain(reply: dict) -> str:
    code = reply.get('error_code')
    description = reply.get('description', 'неизвестная ошибка')
    if code == 403:
        return f'{description} — получатель должен открыть бота и нажать «Старт»'
    if code == 400 and 'chat not found' in description:
        return f'{description} — неверный chat_id или получатель ещё не писал боту'
    if code == 429:
        return f'{description} — Telegram ограничил частоту отправки'
    return description


def scrub(error: Exception, settings: Settings) -> str:
    """Текст ошибки без токена бота: он входит в адрес запроса."""
    return str(error).replace(settings.telegram_bot_token.get_secret_value(), '***') or type(error).__name__
