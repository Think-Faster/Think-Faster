"""Отправка письма через SMTP."""
import smtplib
import ssl
from email.message import EmailMessage
from email.utils import formataddr, formatdate, make_msgid

from config import Settings


class MailError(Exception):
    def __init__(self, status: int, message: str, failed: dict[str, str] | None = None):
        super().__init__(message)
        self.status = status
        self.message = message
        self.failed = failed or {}


def build(settings: Settings, to: list[str], subject: str, text: str) -> EmailMessage:
    sender = settings.mail_from or settings.smtp_user
    msg = EmailMessage()
    msg['From'] = formataddr((settings.mail_from_name, sender))
    msg['To'] = ', '.join(to)
    msg['Subject'] = subject
    msg['Date'] = formatdate(localtime=True)
    msg['Message-ID'] = make_msgid(domain=sender.rpartition('@')[2] or None)
    msg.set_content(text)
    return msg


def send(settings: Settings, to: list[str], subject: str, text: str) -> dict[str, str]:
    """Отправляет одно письмо всем адресатам. Возвращает тех, кого сервер отверг, с причиной.

    Принятое сервером письмо ещё не доставлено: о несуществующем ящике Gmail
    сообщает позже, письмом на адрес отправителя.
    """
    msg = build(settings, to, subject, text)
    context = ssl.create_default_context()
    try:
        if settings.smtp_security == 'ssl':
            smtp = smtplib.SMTP_SSL(settings.smtp_host, settings.smtp_port,
                                    timeout=settings.smtp_timeout, context=context)
        else:
            smtp = smtplib.SMTP(settings.smtp_host, settings.smtp_port, timeout=settings.smtp_timeout)
        with smtp:
            if settings.smtp_security == 'starttls':
                smtp.starttls(context=context)
            if settings.smtp_user:
                smtp.login(settings.smtp_user, settings.smtp_password.get_secret_value())
            refused = smtp.send_message(msg)
    # исключения smtplib — наследники OSError, поэтому они перехватываются раньше
    except smtplib.SMTPAuthenticationError:
        raise MailError(502, 'почтовый сервер не принял логин и пароль. '
                             'Для Gmail нужен пароль приложения, а не пароль от почты') from None
    except smtplib.SMTPRecipientsRefused as e:
        raise MailError(422, 'почтовый сервер отверг всех адресатов', reasons(e.recipients)) from None
    except smtplib.SMTPSenderRefused as e:
        raise MailError(502, f'почтовый сервер отверг отправителя: {e.smtp_code} {decode(e.smtp_error)}') from None
    except smtplib.SMTPException as e:
        raise MailError(502, f'ошибка почтового сервера: {e}') from None
    except OSError as e:
        raise MailError(503, f'почтовый сервер {settings.smtp_host}:{settings.smtp_port} недоступен: {e}') from None
    return reasons(refused)


def reasons(refused: dict[str, tuple[int, bytes]]) -> dict[str, str]:
    return {address: f'{code} {decode(message)}' for address, (code, message) in refused.items()}


def decode(message: bytes | str) -> str:
    return message.decode(errors='replace') if isinstance(message, bytes) else message
