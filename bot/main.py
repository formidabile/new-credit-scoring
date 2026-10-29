"""Telegram-бот: диалог с пользователем и вывод ориентировочной оценки.

Запуск из корня репозитория: python -m bot.main
Токен берётся из переменной окружения TELEGRAM_BOT_TOKEN или из файла .env в корне репозитория (python-dotenv).
Токен нельзя коммитить.
"""
from __future__ import annotations

import asyncio
import logging
import os
import time
from pathlib import Path

from dotenv import load_dotenv
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.error import NetworkError
from telegram.ext import Application, CallbackQueryHandler, CommandHandler, ContextTypes, MessageHandler, filters

from bot.dialog import Dialog, Question
from bot.predictor import DISCLAIMER, OutOfRangeError, Predictor, format_message

logging.basicConfig(format="%(asctime)s %(levelname)s %(name)s: %(message)s", level=logging.INFO)
# httpx на уровне INFO пишет в лог адреса запросов, а в них зашит токен бота
logging.getLogger("httpx").setLevel(logging.WARNING)
log = logging.getLogger("bot")

ROOT = Path(__file__).resolve().parents[1]
PLACEHOLDER = "put_your_token_here"

INTRO = (
    "Здравствуйте! Я задам несколько вопросов и дам ориентировочную оценку риска невозврата кредита.\n\n"
    "Суммы вводите в условных единицах датасета Home Credit (валюта в данных не указана; "
    "типичные значения: доход около 150 000, платёж около 25 000, кредит около 500 000; доход указывайте за тот же период, что и платёж). Можно писать 150000 или 150k.\n\n"
    + DISCLAIMER + "\n\nКоманды: /start начать заново, /cancel отменить, /help справка."
)
HELP = (
    "Я работаю по вашим ответам на 11–16 вопросов. Модель простая и опирается только на то, что вы указали: "
    "оценка ориентировочная. Я не даю советов, как изменить ответы, чтобы получить одобрение.\n\n"
    "Чтобы начать, нажмите /start."
)


def keyboard(dialog: Dialog, q: Question) -> InlineKeyboardMarkup | None:
    options = dialog.options(q)
    if not options:
        return None
    # callback_data содержит ключ вопроса, чтобы нажатия на старые кнопки не попадали в новый вопрос
    rows = [[InlineKeyboardButton(label, callback_data=f"{q.key}:{value}")] for value, label in options]
    return InlineKeyboardMarkup(rows)


async def send(message, text: str, **kwargs):
    """Отправка с замером времени: помогает отличить медленную сеть от медленного кода."""
    t0 = time.perf_counter()
    result = await message.reply_text(text, **kwargs)
    log.info("отправка сообщения заняла %.0f мс", (time.perf_counter() - t0) * 1000)
    return result


async def ask_next(message, context: ContextTypes.DEFAULT_TYPE) -> None:
    dialog: Dialog = context.user_data["dialog"]
    q = dialog.current()
    if q is None:
        await finish(message, context)
        return
    await send(message, q.text, reply_markup=keyboard(dialog, q))


async def finish(message, context: ContextTypes.DEFAULT_TYPE) -> None:
    dialog: Dialog = context.user_data["dialog"]
    if dialog.stop_reason:  # опрос завершён досрочно (например, редкий тип занятости)
        context.user_data.clear()
        await send(message, dialog.stop_reason + "\n\nНажмите /start, чтобы пройти опрос заново.")
        return
    predictor: Predictor = context.application.bot_data["predictor"]
    try:
        t0 = time.perf_counter()
        result = await asyncio.to_thread(predictor.assess, dialog.answers)
        log.info("оценка заняла %.0f мс", (time.perf_counter() - t0) * 1000)
        text = format_message(result)
    except OutOfRangeError as e:
        text = f"{e}\n\nНажмите /start, чтобы пройти опрос заново."
    except ValueError as e:
        text = f"Не удалось выполнить оценку: {e}\n\nНажмите /start, чтобы пройти опрос заново."
    except Exception:  # неожиданная ошибка: пользователю общий текст, в лог трассировка без ответов пользователя
        log.exception("Ошибка при оценке")
        text = "Что-то пошло не так. Попробуйте ещё раз: /start."
    context.user_data.clear()  # ответы не сохраняются
    await send(message, text)


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    context.user_data["dialog"] = Dialog()
    message = update.effective_message
    await send(message, INTRO)
    await ask_next(message, context)


async def help_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await send(update.effective_message, HELP)


async def cancel(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    context.user_data.clear()
    await send(update.effective_message, "Опрос отменён. Ответы не сохранены. /start, чтобы начать заново.")


async def on_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.effective_message  # у отредактированных сообщений update.message пуст, поэтому не он
    if message is None:
        return
    dialog: Dialog | None = context.user_data.get("dialog")
    if dialog is None:
        await send(message, "Нажмите /start, чтобы начать опрос.")
        return
    error = dialog.submit(message.text or "")
    if error:
        await send(message, error)
        return
    await ask_next(message, context)


async def on_other(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Фото, голосовые, стикеры и т.п.: сообщаем, что бот понимает только текст и кнопки."""
    if update.effective_message:
        await send(update.effective_message, "Я понимаю только текст и нажатия на кнопки. Ответьте на вопрос текстом или нажмите /help.")


async def on_button(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await query.answer()
    dialog: Dialog | None = context.user_data.get("dialog")
    if dialog is None:
        await send(query.message, "Сессия закончилась. Нажмите /start.")
        return
    key, _, value = (query.data or "").partition(":")
    current = dialog.current()
    if current is None or current.key != key:  # нажали кнопку от уже пройденного вопроса
        return
    error = dialog.submit(value)
    if error:
        await send(query.message, error)
        return
    await query.edit_message_reply_markup(reply_markup=None)
    await ask_next(query.message, context)


async def on_error(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Сетевые сбои Telegram логируем коротко; остальные ошибки логируем с трассировкой и отвечаем пользователю."""
    if isinstance(context.error, NetworkError):
        log.warning("Сетевая ошибка при обращении к Telegram: %s", context.error)
        return
    log.error("Необработанная ошибка", exc_info=context.error)
    if isinstance(update, Update) and update.effective_message:
        try:
            await update.effective_message.reply_text("Что-то пошло не так. Нажмите /start, чтобы начать заново.")
        except Exception:
            pass


def get_token() -> str:
    token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    if not token:
        env_file = ROOT / ".env"
        where = ("файл .env найден, но в нём нет строки TELEGRAM_BOT_TOKEN=... (проверьте, что файл сохранён)"
                 if env_file.exists() else f"файла {env_file} нет")
        raise SystemExit(f"Не задан TELEGRAM_BOT_TOKEN: в переменных окружения его нет, {where}.")
    if token == PLACEHOLDER:
        raise SystemExit("В .env остался шаблонный токен. Вставьте настоящий токен от @BotFather.")
    return token


def make_ipv4_requests():
    """Клиенты Telegram, привязанные к IPv4 (если IPv6 в сети сломан и соединение зависает на TLS)."""
    import httpx
    from telegram.request import HTTPXRequest

    def make(read_timeout):
        return HTTPXRequest(
            connect_timeout=20, read_timeout=read_timeout, write_timeout=20, pool_timeout=20,
            httpx_kwargs={"transport": httpx.AsyncHTTPTransport(local_address="0.0.0.0")},
        )

    return make(20), make(30)


def main() -> None:
    load_dotenv(ROOT / ".env")  # явный путь: не зависит от того, откуда запущен скрипт; переменные окружения приоритетнее
    builder = Application.builder().token(get_token())
    proxy = os.getenv("TELEGRAM_PROXY", "").strip()  # например, http://127.0.0.1:8080 или socks5://127.0.0.1:1080
    force_ipv4 = os.getenv("TELEGRAM_FORCE_IPV4", "").strip().lower() in {"1", "true", "yes", "да"}

    if force_ipv4 and not proxy:
        request, updates_request = make_ipv4_requests()
        builder = builder.request(request).get_updates_request(updates_request)
        log.info("Соединение только по IPv4")
    else:
        if force_ipv4 and proxy:
            log.warning("TELEGRAM_FORCE_IPV4 игнорируется, потому что задан прокси")
        builder = (builder.connect_timeout(20).read_timeout(20).write_timeout(20).pool_timeout(20)
                   .get_updates_connect_timeout(20).get_updates_read_timeout(30))
        if proxy:
            try:
                builder = builder.proxy(proxy).get_updates_proxy(proxy)
            except AttributeError:  # старые версии python-telegram-bot
                builder = builder.proxy_url(proxy).get_updates_proxy_url(proxy)
            log.info("Используется прокси")

    app = builder.build()
    app.bot_data["predictor"] = Predictor()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_cmd))
    app.add_handler(CommandHandler("cancel", cancel))
    app.add_handler(CallbackQueryHandler(on_button))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND & filters.UpdateType.MESSAGE, on_text))
    app.add_handler(MessageHandler(~filters.TEXT & ~filters.COMMAND & filters.UpdateType.MESSAGE, on_other))
    app.add_error_handler(on_error)
    log.info("Бот запущен")
    # сообщения, накопленные, пока бот был выключен, не обрабатываем: состояние диалога хранится в памяти
    app.run_polling(bootstrap_retries=5, drop_pending_updates=True)


if __name__ == "__main__":
    main()
