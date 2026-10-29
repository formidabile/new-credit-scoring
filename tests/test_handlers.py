"""Тесты обработчиков бота на поддельных объектах Telegram (без сети и без токена).

Если python-telegram-bot не установлен, подставляются заглушки: обработчики обращаются только к простым
атрибутам (effective_message, text, data). Запуск: python -m pytest tests -q  или  python tests/test_handlers.py
"""
from __future__ import annotations

import asyncio
import sys
import types
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _install_stubs():
    anything = lambda *a, **k: None  # noqa: E731
    tg = types.ModuleType("telegram")
    tg.InlineKeyboardButton, tg.InlineKeyboardMarkup = anything, anything
    tg.Update = type("Update", (), {})
    err = types.ModuleType("telegram.error")
    err.NetworkError = type("NetworkError", (Exception,), {})
    ext = types.ModuleType("telegram.ext")
    for name in ["Application", "CallbackQueryHandler", "CommandHandler", "ContextTypes", "MessageHandler"]:
        setattr(ext, name, type(name, (), {}))
    ext.filters = SimpleNamespace()
    dot = types.ModuleType("dotenv")
    dot.load_dotenv = anything
    tg.error, tg.ext = err, ext
    sys.modules.update({"telegram": tg, "telegram.error": err, "telegram.ext": ext, "dotenv": dot})


try:
    import dotenv  # noqa: F401
    import telegram  # noqa: F401
    import telegram.error  # noqa: F401
    import telegram.ext  # noqa: F401
except ImportError:
    _install_stubs()

import bot.main as m  # noqa: E402
from tests.test_bot import make_predictor  # noqa: E402

NetworkError = sys.modules["telegram.error"].NetworkError


class FakeMessage:
    def __init__(self, text=None):
        self.text = text
        self.replies = []

    async def reply_text(self, text, reply_markup=None, **kwargs):
        self.replies.append(text)
        return self


def make_context(p=0.30, contrib=None):
    return SimpleNamespace(user_data={}, application=SimpleNamespace(
        bot_data={"predictor": make_predictor(p, contrib or {"BUREAU_HAS_OVERDUE_SUM": 0.5})}))


def msg_update(text=None):
    message = FakeMessage(text)
    return SimpleNamespace(message=message, effective_message=message), message


# в Telegram вопросы «да/нет» принимают слова или кнопки; номера вариантов понимает только консольная версия
ANSWERS = ["1", "150000", "400000", "20000", "400000", "1", "5", "4", "нет", "да", "3", "1", "60000", "нет", "0"]


def run(coro):
    return asyncio.run(coro)


def test_full_dialog_ends_with_result_and_clears_state():
    ctx = make_context()
    update, first = msg_update()
    run(m.start(update, ctx))
    assert "Здравствуйте" in first.replies[0] and "dialog" in ctx.user_data
    last = None
    for a in ANSWERS:
        update, last = msg_update(a)
        run(m.on_text(update, ctx))
    assert "Ориентировочная оценка" in last.replies[-1] and "учебный проект" in last.replies[-1].lower()
    assert ctx.user_data == {}  # ответы не сохраняются


def test_text_without_session_asks_to_start():
    ctx = make_context()
    update, message = msg_update("привет")
    run(m.on_text(update, ctx))
    assert "/start" in message.replies[0]


def test_edited_message_is_handled_without_crash():
    # у отредактированного сообщения update.message пуст, есть только effective_message
    ctx = make_context()
    update, _ = msg_update()
    run(m.start(update, ctx))
    message = FakeMessage("1")
    edited = SimpleNamespace(message=None, effective_message=message)
    run(m.on_text(edited, ctx))
    assert message.replies and ctx.user_data["dialog"].answers["contract_type"] == "Cash loans"


def test_validation_error_is_reported_and_dialog_stays():
    ctx = make_context()
    update, _ = msg_update()
    run(m.start(update, ctx))
    update, message = msg_update("9")  # нет такого варианта
    run(m.on_text(update, ctx))
    assert "вариант" in message.replies[0].lower()
    assert ctx.user_data["dialog"].current().key == "contract_type"


def _button(data, message=None):
    async def noop(*a, **k):
        return None
    query = SimpleNamespace(data=data, message=message or FakeMessage(), answer=noop, edit_message_reply_markup=noop)
    return SimpleNamespace(callback_query=query), query


def test_button_for_old_question_is_ignored():
    ctx = make_context()
    update, _ = msg_update()
    run(m.start(update, ctx))
    update, query = _button("contract_type:Cash loans")
    run(m.on_button(update, ctx))
    assert ctx.user_data["dialog"].current().key == "income"
    n = len(query.message.replies)
    update, query2 = _button("contract_type:Revolving loans")  # повторное нажатие на кнопку пройденного вопроса
    run(m.on_button(update, ctx))
    assert ctx.user_data["dialog"].answers["contract_type"] == "Cash loans" and not query2.message.replies
    assert n >= 1


def test_button_without_session():
    update, query = _button("contract_type:Cash loans")
    run(m.on_button(update, make_context()))
    assert "/start" in query.message.replies[0]


def test_other_content_gets_hint():
    update, message = msg_update()
    run(m.on_other(update, make_context()))
    assert "только текст" in message.replies[0]


def test_out_of_range_answers_give_hint_and_clear_state():
    ctx = make_context()
    update, _ = msg_update()
    run(m.start(update, ctx))
    answers = ["1", "150000", "400000", "100", "-", "1", "5", "4", "нет", "да", "0"]  # платёж 100 при кредите 400000
    last = None
    for a in answers:
        update, last = msg_update(a)
        run(m.on_text(update, ctx))
    text = last.replies[-1]
    assert "слишком далеки" in text and "Подсказка" in text and "3–12%" in text
    assert ctx.user_data == {}


def test_rare_income_type_ends_dialog_early():
    ctx = make_context()
    update, _ = msg_update()
    run(m.start(update, ctx))
    last = None
    for a in ["1", "150000", "400000", "20000", "-", "8"]:  # 8 = «Не работаю»
        update, last = msg_update(a)
        run(m.on_text(update, ctx))
    assert "слишком редко" in last.replies[-1] and "/start" in last.replies[-1]
    assert ctx.user_data == {}


def test_error_handler():
    ctx = SimpleNamespace(error=RuntimeError("boom"))
    message = FakeMessage()
    update = SimpleNamespace(effective_message=message)  # не настоящий Update: ответ пользователю не отправляется
    run(m.on_error(update, ctx))  # не должен падать
    assert message.replies == []
    run(m.on_error(None, SimpleNamespace(error=NetworkError("net"))))  # сетевые ошибки не падают и ничего не отвечают


if __name__ == "__main__":
    import traceback

    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    failed = []
    for name, fn in tests:
        try:
            fn()
            print("ok  ", name)
        except Exception:
            failed.append(name)
            print("FAIL", name)
            traceback.print_exc()
    print(f"\nпройдено {len(tests) - len(failed)} из {len(tests)}")
    if failed:
        print("Не прошли:", ", ".join(failed))
        sys.exit(1)
