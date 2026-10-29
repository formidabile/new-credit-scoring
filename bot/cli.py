"""Консольная версия бота: тот же диалог и тот же Predictor, но в терминале и без Telegram.

Запуск из корня репозитория: python -m bot.cli
Нужна, чтобы проверить модель, файлы из models/ и тексты ответов, когда Telegram недоступен.
"""
from __future__ import annotations

from bot.dialog import Dialog
from bot.predictor import OutOfRangeError, Predictor, format_message


def run(predictor: Predictor, input_fn=input, print_fn=print) -> None:
    print_fn("Консольная проверка бота. Суммы вводите в условных единицах датасета (типичные значения: доход 150000, платёж 25000, кредит 500000; доход и платёж за один период).")
    print_fn("Пустой ввод или Ctrl+C завершают работу.\n")
    dialog = Dialog()
    while (q := dialog.current()) is not None:
        print_fn(q.text)
        for i, (_, label) in enumerate(dialog.options(q), 1):
            print_fn(f"  {i}. {label}")
        raw = input_fn("> ").strip()
        if not raw:
            print_fn("Отменено.")
            return
        # для вопросов да/нет номер означает 1 = Да, 2 = Нет
        if q.kind in ("yn", "bool") and raw in ("1", "2"):
            raw = "Y" if raw == "1" else "N"
        error = dialog.submit(raw)
        if error:
            print_fn(error)
    print_fn("")
    if dialog.stop_reason:
        print_fn(dialog.stop_reason)
        return
    try:
        result = predictor.assess(dialog.answers)
        print_fn(format_message(result))
        print_fn(f"\n[отладка] точная вероятность: {result.probability:.1%}")  # в Telegram-боте не показывается
    except OutOfRangeError as e:
        print_fn(str(e))
    except ValueError as e:
        print_fn(f"Не удалось выполнить оценку: {e}")


if __name__ == "__main__":
    run(Predictor())
