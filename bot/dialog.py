"""Диалог бота: порядок вопросов, разбор и проверка ответов. Не зависит от Telegram, поэтому тестируется отдельно."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable

from src.bot_features import (CONTRACT_TYPES, EDUCATION_TYPES, INCOME_TYPES, NO_EMPLOYMENT_TYPES, RARE_INCOME_TYPES,
                              rare_income_message)

_MULT = {"k": 1e3, "к": 1e3, "тыс": 1e3, "m": 1e6, "м": 1e6, "млн": 1e6}
_YES = {"да", "д", "y", "yes", "+", "есть"}
_NO = {"нет", "н", "n", "no", "-", "нету"}
_SKIP = {"-", "нет", "не знаю", "0"}


def parse_number(text: str) -> float:
    """'150000', '150 000', '150k', '1,5 млн' -> число."""
    t = text.strip().lower().replace("\u00a0", " ").replace(" ", "").replace(",", ".")
    m = re.fullmatch(r"(\d+(?:\.\d+)?)(k|к|тыс\.?|m|м|млн\.?)?", t)
    if not m:
        raise ValueError("Введите число, например 150000 или 150k.")
    return float(m.group(1)) * _MULT.get((m.group(2) or "").rstrip("."), 1)


@dataclass
class Question:
    key: str
    text: str
    kind: str  # number | int | choice | yn | bool
    choices: dict | None = None
    minimum: float | None = None
    maximum: float | None = None
    max_from: str | None = None  # верхняя граница берётся из другого ответа
    optional: bool = False
    skip_if: Callable[[dict], bool] | None = None
    default: object = None


def _no_history(a):
    return a.get("bureau_count", 0) == 0


QUESTIONS = [
    Question("contract_type", "Какой тип кредита вы рассматриваете?", "choice", choices=CONTRACT_TYPES),
    Question("income", "Ваш доход в условных единицах данных за тот же период, что и платёж по кредиту (например, "
             "ежемесячный доход, если платёж ежемесячный). Типичные значения в данных: доход около 150 000, платёж около "
             "25 000, кредит около 500 000. Можно писать 150000 или 150k.", "number", minimum=1),
    Question("credit", "Сумма кредита?", "number", minimum=1),
    Question("annuity", "Размер регулярного платежа по кредиту за один период (обычно месяц)?", "number", minimum=1),
    Question("goods_price", "Стоимость товара, на который берётся кредит. Для кредита наличными укажите сумму кредита. "
             "Если неизвестно, напишите «-».", "number", minimum=1, optional=True),
    Question("income_type", "Ваш тип занятости?", "choice", choices=INCOME_TYPES),
    Question("employment_years", "Сколько лет вы работаете на текущем месте? (например, 0.5 или 7)", "number",
             minimum=0, maximum=60, skip_if=lambda a: a["income_type"] in NO_EMPLOYMENT_TYPES),
    Question("education", "Ваше образование?", "choice", choices=EDUCATION_TYPES),
    Question("own_car", "Есть ли у вас автомобиль?", "yn"),
    Question("car_age", "Сколько лет автомобилю?", "number", minimum=0, maximum=80,
             skip_if=lambda a: a["own_car"] != "Y"),
    Question("own_realty", "Есть ли у вас недвижимость (квартира, дом)?", "yn"),
    Question("bureau_count", "Сколько всего кредитов и кредитных карт вы брали в других банках за всё время? "
             "(0, если не брали)", "int", minimum=0, maximum=200),
    Question("bureau_active", "Сколько из них активны сейчас?", "int", minimum=0, max_from="bureau_count",
             skip_if=_no_history, default=0),
    Question("bureau_debt", "Какова общая сумма вашей задолженности по ним сейчас? (0, если долгов нет)", "number",
             minimum=0, skip_if=_no_history, default=0),
    Question("bureau_overdue", "Были ли по этим кредитам просрочки платежей?", "bool",
             skip_if=_no_history, default=False),
    Question("bureau_last_year", "Сколько из них вы открыли за последний год?", "int", minimum=0,
             max_from="bureau_count", skip_if=_no_history, default=0),
]


class Dialog:
    def __init__(self):
        self.answers: dict = {}
        self.index = 0
        self.stop_reason: str | None = None  # причина досрочного завершения опроса

    def current(self) -> Question | None:
        """Следующий вопрос; пропущенным вопросам проставляются значения по умолчанию."""
        if self.stop_reason:
            return None
        while self.index < len(QUESTIONS):
            q = QUESTIONS[self.index]
            if q.skip_if and q.skip_if(self.answers):
                self.answers[q.key] = q.default
                self.index += 1
                continue
            return q
        return None

    def options(self, q: Question) -> list[tuple[str, str]]:
        """Варианты для кнопок: (значение, подпись)."""
        if q.kind == "choice":
            return list(q.choices.items())
        if q.kind in ("yn", "bool"):
            return [("Y", "Да"), ("N", "Нет")]
        return []

    def submit(self, raw: str) -> str | None:
        """Принять ответ на текущий вопрос. Возвращает текст ошибки или None."""
        q = self.current()
        if q is None:
            return "Все вопросы уже заданы."
        try:
            value = self._parse(q, raw.strip())
        except ValueError as e:
            return str(e)
        self.answers[q.key] = value
        self.index += 1
        if q.key == "income_type" and value in RARE_INCOME_TYPES:
            self.stop_reason = rare_income_message(value)
        return None

    def _parse(self, q: Question, raw: str):
        if q.kind == "choice":
            return self._choice(q, raw)
        if q.kind in ("yn", "bool"):
            low = raw.lower()
            if low in _YES or raw == "Y":
                value = "Y"
            elif low in _NO or raw == "N":
                value = "N"
            else:
                raise ValueError("Ответьте «да» или «нет» или нажмите кнопку.")
            return value == "Y" if q.kind == "bool" else value
        if q.optional and raw.lower() in _SKIP:
            return None
        value = parse_number(raw)
        if q.kind == "int":
            if value != int(value):
                raise ValueError("Нужно целое число.")
            value = int(value)
        maximum = self.answers[q.max_from] if q.max_from else q.maximum
        if q.minimum is not None and value < q.minimum:
            raise ValueError(f"Значение должно быть не меньше {q.minimum:g}.")
        if maximum is not None and value > maximum:
            raise ValueError(f"Значение должно быть не больше {maximum:g}.")
        return value

    @staticmethod
    def _choice(q: Question, raw: str):
        if raw in q.choices:
            return raw
        for key, label in q.choices.items():
            if raw.lower() == label.lower():
                return key
        if raw.isdigit() and 1 <= int(raw) <= len(q.choices):
            return list(q.choices)[int(raw) - 1]
        raise ValueError("Выберите вариант кнопкой или введите его номер.")
