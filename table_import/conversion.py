"""Явные строгие преобразования значений, уже прочитанных источником"""

from datetime import date, datetime
from decimal import Decimal, InvalidOperation
import pandas as pd

SUPPORTED_DTYPES = {"Int64", "int64", "Float64", "float64", "string", "boolean", "datetime64[ns]"}


def _integer(value):
    if pd.isna(value):
        return pd.NA
    if isinstance(value, bool):
        return int(value)
    try:
        number = Decimal(str(value))
    except InvalidOperation as exc:
        raise ValueError("Ожидается целое число") from exc
    if not number.is_finite() or number != number.to_integral_value():
        raise ValueError("Дробное или бесконечное значение нельзя преобразовать в целое")
    if number < -(2 ** 63) or number > 2 ** 63 - 1:
        raise ValueError("Число выходит за диапазон int64")
    return int(number)


def _boolean(value):
    if pd.isna(value):
        return pd.NA
    if isinstance(value, str):
        value = value.strip().lower()
        if value in ("true", "1"):
            return True
        if value in ("false", "0"):
            return False
    elif value == 1 or value == 0:
        return bool(value)
    raise ValueError("Ожидается true/false или 1/0")


def _datetime(value):
    if pd.isna(value):
        return pd.NaT
    if not isinstance(value, (str, date, datetime)):
        raise ValueError("Ожидается дата или строка ISO 8601, а не числовая метка времени")
    result = pd.to_datetime(value, format="ISO8601", errors="raise")
    if pd.isna(result):
        raise ValueError("Непустое значение не должно превращаться в NaT")
    if result.tzinfo is not None:
        raise ValueError("В этой версии поддерживаются даты без часового пояса")
    return result


def convert_series(series: pd.Series, target: str) -> pd.Series:
    if target in {"Int64", "int64"}:
        # Создание nullable-массива не теряет точность больших целых при NULL
        values = pd.array([_integer(value) for value in series], dtype="Int64")
        return pd.Series(values, index=series.index, name=series.name).astype(target)
    if target == "boolean":
        return pd.Series(
            pd.array([_boolean(value) for value in series], dtype="boolean"),
            index=series.index, name=series.name,
        )
    if target == "datetime64[ns]":
        return pd.Series(
            [_datetime(value) for value in series], index=series.index,
            name=series.name, dtype=target,
        )
    if target in {"Float64", "float64"}:
        return pd.to_numeric(series, errors="raise").astype(target)
    return series.astype(target)
