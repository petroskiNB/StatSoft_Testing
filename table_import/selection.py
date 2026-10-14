"""
ДОП. ФУНКЦИЯ - выбор интересующих столбцов
Общие правила выбора столбцов для всех источников
"""

from collections.abc import Sequence
from .exceptions import ImportConfigurationError, SourceReadError


def normalize_columns(columns: Sequence[str] | None) -> tuple[str, ...] | None:
    if columns is None:
        return None
    if isinstance(columns, (str, bytes)) or not isinstance(columns, Sequence):
        raise ImportConfigurationError("columns должен быть списком или кортежем имен")
    result = tuple(columns)
    if not result or any(not isinstance(name, str) or not name for name in result):
        raise ImportConfigurationError("columns должен содержать непустые строковые имена")
    if len(set(result)) != len(result):
        raise ImportConfigurationError("Имена в columns не должны повторяться")
    return result


def selected_names(available, columns, path) -> list[str]:
    names = list(available)
    if columns is None:
        return names
    missing = [name for name in columns if name not in names]
    if missing:
        raise SourceReadError(
            path, "выбор столбцов",
            f"Столбцы не найдены: {missing}. Доступны: {names}")
    return list(columns)
