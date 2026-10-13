"""Исключения, которые может обрабатывать пользователь библиотеки."""

from pathlib import Path

class TableImportError(Exception):
    """ Базовая ошибка библиотеки """


class SourceReadError(TableImportError):
    """ Ошибка открытия или чтения источника """

    def __init__(self, path: Path, operation: str, reason: str):
        self.path = path
        self.operation = operation
        self.reason = reason
        super().__init__(
            f"Не удалось выполнить {operation} для '{path}': {reason}"
        )


class ImportStateError(TableImportError):
    """Операция недопустима в текущем состоянии сессии."""


class ImportConfigurationError(TableImportError, ValueError):
    """Некорректные настройки импорта."""


class TypeConversionError(TableImportError):
    """Строка нумеруется с 1 среди строк данных, без заголовка."""

    def __init__(self, source, sheet, column, target_type, reason, row=None, value=None):
        self.source = source
        self.sheet = sheet
        self.column = column
        self.target_type = target_type
        self.row = row
        self.value = value
        self.reason = reason
        location = f", строка данных {row}, значение {value!r}" if row is not None else ""
        super().__init__(
            f"Источник '{source}', лист '{sheet}', столбец '{column}'{location}: "
            f"невозможно преобразовать в {target_type}: {reason}"
        )
