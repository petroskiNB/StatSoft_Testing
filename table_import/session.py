"""Управление импортом независимо от формата источника."""

from collections.abc import Mapping
import pandas as pd
from .conversion import SUPPORTED_DTYPES, convert_series
from .exceptions import (ImportConfigurationError, ImportStateError, SourceReadError, TypeConversionError,)
from .models import ImportProgress, SheetMetadata
from .source import TableSource


class ImportSession:
    """
    Синхронная сессия
    Пауза сохраняет открытый читатель в текущем процессе
    """

    def __init__(
        self, source: TableSource, sheet: str, chunk_size: int = 1000,
        dtype: Mapping[str, str] | None = None, *, metadata: SheetMetadata | None = None,
    ):
        if not isinstance(sheet, str) or not sheet:
            raise ImportConfigurationError("sheet должен быть непустой строкой")
        if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size <= 0:
            raise ImportConfigurationError("chunk_size должен быть положительным целым числом")
        if dtype is not None and not isinstance(dtype, Mapping):
            raise ImportConfigurationError("dtype должен быть словарём: столбец → тип")
        self._dtype = dict(dtype or {})
        for column, target in self._dtype.items():
            if not isinstance(column, str) or not isinstance(target, str) or target not in SUPPORTED_DTYPES:
                raise ImportConfigurationError(
                    f"Некорректный тип для '{column}': {target!r}. "
                    f"Поддерживаются: {sorted(SUPPORTED_DTYPES)}"
                )
        self._source = source
        if metadata is not None:
            if not isinstance(metadata, SheetMetadata):
                raise ImportConfigurationError("metadata должен быть объектом SheetMetadata")
            if metadata.name != sheet:
                raise ImportConfigurationError("Имя листа metadata не совпадает с sheet")
            total = metadata.row_count
            if total is not None and (isinstance(total, bool) or not isinstance(total, int) or total < 0):
                raise ImportConfigurationError("row_count должен быть неотрицательным целым или None")
            if not isinstance(metadata.row_count_is_estimate, bool):
                raise ImportConfigurationError("row_count_is_estimate должен быть bool")
            if total is None and metadata.row_count_is_estimate:
                raise ImportConfigurationError("Нельзя пометить неизвестное число строк как оценку")
        self._total_rows = metadata.row_count if metadata is not None else None
        self._total_is_estimate = metadata.row_count_is_estimate if metadata is not None else False
        self._sheet = sheet
        self._chunk_size = chunk_size
        self._reader = None
        self._parts = []
        self._columns = None
        self._result = None
        self._rows = 0
        self._chunks = 0
        self._status = "ready"

    @property
    def progress(self) -> ImportProgress:
        complete = self._status == "completed"
        total = self._rows if complete else self._total_rows
        exceeded = not complete and total is not None and self._rows > total
        fraction = None
        if complete:
            fraction = 1.0
        elif total is not None and total > 0 and not exceeded:
            fraction = self._rows / total
        return ImportProgress(
            status=self._status, rows_imported=self._rows,
            chunks_imported=self._chunks,
            total_rows=total,
            fraction=fraction,
            total_rows_is_estimate=self._total_is_estimate if not complete else False,
            total_rows_exceeded=exceeded,
        )

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def __iter__(self):
        return self

    def _close_reader(self):
        if self._reader is not None:
            reader, self._reader = self._reader, None
            reader.close()

    def __next__(self) -> pd.DataFrame:
        if self._status in {"paused", "completed"}:
            raise StopIteration
        if self._status in {"closed", "failed"}:
            raise ImportStateError(f"Нельзя продолжить сессию в состоянии {self._status}")
        try:
            if self._reader is None:
                self._reader = self._source.iter_chunks(
                    sheet=self._sheet, chunk_size=self._chunk_size,
                )
            self._status = "running"
            try:
                chunk = next(self._reader)
            except StopIteration:
                self._finish()
                chunk = None
            if chunk is not None:
                if not isinstance(chunk, pd.DataFrame) or not chunk.columns.is_unique:
                    raise SourceReadError(
                        self._source_name, "импорт", "Ожидается DataFrame с уникальными столбцами",
                    )
                columns = list(chunk.columns)
                if self._columns is not None and columns != self._columns:
                    raise SourceReadError(
                        self._source_name, "импорт", "Набор столбцов изменился между порциями",
                    )
                converted = self._convert(chunk, self._rows)
                self._columns = columns
                self._parts.append(converted.copy(deep=True))
                self._rows += len(converted)
                self._chunks += int(len(converted) > 0)
                return converted
        except Exception:
            # не пропускаем незаметно плохие строки
            self._status = "failed"
            self._parts.clear()
            self._close_reader()
            raise
        raise StopIteration

    @property
    def _source_name(self):
        return getattr(self._source, "path", type(self._source).__name__)

    def _convert(self, chunk: pd.DataFrame, offset: int) -> pd.DataFrame:
        result = chunk.copy(deep=True)
        for column, target in self._dtype.items():
            if column not in chunk.columns:
                raise TypeConversionError(
                    self._source_name, self._sheet, column, target, "Столбец отсутствует"
                )
            try:
                result[column] = convert_series(chunk[column], target)
            except (ValueError, TypeError, OverflowError) as exc:
                # Диагностика по значениям запускается только при ошибке
                row, value = None, None
                for position in range(len(chunk)):
                    try:
                        convert_series(chunk[column].iloc[position:position + 1], target)
                    except (ValueError, TypeError, OverflowError):
                        row = offset + position + 1
                        value = chunk[column].iloc[position]
                        break
                raise TypeConversionError(
                    self._source_name, self._sheet, column, target, str(exc), row, value,
                ) from exc
        return result

    def _finish(self):
        self._close_reader()
        combined = pd.concat(self._parts, ignore_index=True) if self._parts else pd.DataFrame()
        # Автовыбор nullable-типов делаем по всему результату, не по первой порции
        for column in combined.columns:
            if column not in self._dtype:
                combined[column] = combined[column].convert_dtypes()
        self._result = self._convert(combined, 0)
        self._parts.clear()
        self._status = "completed"

    def pause(self) -> None:
        if self._status not in {"ready", "running", "paused"}:
            raise ImportStateError(f"Нельзя поставить на паузу: {self._status}")
        self._status = "paused"

    def resume(self) -> None:
        if self._status != "paused":
            raise ImportStateError(f"Продолжить можно только paused-сессию, сейчас {self._status}")
        self._status = "running"

    def run(self) -> pd.DataFrame:
        """Дочитывает до конца и возвращает итог (паузу снимает только resume())."""
        if self._status == "paused":
            raise ImportStateError("Сначала вызовите resume()")
        for _ in self:
            pass
        return self.result()

    def result(self) -> pd.DataFrame:
        if self._status != "completed":
            raise ImportStateError(f"Импорт не завершён: {self._status}")
        return self._result.copy(deep=True)

    def close(self) -> None:
        """Окончательно закрывает незавершённый импорт - это не пауза"""
        try:
            self._close_reader()
        finally:
            self._parts.clear()
            if self._status not in {"completed", "failed"}:
                self._status = "closed"
