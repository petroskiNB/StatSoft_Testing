""" Структуры результата предварительного сканирования """

from dataclasses import dataclass

@dataclass(frozen=True)
class ColumnMetadata:
    """ Тип и количество пропусков относятся только к выборке """

    name: str
    dtype: str
    missing_count: int


@dataclass(frozen=True)
class SheetMetadata:
    """ Описание одной таблицы (логического листа) """

    name: str
    row_count: int | None
    column_count: int
    sampled_rows: int
    columns: tuple[ColumnMetadata, ...]
    row_count_is_estimate: bool = False


@dataclass(frozen=True)
class ImportProgress:
    """Снимок прогресса: считаются только успешно принятые данные"""

    status: str
    rows_imported: int
    chunks_imported: int
    total_rows: int | None
    fraction: float | None
    total_rows_is_estimate: bool = False
    total_rows_exceeded: bool = False
