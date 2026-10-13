""" Чтение одной таблицы из существующей файловой базы SQLite """

from collections.abc import Generator, Mapping, Sequence
from contextlib import closing, contextmanager
from pathlib import Path
import sqlite3
import pandas as pd
from .exceptions import SourceReadError
from .models import ColumnMetadata, SheetMetadata


class SqlSource:
    """Один лист - результат SELECT или выбранная таблица SQLite"""

    def __init__(
        self, path: str | Path, table: str | None = None, *,
        query: str | None = None, params=None, sheet_name: str | None = None,
    ):
        if (table is None) == (query is None):
            raise ValueError("Укажите либо table, либо query")
        if table is not None and (not isinstance(table, str) or not table.strip() or "\x00" in table):
            raise ValueError("table должен быть непустым именем таблицы без нулевого символа")
        if query is not None and (not isinstance(query, str) or not query.strip()):
            raise ValueError("query должен быть непустым SELECT-запросом")
        if table is not None and params is not None:
            raise ValueError("params применим только к query")
        if params is None:
            self.params = ()
        elif isinstance(params, Mapping):
            self.params = dict(params)
        elif isinstance(params, Sequence) and not isinstance(params, (str, bytes, bytearray)):
            self.params = tuple(params)
        else:
            raise ValueError("params должен быть последовательностью или словарём")
        if sheet_name is not None and (not isinstance(sheet_name, str) or not sheet_name.strip()):
            raise ValueError("sheet_name должен быть непустой строкой")
        self.path = Path(path)
        self.table = table
        self.query = query
        self.sheet_name = sheet_name if sheet_name is not None else (table if table is not None else "query")

    @staticmethod
    def _validate_size(value: int, name: str) -> None:
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} должен быть положительным целым числом")

    @contextmanager
    def _open_connection(self, operation: str):
        """Открывает базу только для чтения и всегда закрывает соединение"""
        try:
            # mode=ro запрещает изменения и создание отсутствующей базы
            uri = self.path.resolve().as_uri() + "?mode=ro"
            with closing(sqlite3.connect(uri, uri=True)) as connection:
                # SQLite проверяет операции при компиляции запроса, а не по
                # первому слову SQL: разрешаем SELECT/CTE, запрещаем DML/DDL/ATTACH
                allowed = {sqlite3.SQLITE_SELECT, sqlite3.SQLITE_READ,
                           sqlite3.SQLITE_FUNCTION, sqlite3.SQLITE_RECURSIVE}
                connection.set_authorizer(
                    lambda action, arg1, arg2, database, trigger:
                    sqlite3.SQLITE_OK if action in allowed else sqlite3.SQLITE_DENY
                )
                yield connection
        except (sqlite3.Error, OSError) as exc:
            raise SourceReadError(
                self.path, f"{operation}, лист '{self.sheet_name}'", str(exc),
            ) from exc

    def _check_table(self, connection: sqlite3.Connection) -> None:
        # Имя здесь является значением условия, поэтому передаётся через "?"
        with closing(connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name = ?",
            (self.table,)
        )) as cursor:
            if cursor.fetchone() is None:
                raise SourceReadError(
                    self.path, "выбор таблицы SQLite",
                    f"Таблица '{self.table}' не найдена"
                )

    def list_sheets(self) -> list[str]:
        """Для query возвращает логическое имя; SQL проверяется при чтении"""
        with self._open_connection("получение листа SQLite") as connection:
            if self.table is not None:
                self._check_table(connection)
            else:
                # Проверка файла базы без запуска потенциально дорогого запроса.
                with closing(connection.execute("SELECT name FROM sqlite_master LIMIT 0")):
                    pass
            return [self.sheet_name]

    def _check_sheet(self, sheet: str | None) -> None:
        if sheet is not None and sheet != self.sheet_name:
            raise SourceReadError(
                self.path, "выбор листа SQLite",
                f"Лист '{sheet}' не найден; доступен '{self.sheet_name}'",
            )

    def scan(self, sample_rows: int = 100, *, sheet: str | None = None) -> SheetMetadata:
        """Получает не более sample_rows строк результата и закрывает курсор."""
        self._validate_size(sample_rows, "sample_rows")
        self._check_sheet(sheet)
        with closing(self._read_chunks(sample_rows, limit=sample_rows)) as chunks:
            sample = next(chunks)

        columns = tuple(
            ColumnMetadata(
                name=name,
                dtype=str(sample[name].dtype),
                missing_count=int(sample[name].isna().sum()),
            )
            for name in sample.columns
        )
        return SheetMetadata(
            name=self.sheet_name,
            row_count=None,
            column_count=len(sample.columns),
            sampled_rows=len(sample),
            columns=columns,
        )

    def iter_chunks(
        self, chunk_size: int = 1000, *, sheet: str | None = None,
    ) -> Generator[pd.DataFrame, None, None]:
        """Читает следующий фрагмент одного запроса"""
        self._validate_size(chunk_size, "chunk_size")
        self._check_sheet(sheet)
        yield from self._read_chunks(chunk_size)

    def _read_chunks(self, chunk_size: int, limit: int | None = None) -> Generator[pd.DataFrame, None, None]:
        with self._open_connection("чтение SQLite") as connection:
            if self.table is not None:
                self._check_table(connection)
            # Идентификатор таблицы нельзя передать параметром ?
            # Заключаем имя в двойные кавычки, удваивая кавычки внутри него
            if self.table is not None:
                quoted_table = '"' + self.table.replace('"', '""') + '"'
                query = f"SELECT * FROM {quoted_table}"
                parameters = ()
                if limit is not None:
                    query += " LIMIT ?"
                    parameters = (limit,)
            else:
                query, parameters = self.query, self.params

            with closing(connection.cursor()) as cursor:
                cursor.execute(query, parameters)
                if cursor.description is None:
                    raise SourceReadError(self.path, "чтение SQLite", "Запрос должен возвращать таблицу")
                names = [column[0] for column in cursor.description]
                if len(set(names)) != len(names):
                    raise SourceReadError(
                        self.path, f"чтение листа '{self.sheet_name}'",
                        "Названия столбцов результата повторяются; задайте уникальные псевдонимы AS"
                    )
                emitted = False
                remaining = limit
                while True:
                    rows = cursor.fetchmany(chunk_size if remaining is None else min(chunk_size, remaining))
                    if not rows:
                        if not emitted:
                            yield pd.DataFrame(columns=names)
                        break
                    yield pd.DataFrame.from_records(rows, columns=names)
                    emitted = True
                    if remaining is not None:
                        remaining -= len(rows)
                        if remaining == 0:
                            break
