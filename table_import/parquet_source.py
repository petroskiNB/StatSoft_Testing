"""Порционное чтение одного локального Parquet-файла"""

from collections.abc import Generator
from contextlib import closing, contextmanager
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .exceptions import SourceReadError
from .models import ColumnMetadata, SheetMetadata


def _pandas_type(arrow_type):
    """Сохраняем целые с NULL без промежуточного преобразования во float"""
    return {
        pa.int8(): pd.Int8Dtype(), pa.int16(): pd.Int16Dtype(),
        pa.int32(): pd.Int32Dtype(), pa.int64(): pd.Int64Dtype(),
        pa.uint8(): pd.UInt8Dtype(), pa.uint16(): pd.UInt16Dtype(),
        pa.uint32(): pd.UInt32Dtype(), pa.uint64(): pd.UInt64Dtype(),
        pa.float32(): pd.Float32Dtype(), pa.float64(): pd.Float64Dtype(),
        pa.bool_(): pd.BooleanDtype(),
        pa.string(): pd.StringDtype(),
        pa.large_string(): pd.StringDtype(),
    }.get(arrow_type)


class ParquetSource:
    """Один файл - один лист с именем файла без расширения."""

    def __init__(self, path: str | Path):
        self.path = Path(path)

    @staticmethod
    def _validate_size(value: int, name: str) -> None:
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} должен быть положительным целым числом")

    def _check_sheet(self, sheet: str | None) -> None:
        if sheet is not None and sheet != self.path.stem:
            raise SourceReadError(
                self.path, "выбор листа Parquet",
                f"Лист '{sheet}' не найден; доступен '{self.path.stem}'",
            )

    @contextmanager
    def _open_file(self):
        try:
            with self.path.open("rb") as file:
                with closing(pq.ParquetFile(file, pre_buffer=False)) as parquet:
                    schema = parquet.schema_arrow
                    if len(set(schema.names)) != len(schema.names):
                        raise ValueError("Названия столбцов должны быть уникальными")
                    for field in schema:
                        value_type = field.type
                        if pa.types.is_dictionary(value_type):
                            value_type = value_type.value_type
                        if pa.types.is_nested(value_type):
                            raise ValueError(
                                f"Столбец '{field.name}': вложенный тип {field.type} "
                                "пока не поддерживается; требуется плоская таблица"
                            )
                    yield parquet
        except (OSError, pa.ArrowException, ValueError, TypeError) as exc:
            raise SourceReadError(self.path, "чтение Parquet", str(exc)) from exc

    def list_sheets(self) -> list[str]:
        with self._open_file():
            return [self.path.stem]

    @staticmethod
    def _to_frame(batch) -> pd.DataFrame:
        # Физически записанный pandas-индекс читается как обычный столбец
        return batch.to_pandas(ignore_metadata=True, types_mapper=_pandas_type)

    def scan(
        self, sample_rows: int = 100, *, sheet: str | None = None,
    ) -> SheetMetadata:
        self._validate_size(sample_rows, "sample_rows")
        self._check_sheet(sheet)
        with self._open_file() as parquet:
            with closing(parquet.iter_batches(batch_size=sample_rows, use_threads=False)) as batches:
                batch = next(batches, None)
                if batch is None:
                    batch = pa.Table.from_batches([], schema=parquet.schema_arrow)
                sample = self._to_frame(batch)
            return SheetMetadata(
                name=self.path.stem,
                row_count=parquet.metadata.num_rows,
                column_count=len(sample.columns),
                sampled_rows=len(sample),
                columns=tuple(
                    ColumnMetadata(
                        name=name, dtype=str(sample[name].dtype),
                        missing_count=int(sample[name].isna().sum()),
                    )
                    for name in sample.columns
                ),
            )

    def iter_chunks(
        self, chunk_size: int = 1000, *, sheet: str | None = None,
    ) -> Generator[pd.DataFrame, None, None]:
        self._validate_size(chunk_size, "chunk_size")
        self._check_sheet(sheet)
        with self._open_file() as parquet:
            emitted = False
            with closing(parquet.iter_batches(batch_size=chunk_size, use_threads=False)) as batches:
                for batch in batches:
                    emitted = True
                    yield self._to_frame(batch)
            if not emitted:
                yield self._to_frame(pa.Table.from_batches([], schema=parquet.schema_arrow))
