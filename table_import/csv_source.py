# Для csv с заголовком на первой строке

from collections.abc import Generator
import codecs
import csv
from io import StringIO
import math
from pathlib import Path
import pandas as pd
from .exceptions import SourceReadError
from .models import ColumnMetadata, SheetMetadata

class CsvSource:
    """ Читает CSV с явно заданными разделителем и кодировкой """

    def __init__(self, path: str | Path, delimiter: str = ",", encoding: str = "utf-8"):
        if not isinstance(delimiter, str) or len(delimiter) != 1:
            raise ValueError("delimiter должен быть одним символом")
        if delimiter in {"\n", "\r"}:
            raise ValueError("delimiter не может быть переводом строки")
        if not isinstance(encoding, str) or not encoding:
            raise ValueError("encoding должен быть непустой строкой")

        self.path = Path(path)
        self.delimiter = delimiter
        self.encoding = encoding

    @staticmethod
    def _validate_size(value: int, name: str) -> None:
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} должен быть положительным целым числом")

    def list_sheets(self) -> list[str]:
        return [self.path.stem]

    def _check_sheet(self, sheet: str | None) -> None:
        if sheet is not None and sheet != self.path.stem:
            raise SourceReadError(
                self.path, "выбор листа CSV",
                f"Лист '{sheet}' не найден; доступен '{self.path.stem}'",
            )

    def scan(self, sample_rows: int = 100, *, sheet: str | None = None) -> SheetMetadata:
        """ Возвращает метаданные первых sample_rows строк данных """
        self._validate_size(sample_rows, "sample_rows")
        self._check_sheet(sheet)

        try:
            sample, row_count, estimated = self._scan_sample(sample_rows)
        except (OSError, UnicodeError, LookupError, ValueError, csv.Error, pd.errors.ParserError) as exc:
            raise SourceReadError(self.path, "сканирование CSV", str(exc)) from exc

        columns = tuple(ColumnMetadata(
                name=str(name),
                dtype=str(sample[name].dtype),
                missing_count=int(sample[name].isna().sum())
            )
            for name in sample.columns
        )

        return SheetMetadata(
            name=self.path.stem,
            row_count=row_count,
            column_count=len(sample.columns),
            sampled_rows=len(sample),
            columns=columns,
            row_count_is_estimate=estimated,
        )

    def _scan_sample(self, sample_rows: int):
        """Считаем исходные байты записей, не позицию буфера pandas."""
        # Инкрементальный encoder не добавляет BOM к каждой физической строке.
        encoder = codecs.getincrementalencoder(self.encoding)()
        automatic_bom = encoder.encode("")
        with self.path.open("rb") as binary:
            prefix = binary.read(4)
        # UTF-8-SIG может читать как файлы с BOM, так и без него.
        consumed = len(automatic_bom) if prefix.startswith(automatic_bom) else 0
        # UTF-16/32 противоположного endian имеют BOM той же длины.
        if automatic_bom and any(prefix.startswith(bom) for bom in (
            codecs.BOM_UTF32_LE, codecs.BOM_UTF32_BE,
            codecs.BOM_UTF16_LE, codecs.BOM_UTF16_BE,
        )):
            consumed = len(automatic_bom)
        file_size = self.path.stat().st_size
        pieces = []
        record_start = 0
        header_end = None
        records = 0
        exhausted = False

        with self.path.open("r", encoding=self.encoding, newline="") as file:
            def lines():
                nonlocal consumed
                for line in file:
                    consumed += len(encoder.encode(line))
                    pieces.append(line)
                    yield line

            reader = csv.reader(lines(), delimiter=self.delimiter)
            while records < sample_rows:
                try:
                    row = next(reader)
                except StopIteration:
                    exhausted = True
                    break
                record_text = "".join(pieces[record_start:])
                record_start = len(pieces)
                if not row or not record_text.strip():
                    continue  # pandas по умолчанию пропускает пустые строки.
                if header_end is None:
                    header_end = consumed
                else:
                    records += 1

        # Типы, названия столбцов и правила пропусков остаются правилами pandas.
        sample = pd.read_csv(StringIO("".join(pieces)), sep=self.delimiter, nrows=sample_rows)
        if exhausted or consumed >= file_size:
            return sample, len(sample), False
        data_bytes = consumed - header_end
        if not len(sample) or data_bytes <= 0:
            return sample, None, False
        estimated = max(len(sample), math.ceil((file_size - header_end) * len(sample) / data_bytes))
        return sample, estimated, True

    def iter_chunks(
        self, chunk_size: int = 1000, *, sheet: str | None = None,
    ) -> Generator[pd.DataFrame, None, None]:
        """ 
        Выдаёт порции DataFrame. при досрочном выходе вызовите close() 
        """
        self._validate_size(chunk_size, "chunk_size")
        self._check_sheet(sheet)
        try:
            with pd.read_csv(
                self.path,
                sep=self.delimiter,
                encoding=self.encoding,
                chunksize=chunk_size
            ) as reader:
                for chunk in reader:
                    yield chunk
        except (OSError, UnicodeError, LookupError, ValueError, pd.errors.ParserError) as exc:
            raise SourceReadError(self.path, "чтение CSV порциями", str(exc)) from exc
