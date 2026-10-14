"""Общий контракт источников: наследование от Protocol не обязательно"""

from collections.abc import Generator, Sequence
from typing import Protocol
import pandas as pd
from .models import SheetMetadata


class TableSource(Protocol):
    def list_sheets(self) -> list[str]: ...

    def scan(self, *, sheet: str, sample_rows: int = 100) -> SheetMetadata: ...

    def iter_chunks(
        self, *, sheet: str, chunk_size: int = 1000,
        columns: Sequence[str] | None = None,
    ) -> Generator[pd.DataFrame, None, None]: ...
