"""Запуск: python sqlite_example.py. База создаётся во временной папке."""

from contextlib import closing
from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory
import pandas as pd
from table_import import SqlSource


def main():
    with TemporaryDirectory() as directory:
        path = Path(directory) / "sales.db"
        # Подготовка настоящей SQLite-базы для демонстрации.
        with closing(sqlite3.connect(path)) as connection:
            connection.execute("CREATE TABLE sales (id INTEGER PRIMARY KEY, amount REAL)")
            connection.executemany(
                "INSERT INTO sales VALUES (?, ?)",
                [(1, 10), (2, None), (3, 30), (4, 40), (5, 50)],
            )
            connection.commit()

        source = SqlSource(path, table="sales")
        print("Листы:", source.list_sheets())
        print("Метаданные:", source.scan(sample_rows=3))
        parts = []
        with closing(source.iter_chunks(chunk_size=2)) as chunks:
            for number, chunk in enumerate(chunks, start=1):
                print(f"Порция {number}:")
                print(chunk)
                parts.append(chunk)
        print("Итог:")
        print(pd.concat(parts, ignore_index=True))


if __name__ == "__main__":
    main()
