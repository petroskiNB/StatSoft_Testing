"""Выбор столбцов на настоящих файлах и в SQLite."""

from contextlib import closing
from pathlib import Path
import sqlite3
import tempfile
import unittest

import pandas as pd
from pandas.testing import assert_frame_equal

from table_import import (
    CsvSource, XlsxSource, SqlSource, ParquetSource, ImportSession,
    ImportConfigurationError, SourceReadError,
)


class SelectionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)

    def sources(self, empty=False):
        frame = pd.DataFrame({"id": [1, 2, 3], "amount": [10, 20, 30], "unused": ["bad", "bad", "bad"]})
        if empty:
            frame = frame.iloc[:0]
        
        csv = self.directory / "sales.csv"
        xlsx = self.directory / "sales.xlsx"
        parquet = self.directory / "sales.parquet"
        db = self.directory / "sales.db"
        frame.to_csv(csv, index=False)
        frame.to_excel(xlsx, sheet_name="sales", index=False)
        frame.to_parquet(parquet, index=False)
        with closing(sqlite3.connect(db)) as connection:
            frame.to_sql("sales", connection, index=False, if_exists="replace")
        return [
            CsvSource(csv), XlsxSource(xlsx), ParquetSource(parquet),
            SqlSource(db, "sales"),
            SqlSource(db, query='WITH data AS (SELECT * FROM sales) SELECT * FROM data WHERE id > ? ORDER BY id; -- comment', params=(0,))
        ]

    def test_selection_order_pause_conversion_and_progress(self):
        expected = pd.DataFrame({
            "amount": pd.Series([10, 20, 30], dtype="Float64"),
            "id": pd.Series([1, 2, 3], dtype="Int64")})
        for source in self.sources():
            with self.subTest(source=type(source).__name__):
                sheet = source.list_sheets()[0]
                metadata = source.scan(sheet=sheet, sample_rows=2)
                self.assertEqual(metadata.column_count, 3)
                selected = ["amount", "id"]
                with ImportSession(
                    source, sheet, chunk_size=2, columns=selected,
                    dtype={"amount": "Float64"}, metadata=metadata,
                ) as session:
                    selected.reverse()  # Сессия сохраняет собственную копию настройки.
                    first = next(session)
                    self.assertEqual(first.columns.tolist(), ["amount", "id"])
                    self.assertEqual(session.progress.rows_imported, 2)
                    session.pause()
                    self.assertEqual(list(session), [])
                    session.resume()
                    assert_frame_equal(session.run(), expected)
                    self.assertEqual(session.progress.total_rows, 3)

    def test_direct_chunks_and_default_all_columns(self):
        for source in self.sources():
            sheet = source.list_sheets()[0]
            with closing(source.iter_chunks(sheet=sheet, chunk_size=1, columns=("unused", "id"))) as chunks:
                parts = list(chunks)
            self.assertEqual([len(part) for part in parts], [1, 1, 1])
            self.assertTrue(all(part.columns.tolist() == ["unused", "id"] for part in parts))
            self.assertEqual(pd.concat(parts)["id"].tolist(), [1, 2, 3])
            with ImportSession(source, sheet) as session:
                self.assertEqual(session.run().columns.tolist(), ["id", "amount", "unused"])

    def test_empty_results_keep_selected_schema(self):
        for source in self.sources(empty=True):
            with ImportSession(
                source, source.list_sheets()[0], columns=["amount", "id"],
                dtype={"amount": "Float64", "id": "Int64"},
            ) as session:
                result = session.run()
                self.assertEqual(result.shape, (0, 2))
                self.assertEqual(result.dtypes.astype(str).to_dict(), {"amount": "Float64", "id": "Int64"})

    def test_missing_names_fail_even_for_empty_results(self):
        for empty in (False, True):
            for source in self.sources(empty=empty):
                with ImportSession(source, source.list_sheets()[0], columns=["absent"]) as session:
                    with self.assertRaises(SourceReadError) as caught:
                        session.run()
                    self.assertIn("absent", str(caught.exception))
                    self.assertEqual(session.progress.status, "failed")
                    self.assertEqual(session.progress.rows_imported, 0)

    def test_invalid_selection_and_excluded_dtype(self):
        for source in self.sources():
            sheet = source.list_sheets()[0]
            for columns in ([], "id", ["id", "id"], [1], [""], {"id"}, {"id": 1}):
                with self.assertRaises(ImportConfigurationError):
                    ImportSession(source, sheet, columns=columns)
                with closing(source.iter_chunks(sheet=sheet, columns=columns)) as chunks:
                    with self.assertRaises(ImportConfigurationError):
                        next(chunks)
            with self.assertRaises(ImportConfigurationError):
                ImportSession(source, sheet, columns=["id"], dtype={"unused": "Int64"})


if __name__ == "__main__":
    unittest.main()
