"""Полный импорт настоящих CSV, XLSX и SQLite через один интерфейс."""

from contextlib import closing
from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory
import unittest

import pandas as pd
from openpyxl import Workbook
from pandas.testing import assert_frame_equal

from table_import import (
    CsvSource, XlsxSource, SqlSource, ImportSession, ImportStateError,
    ImportConfigurationError, TypeConversionError, SourceReadError,
)


class ImportSessionTests(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.headers = ["id", "amount", "active", "date", "name"]
        self.rows = [
            [1, 10.0, "true", "2026-01-01", "Москва"],
            [2, None, "false", "2026-01-02", "Казань"],
            [3, 30.5, "1", None, "Омск"],
            [4, 40.0, "0", "2026-01-04", "Тула"],
            [5, 50.0, "true", "2026-01-05", "Пермь"],
        ]
        self.dtype = {
            "id": "Int64", "amount": "Float64", "active": "boolean",
            "date": "datetime64[ns]", "name": "string",
        }

    def make_sources(self, rows=None, prefix="sales"):
        rows = self.rows if rows is None else rows
        csv_path = self.directory / f"{prefix}.csv"
        xlsx_path = self.directory / f"{prefix}.xlsx"
        sql_path = self.directory / f"{prefix}.db"
        pd.DataFrame(rows, columns=self.headers).to_csv(csv_path, index=False)
        book = Workbook()
        sheet = book.active
        sheet.title = "sales"
        sheet.append(self.headers)
        for row in rows:
            sheet.append(row)
        book.save(xlsx_path)
        book.close()
        with closing(sqlite3.connect(sql_path)) as connection:
            connection.execute(
                "CREATE TABLE sales (id INTEGER PRIMARY KEY, amount REAL, active TEXT, date TEXT, name TEXT)"
            )
            connection.executemany("INSERT INTO sales VALUES (?, ?, ?, ?, ?)", rows)
            connection.commit()
        return [CsvSource(csv_path), XlsxSource(xlsx_path), SqlSource(sql_path, "sales")]

    def expected(self):
        return pd.DataFrame({
            "id": pd.Series([1, 2, 3, 4, 5], dtype="Int64"),
            "amount": pd.Series([10, None, 30.5, 40, 50], dtype="Float64"),
            "active": pd.Series([True, False, True, False, True], dtype="boolean"),
            "date": pd.Series(pd.to_datetime([
                "2026-01-01", "2026-01-02", None, "2026-01-04", "2026-01-05",
            ]), dtype="datetime64[ns]"),
            "name": pd.Series(["Москва", "Казань", "Омск", "Тула", "Пермь"], dtype="string"),
        })

    def test_all_sources_same_contract_and_final_types(self):
        for source in self.make_sources():
            sheet = source.list_sheets()[0]
            self.assertEqual(source.scan(sheet=sheet, sample_rows=2).sampled_rows, 2)
            for size in (1, 2, 5, 20):
                with self.subTest(source=type(source).__name__, size=size):
                    with ImportSession(source, sheet, chunk_size=size, dtype=self.dtype) as session:
                        result = session.run()
                        assert_frame_equal(result, self.expected())
                        self.assertEqual(session.progress.status, "completed")
                        self.assertEqual(session.progress.rows_imported, 5)
                        self.assertEqual(session.progress.chunks_imported, (5 + size - 1) // size)
                        self.assertEqual(session.progress.total_rows, 5)
                        self.assertEqual(session.progress.fraction, 1.0)
                        assert_frame_equal(session.run(), result)

    def test_pause_resume_progress_and_result_protection(self):
        for source in self.make_sources():
            with self.subTest(source=type(source).__name__):
                with ImportSession(source, source.list_sheets()[0], 2, self.dtype) as session:
                    before = session.progress
                    self.assertEqual(before.status, "ready")
                    self.assertIsNone(before.fraction)
                    with self.assertRaises(ImportStateError):
                        session.result()
                    first = next(session)
                    self.assertEqual(session.progress.rows_imported, 2)
                    self.assertEqual(session.progress.chunks_imported, 1)
                    first.loc[0, "id"] = 999  # Выданная порция не портит внутренний результат.
                    session.pause()
                    self.assertEqual(list(session), [])
                    self.assertEqual(session.progress.status, "paused")
                    self.assertEqual(session.progress.rows_imported, 2)
                    with self.assertRaises(ImportStateError):
                        session.run()
                    session.resume()
                    result = session.run()
                    assert_frame_equal(result, self.expected())
                    result.loc[0, "id"] = 888
                    assert_frame_equal(session.result(), self.expected())
                    self.assertEqual(before.rows_imported, 0)  # Снимок не меняется задним числом.

    def test_error_in_later_chunk_is_terminal_and_detailed(self):
        rows = [list(row) for row in self.rows]
        rows[2][1] = "ошибка"
        for source in self.make_sources(rows):
            with self.subTest(source=type(source).__name__):
                with ImportSession(source, source.list_sheets()[0], 2, self.dtype) as session:
                    next(session)
                    with self.assertRaises(TypeConversionError) as caught:
                        next(session)
                    error = caught.exception
                    self.assertEqual(error.row, 3)
                    self.assertEqual(error.column, "amount")
                    self.assertEqual(error.value, "ошибка")
                    self.assertEqual(error.target_type, "Float64")
                    self.assertIsNotNone(error.__cause__)
                    self.assertEqual(session.progress.status, "failed")
                    self.assertEqual(session.progress.rows_imported, 2)
                    self.assertEqual(session.progress.chunks_imported, 1)
                    with self.assertRaises(ImportStateError):
                        session.result()
                    with self.assertRaises(ImportStateError):
                        session.resume()
                source.path.unlink()  # Ресурс освобождён после ошибки.

    def test_empty_sources_preserve_schema(self):
        for source in self.make_sources([]):
            with self.subTest(source=type(source).__name__):
                with ImportSession(source, source.list_sheets()[0], dtype=self.dtype) as session:
                    result = session.run()
                    self.assertTrue(result.empty)
                    self.assertEqual(result.columns.tolist(), self.headers)
                    self.assertEqual(result.dtypes.astype(str).to_dict(), self.dtype)
                    self.assertEqual(session.progress.rows_imported, 0)
                    self.assertEqual(session.progress.chunks_imported, 0)
                    self.assertEqual(session.progress.fraction, 1.0)

    def test_missing_column_and_sheet(self):
        for source in self.make_sources():
            with self.subTest(source=type(source).__name__):
                with ImportSession(source, source.list_sheets()[0], dtype={"absent": "Int64"}) as session:
                    with self.assertRaises(TypeConversionError) as caught:
                        session.run()
                    self.assertEqual(caught.exception.column, "absent")
                with ImportSession(source, "absent") as session:
                    with self.assertRaises(SourceReadError):
                        session.run()
                    self.assertEqual(session.progress.status, "failed")

    def test_close_releases_reader_and_prevents_resume(self):
        for source in self.make_sources():
            with self.subTest(source=type(source).__name__):
                with ImportSession(source, source.list_sheets()[0], 2) as session:
                    next(session)
                    session.pause()
                self.assertEqual(session.progress.status, "closed")
                with self.assertRaises(ImportStateError):
                    session.resume()
                with self.assertRaises(ImportStateError):
                    next(session)
                if isinstance(source, SqlSource):
                    with closing(sqlite3.connect(source.path, timeout=0)) as connection:
                        connection.execute("UPDATE sales SET amount=1")
                        connection.commit()
                source.path.unlink()

    def test_strict_conversion_does_not_truncate_or_guess(self):
        for number, (column, value, target) in enumerate([
            ("amount", 1.5, "int64"),
            ("amount", None, "int64"),
            ("active", "maybe", "boolean"),
            ("date", "not-a-date", "datetime64[ns]"),
            ("date", "2026-01-01T00:00:00Z", "datetime64[ns]"),
            ("amount", "9223372036854775808", "Int64"),
        ]):
            rows = [list(self.rows[0])]
            rows[0][self.headers.index(column)] = value
            for source in self.make_sources(rows, prefix=f"invalid{number}"):
                with self.subTest(column=column, value=value, source=type(source).__name__):
                    with ImportSession(source, source.list_sheets()[0], dtype={column: target}) as session:
                        with self.assertRaises(TypeConversionError):
                            session.run()

    def test_unspecified_numeric_types_normalized_after_concat(self):
        for source in self.make_sources():
            results = []
            for size in (1, 4):
                with ImportSession(source, source.list_sheets()[0], size) as session:
                    results.append(session.run())
            assert_frame_equal(results[0][["id", "amount"]], results[1][["id", "amount"]])

    def test_pause_before_start_and_invalid_settings(self):
        source = self.make_sources()[0]
        with ImportSession(source, "sales") as session:
            session.pause()
            self.assertEqual(list(session), [])
            session.resume()
            self.assertEqual(len(session.run()), 5)
        for options in ({"chunk_size": 0}, {"chunk_size": True}, {"dtype": []}, {"dtype": {"id": "unknown"}}):
            with self.subTest(options=options):
                with self.assertRaises(ImportConfigurationError):
                    ImportSession(source, "sales", **options)

    def test_custom_source_needs_no_session_changes(self):
        class MemorySource:
            def list_sheets(self):
                return ["memory"]

            def iter_chunks(self, *, sheet, chunk_size):
                self.closed = False
                try:
                    yield pd.DataFrame({"value": ["12", "34"]})
                finally:
                    self.closed = True

        source = MemorySource()
        with ImportSession(source, "memory", dtype={"value": "Int64"}) as session:
            self.assertEqual(session.run()["value"].tolist(), [12, 34])
        self.assertTrue(source.closed)

    def test_read_error_and_schema_change_close_custom_source(self):
        class BrokenSource:
            def iter_chunks(self, *, sheet, chunk_size):
                try:
                    yield pd.DataFrame({"a": [1]})
                    yield pd.DataFrame({"b": [2]})
                finally:
                    self.closed = True

        source = BrokenSource()
        with ImportSession(source, "sheet") as session:
            with self.assertRaises(SourceReadError):
                session.run()
            self.assertEqual(session.progress.status, "failed")
        self.assertTrue(source.closed)
        with ImportSession(CsvSource(self.directory / "missing.csv"), "missing") as session:
            with self.assertRaises(SourceReadError):
                session.run()
            self.assertEqual(session.progress.rows_imported, 0)


if __name__ == "__main__":
    unittest.main()
