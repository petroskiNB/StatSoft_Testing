"""Тесты SQLite"""

from contextlib import closing
from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory
import unittest
import pandas as pd
from pandas.testing import assert_frame_equal
from table_import import SqlSource, SourceReadError


class SqlSourceTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.path = Path(self.temp_dir.name) / "продажи #1.db"
        with closing(sqlite3.connect(self.path)) as connection:
            connection.execute("CREATE TABLE sales (id INTEGER PRIMARY KEY, amount REAL)")
            connection.executemany(
                "INSERT INTO sales VALUES (?, ?)",
                [(i, None if i in (2, 6) else i * 10) for i in range(1, 11)],
            )
            connection.execute("CREATE TABLE empty_table (id INTEGER, name TEXT)")
            connection.execute('CREATE TABLE "Продажи ""2026""" (value TEXT)')
            connection.execute('INSERT INTO "Продажи ""2026""" VALUES (?)', ("Москва",))
            connection.commit()
        self.source = SqlSource(self.path, table="sales")

    def test_selected_table_is_only_sheet(self):
        self.assertEqual(self.source.list_sheets(), ["sales"])

    def test_scan_is_limited(self):
        metadata = self.source.scan(sample_rows=3)
        self.assertEqual(metadata.name, "sales")
        self.assertEqual(metadata.sampled_rows, 3)
        self.assertEqual(metadata.column_count, 2)
        self.assertIsNone(metadata.row_count)
        self.assertEqual(metadata.columns[0].dtype, "int64")
        self.assertEqual(metadata.columns[1].missing_count, 1)

    def test_chunks_and_final_result(self):
        expected = pd.DataFrame({
            "id": range(1, 11),
            "amount": [10.0, None, 30.0, 40.0, 50.0, None, 70.0, 80.0, 90.0, 100.0],
        })
        for size in (1, 4, 5, 20):
            with self.subTest(size=size):
                chunks = list(self.source.iter_chunks(chunk_size=size))
                if size == 4:
                    self.assertEqual([len(chunk) for chunk in chunks], [4, 4, 2])
                result = pd.concat(chunks, ignore_index=True).sort_values("id").reset_index(drop=True)
                # Проверяем значения с общей nullable-схемой. Порция из одного
                # NULL получает object; унификация типов будет в ImportSession
                assert_frame_equal(result.convert_dtypes(), expected.convert_dtypes())

    def test_pause_resume_and_close(self):
        with closing(self.source.iter_chunks(chunk_size=4)) as chunks:
            first = next(chunks)
            remaining = list(chunks)
            result = pd.concat([first, *remaining], ignore_index=True)
            self.assertEqual(sorted(result["id"].tolist()), list(range(1, 11)))
        with closing(self.source.iter_chunks(chunk_size=4)) as chunks:
            next(chunks)
        # Незакрытый SELECT мог бы удерживать блокировку записи
        with closing(sqlite3.connect(self.path, timeout=0)) as connection:
            connection.execute("UPDATE sales SET amount=99 WHERE id=1")
            connection.commit()
        self.path.unlink()

    def test_empty_table_preserves_columns(self):
        source = SqlSource(self.path, "empty_table")
        metadata = source.scan()
        self.assertEqual(metadata.sampled_rows, 0)
        self.assertEqual(metadata.column_count, 2)
        chunks = list(source.iter_chunks())
        self.assertEqual(len(chunks), 1)
        self.assertTrue(chunks[0].empty)
        self.assertEqual(chunks[0].columns.tolist(), ["id", "name"])

    def test_quoted_table_name_and_injection_like_name(self):
        source = SqlSource(self.path, 'Продажи "2026"')
        result = pd.concat(list(source.iter_chunks()), ignore_index=True)
        self.assertEqual(result["value"].tolist(), ["Москва"])
        with self.assertRaises(SourceReadError):
            SqlSource(self.path, 'sales"; DROP TABLE sales; --').scan()
        self.assertEqual(self.source.list_sheets(), ["sales"])

    def test_missing_database_is_not_created(self):
        path = self.path.parent / "missing.db"
        with self.assertRaises(SourceReadError) as caught:
            SqlSource(path, "sales").scan()
        self.assertFalse(path.exists())
        self.assertIsInstance(caught.exception.__cause__, sqlite3.Error)

    def test_missing_table_and_corrupt_file(self):
        with self.assertRaises(SourceReadError) as caught:
            SqlSource(self.path, "missing").scan()
        self.assertIn("missing", str(caught.exception))
        bad_path = self.path.parent / "bad.db"
        bad_path.write_text("not a SQLite database", encoding="utf-8")
        with self.assertRaises(SourceReadError):
            SqlSource(bad_path, "sales").list_sheets()

    def test_read_only_connection(self):
        # Проверяем непосредственно гарантию режима открытия, включая причину ошибки
        with self.assertRaises(SourceReadError) as caught:
            with self.source._open_connection("проверка режима чтения") as connection:
                connection.execute("DELETE FROM sales")
        # Запись теперь отклоняется authorizer до выполнения SQL.
        self.assertIsInstance(caught.exception.__cause__, sqlite3.DatabaseError)
        result = pd.concat(list(self.source.iter_chunks()), ignore_index=True)
        self.assertEqual(len(result), 10)

    def test_invalid_arguments(self):
        for size in (0, -1, True, 1.5):
            with self.subTest(size=size):
                with self.assertRaises(ValueError):
                    self.source.scan(sample_rows=size)
                with self.assertRaises(ValueError):
                    list(self.source.iter_chunks(chunk_size=size))
        for table in ("", " ", None, "bad\x00name"):
            with self.subTest(table=table):
                with self.assertRaises(ValueError):
                    SqlSource(self.path, table)


if __name__ == "__main__":
    unittest.main()
