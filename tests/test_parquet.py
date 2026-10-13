"""Настоящие Parquet-файлы: row groups, схема, порции и ImportSession."""

from contextlib import closing
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import pandas as pd
from pandas.testing import assert_frame_equal
import pyarrow as pa
import pyarrow.parquet as pq

from table_import import ParquetSource, ImportSession, SourceReadError, TypeConversionError


class ParquetSourceTests(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.path = Path(temporary.name) / "sales.parquet"
        self.table = pa.table({
            "id": pa.array(range(1, 11), type=pa.int64()),
            "amount": pa.array([10, None, 30, 40, 50, None, 70, 80, 90, 100], type=pa.float64()),
            "code": pa.array([f"{i:03}" for i in range(1, 11)]),
        })
        pq.write_table(self.table, self.path, row_group_size=3, compression="snappy")
        self.source = ParquetSource(self.path)

    def expected(self):
        return pd.DataFrame({
            "id": pd.Series(range(1, 11), dtype="Int64"),
            "amount": pd.Series([10, None, 30, 40, 50, None, 70, 80, 90, 100], dtype="Float64"),
            "code": pd.Series([f"{i:03}" for i in range(1, 11)], dtype="string"),
        })

    def test_metadata_uses_footer_count_and_bounded_sample(self):
        self.assertEqual(self.source.list_sheets(), ["sales"])
        metadata = self.source.scan(sheet="sales", sample_rows=4)
        self.assertEqual(metadata.row_count, 10)
        self.assertEqual(metadata.sampled_rows, 4)
        self.assertEqual(metadata.column_count, 3)
        self.assertEqual(metadata.columns[0].dtype, "Int64")
        self.assertEqual(metadata.columns[1].missing_count, 1)

    def test_chunks_cross_row_groups_without_loss(self):
        for size in (1, 4, 10, 20):
            with self.subTest(size=size):
                chunks = list(self.source.iter_chunks(sheet="sales", chunk_size=size))
                self.assertTrue(all(len(chunk) <= size for chunk in chunks))
                if size == 4:
                    self.assertEqual([len(chunk) for chunk in chunks], [4, 4, 2])
                assert_frame_equal(pd.concat(chunks, ignore_index=True), self.expected())

    def test_session_pause_types_progress_and_result(self):
        for size in (1, 4):
            with self.subTest(size=size):
                with ImportSession(
                    self.source, "sales", size,
                    dtype={"id": "Int64", "amount": "Float64", "code": "string"},
                ) as session:
                    next(session)
                    session.pause()
                    self.assertEqual(session.progress.rows_imported, size)
                    self.assertEqual(list(session), [])
                    session.resume()
                    assert_frame_equal(session.run(), self.expected())
                    self.assertEqual(session.progress.rows_imported, 10)
                    self.assertEqual(session.progress.status, "completed")

    def test_empty_file_preserves_schema(self):
        pq.write_table(self.table.slice(0, 0), self.path)
        metadata = self.source.scan()
        self.assertEqual(metadata.row_count, 0)
        self.assertEqual(metadata.sampled_rows, 0)
        self.assertEqual(metadata.columns[0].dtype, "Int64")
        chunks = list(self.source.iter_chunks())
        self.assertEqual(len(chunks), 1)
        assert_frame_equal(chunks[0], self.expected().iloc[:0])
        with ImportSession(self.source, "sales", dtype={"amount": "Float64"}) as session:
            self.assertTrue(session.run().empty)

    def test_large_integer_with_null_stays_exact(self):
        value = 2 ** 60 + 1
        pq.write_table(pa.table({"id": pa.array([value, None], type=pa.int64())}), self.path)
        with ImportSession(self.source, "sales", dtype={"id": "Int64"}) as session:
            result = session.run()
        self.assertEqual(result.loc[0, "id"], value)
        self.assertTrue(pd.isna(result.loc[1, "id"]))

    def test_early_close_releases_file(self):
        with closing(self.source.iter_chunks(chunk_size=2)) as chunks:
            next(chunks)
        self.path.unlink()

    def test_missing_corrupt_and_wrong_sheet(self):
        with self.assertRaises(SourceReadError):
            self.source.scan(sheet="absent")
        with self.assertRaises(SourceReadError):
            list(self.source.iter_chunks(sheet="absent"))
        self.path.unlink()
        with self.assertRaises(SourceReadError) as caught:
            self.source.scan()
        self.assertIsInstance(caught.exception.__cause__, FileNotFoundError)
        self.path.write_bytes(b"ordinary text, not parquet")
        with self.assertRaises(SourceReadError):
            list(self.source.iter_chunks())

    def test_nested_columns_rejected_with_context(self):
        for values in ([[1, 2], None], [{"x": 1}, None]):
            pq.write_table(pa.table({"nested": values}), self.path)
            with self.assertRaises(SourceReadError) as caught:
                self.source.scan()
            self.assertIn("nested", str(caught.exception))
            self.assertIn("вложенный", str(caught.exception))

    def test_invalid_sizes(self):
        for size in (0, -1, True, 1.5):
            with self.subTest(size=size):
                with self.assertRaises(ValueError):
                    self.source.scan(sample_rows=size)
                with self.assertRaises(ValueError):
                    list(self.source.iter_chunks(chunk_size=size))

    def test_conversion_error_closes_reader(self):
        pq.write_table(pa.table({"value": ["1", "2", "bad"]}), self.path)
        with ImportSession(self.source, "sales", 2, {"value": "Int64"}) as session:
            next(session)
            with self.assertRaises(TypeConversionError) as caught:
                session.run()
            self.assertEqual(caught.exception.row, 3)
            self.assertEqual(session.progress.status, "failed")
        self.path.unlink()


if __name__ == "__main__":
    unittest.main()
