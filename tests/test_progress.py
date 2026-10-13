"""
ДОПОЛНИТЕЛЬНАЯ ФУНКЦИЯ
Прогресс по точным, оценочным и неизвестным размерам источника
"""

from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import pandas as pd

from table_import import (
    CsvSource, ParquetSource, ImportSession, ImportConfigurationError,
    ImportStateError, TypeConversionError,
)


class ProgressTests(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        path = self.directory / "sales.csv"
        path.write_text("id;amount\n1;10\n2;20\n3;30\n4;40\n", encoding="utf-8")
        self.source = CsvSource(path, delimiter=";")
        self.metadata = self.source.scan(sample_rows=1)

    def test_csv_estimated_progress_pause_and_completion(self):
        self.assertTrue(self.metadata.row_count_is_estimate)
        self.assertEqual(self.metadata.row_count, 4)
        with ImportSession(self.source, "sales", 2, metadata=self.metadata) as session:
            snapshot = session.progress
            self.assertEqual(snapshot.fraction, 0.0)
            next(session)
            self.assertEqual(session.progress.fraction, 0.5)
            self.assertTrue(session.progress.total_rows_is_estimate)
            session.pause()
            self.assertEqual(session.progress.fraction, 0.5)
            session.resume()
            next(session)
            self.assertEqual(session.progress.fraction, 1.0)
            self.assertEqual(session.progress.status, "running")
            with self.assertRaises(ImportStateError):
                session.result()
            self.assertEqual(len(session.run()), 4)
            self.assertFalse(session.progress.total_rows_is_estimate)
            self.assertEqual(session.progress.status, "completed")
            self.assertEqual(snapshot.rows_imported, 0)

    def test_parquet_exact_progress(self):
        path = self.directory / "data.parquet"
        pd.DataFrame({"id": [1, 2, 3, 4]}).to_parquet(path, index=False)
        source = ParquetSource(path)
        metadata = source.scan(sample_rows=1)
        with ImportSession(source, "data", 1, metadata=metadata) as session:
            next(session)
            self.assertEqual(session.progress.total_rows, 4)
            self.assertEqual(session.progress.fraction, 0.25)
            self.assertFalse(session.progress.total_rows_is_estimate)
            session.run()

    def test_underestimate_or_stale_exact_count_does_not_stop_reading(self):
        for approximate in (True, False):
            with self.subTest(approximate=approximate):
                metadata = replace(self.metadata, row_count=1, row_count_is_estimate=approximate)
                with ImportSession(self.source, "sales", 2, metadata=metadata) as session:
                    next(session)
                    self.assertTrue(session.progress.total_rows_exceeded)
                    self.assertIsNone(session.progress.fraction)
                    self.assertEqual(session.progress.total_rows, 1)
                    self.assertEqual(len(session.run()), 4)
                    self.assertEqual(session.progress.total_rows, 4)
                    self.assertEqual(session.progress.fraction, 1.0)
                    self.assertFalse(session.progress.total_rows_exceeded)

    def test_overestimate_unknown_and_no_metadata(self):
        for metadata in (None, replace(self.metadata, row_count=None, row_count_is_estimate=False),
                         replace(self.metadata, row_count=100)):
            with self.subTest(metadata=metadata):
                with ImportSession(self.source, "sales", 2, metadata=metadata) as session:
                    next(session)
                    expected = 0.02 if metadata is not None and metadata.row_count == 100 else None
                    self.assertEqual(session.progress.fraction, expected)
                    session.run()
                    self.assertEqual(session.progress.total_rows, 4)
                    self.assertEqual(session.progress.fraction, 1.0)

    def test_zero_total_empty_and_stale(self):
        metadata = replace(self.metadata, row_count=0, row_count_is_estimate=False)
        with ImportSession(self.source, "sales", 2, metadata=metadata) as session:
            self.assertIsNone(session.progress.fraction)
            next(session)
            self.assertTrue(session.progress.total_rows_exceeded)
        self.source.path.write_text("id;amount\n", encoding="utf-8")
        with ImportSession(self.source, "sales", metadata=self.source.scan()) as session:
            self.assertIsNone(session.progress.fraction)
            self.assertTrue(session.run().empty)
            self.assertEqual(session.progress.fraction, 1.0)
            self.assertEqual(session.progress.total_rows, 0)

    def test_failed_chunk_not_counted_and_close_retains_progress(self):
        self.source.path.write_text("id;amount\n1;10\n2;bad\n", encoding="utf-8")
        metadata = self.source.scan()
        with ImportSession(self.source, "sales", 1, {"amount": "Float64"}, metadata=metadata) as session:
            next(session)
            with self.assertRaises(TypeConversionError):
                next(session)
            self.assertEqual(session.progress.status, "failed")
            self.assertEqual(session.progress.fraction, 0.5)
            self.assertEqual(session.progress.rows_imported, 1)
        with ImportSession(self.source, "sales", 1, metadata=metadata) as session:
            next(session)
        self.assertEqual(session.progress.status, "closed")
        self.assertEqual(session.progress.fraction, 0.5)

    def test_invalid_metadata(self):
        invalid = ["metadata", replace(self.metadata, name="other"),
                   replace(self.metadata, row_count=-1), replace(self.metadata, row_count=True),
                   replace(self.metadata, row_count=1.5), replace(self.metadata, row_count=None),
                   replace(self.metadata, row_count_is_estimate="yes")]
        for metadata in invalid:
            with self.subTest(metadata=metadata):
                with self.assertRaises(ImportConfigurationError):
                    ImportSession(self.source, "sales", metadata=metadata)


if __name__ == "__main__":
    unittest.main()
