""" Сквозные тесты: настоящий CSV -> публичный API -> результат """

from contextlib import closing
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
import pandas as pd
from pandas.testing import assert_frame_equal
from table_import import CsvSource, SourceReadError


class CsvSourceTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.path = Path(self.temp_dir.name) / "sales.csv"
        self.path.write_text(
            "id;amount\n"
            "1;10\n2;\n3;30\n4;40\n5;50\n"
            "6;\n7;70\n8;80\n9;90\n10;100\n",
            encoding="utf-8"
        )
        self.source = CsvSource(self.path, delimiter=";")

    def test_scan_uses_only_requested_sample(self):
        metadata = self.source.scan(sample_rows=3)
        self.assertEqual(metadata.name, "sales")
        self.assertGreaterEqual(metadata.row_count, metadata.sampled_rows)
        self.assertTrue(metadata.row_count_is_estimate)
        self.assertEqual(metadata.column_count, 2)
        self.assertEqual(metadata.sampled_rows, 3)
        self.assertEqual(metadata.columns[0].name, "id")
        self.assertEqual(metadata.columns[0].dtype, "int64")
        self.assertEqual(metadata.columns[1].missing_count, 1)

    def test_chunks_produce_expected_table(self):
        chunks = list(self.source.iter_chunks(chunk_size=4))
        self.assertEqual([len(chunk) for chunk in chunks], [4, 4, 2])
        result = pd.concat(chunks, ignore_index=True)
        expected = pd.DataFrame({
            "id": range(1, 11),
            "amount": [10.0, None, 30.0, 40.0, 50.0, None, 70.0, 80.0, 90.0, 100.0],
        })
        assert_frame_equal(result, expected)

    def test_same_iterator_can_continue_and_close(self):
        with closing(self.source.iter_chunks(chunk_size=4)) as chunks:
            first = next(chunks)
            self.assertEqual(first["id"].tolist(), [1, 2, 3, 4])
            second = next(chunks)
            self.assertEqual(second["id"].tolist(), [5, 6, 7, 8])
        # Проверяем, что после досрочного закрытия файл можно удалить
        self.path.unlink()

    def test_missing_file_has_library_error_and_original_cause(self):
        source = CsvSource(self.path.parent / "missing.csv")
        for operation in (source.scan, lambda: list(source.iter_chunks())):
            with self.subTest(operation=operation):
                with self.assertRaises(SourceReadError) as caught:
                    operation()
                self.assertEqual(caught.exception.path, source.path)
                self.assertIsInstance(caught.exception.__cause__, FileNotFoundError)

    def test_invalid_sizes(self):
        for size in (0, -1, True, 1.5):
            with self.subTest(size=size):
                with self.assertRaises(ValueError):
                    self.source.scan(sample_rows=size)
                with self.assertRaises(ValueError):
                    list(self.source.iter_chunks(chunk_size=size))

    def test_empty_file_and_header_only_file(self):
        self.path.write_text("", encoding="utf-8")
        with self.assertRaises(SourceReadError):
            self.source.scan()
        self.path.write_text("id;amount\n", encoding="utf-8")
        metadata = self.source.scan()
        self.assertEqual(metadata.sampled_rows, 0)
        self.assertEqual(metadata.column_count, 2)

    def test_quoted_field_can_contain_newline(self):
        self.path.write_text(
            'id;note\n1;"первая\nвторая"\n2;конец\n',
            encoding="utf-8",
            newline="",
        )
        chunks = list(self.source.iter_chunks(chunk_size=1))
        self.assertEqual(len(chunks), 2)
        self.assertEqual(chunks[0].loc[0, "note"], "первая\nвторая")


if __name__ == "__main__":
    unittest.main()
