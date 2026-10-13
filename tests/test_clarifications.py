"""Доп проверка после уточнений"""

from contextlib import closing
from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory
import unittest

from table_import import CsvSource, SqlSource, ImportSession, SourceReadError


class ClarificationTests(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)

    def test_csv_estimate_counts_bytes_and_logical_records(self):
        for encoding in ("utf-8", "utf-8-sig", "cp1251", "utf-16", "utf-32"):
            for ending in ("\n", "\r\n", "\r"):
                with self.subTest(encoding=encoding, ending=ending):
                    path = self.directory / "sample.csv"
                    text = f'id;note{ending}' + f'1;"Москва{ending}Казань"{ending}' * 100
                    path.write_text(text, encoding=encoding, newline="")
                    source = CsvSource(path, delimiter=";", encoding=encoding)
                    metadata = source.scan(sample_rows=3)
                    self.assertEqual(metadata.sampled_rows, 3)
                    self.assertEqual(metadata.row_count, 100)
                    self.assertTrue(metadata.row_count_is_estimate)
                    exact = source.scan(sample_rows=101)
                    self.assertEqual(exact.row_count, 100)
                    self.assertFalse(exact.row_count_is_estimate)

    def test_csv_header_only_exact_boundary_and_blank_lines(self):
        path = self.directory / "sample.csv"
        for text, expected in (("id;value\n", 0), ("id;value\n1;x\n2;y", 2),
                               ("\nid;value\n\n1;x\n  \n2;y\n", 2)):
            path.write_text(text, encoding="utf-8")
            result = CsvSource(path, delimiter=";").scan(sample_rows=2)
            self.assertEqual(result.row_count, expected)
            self.assertFalse(result.row_count_is_estimate)

    def test_csv_uneven_records_is_explicitly_an_estimate(self):
        path = self.directory / "sample.csv"
        path.write_text("id;value\n1;x\n2;y\n3;" + "long" * 200 + "\n", encoding="utf-8")
        result = CsvSource(path, delimiter=";").scan(sample_rows=2)
        self.assertTrue(result.row_count_is_estimate)
        self.assertGreater(result.row_count, 3)

    def make_database(self):
        path = self.directory / "sales.db"
        with closing(sqlite3.connect(path)) as connection:
            connection.executescript(
                "CREATE TABLE customers(id INTEGER, name TEXT);"
                "CREATE TABLE orders(id INTEGER, customer_id INTEGER, amount REAL);"
                "INSERT INTO customers VALUES(1, 'Москва'), (2, 'Казань');"
                "INSERT INTO orders VALUES(1,1,10), (2,2,20), (3,1,30), (4,2,40);"
            )
        return path

    def test_join_params_scan_pause_and_result(self):
        path = self.make_database()
        query = """SELECT o.id, c.name AS customer_name, o.amount
                   FROM orders o JOIN customers c ON c.id=o.customer_id
                   WHERE amount >= ? ORDER BY o.id; -- завершающий комментарий"""
        source = SqlSource(path, query=query, params=(20,), sheet_name="Заказы")
        self.assertEqual(source.list_sheets(), ["Заказы"])
        metadata = source.scan(sheet="Заказы", sample_rows=1)
        self.assertEqual(metadata.sampled_rows, 1)
        self.assertIsNone(metadata.row_count)
        self.assertFalse(metadata.row_count_is_estimate)
        with ImportSession(source, "Заказы", 1, dtype={"amount": "Float64"}) as session:
            next(session)
            session.pause()
            session.resume()
            result = session.run()
        self.assertEqual(result["id"].tolist(), [2, 3, 4])
        self.assertEqual(result["customer_name"].tolist(), ["Казань", "Москва", "Казань"])

    def test_cte_named_params_and_empty_result(self):
        path = self.make_database()
        source = SqlSource(path, query="WITH selected AS (SELECT * FROM orders WHERE amount>:min) SELECT id FROM selected ORDER BY id", params={"min": 20})
        with ImportSession(source, "query") as session:
            self.assertEqual(session.run()["id"].tolist(), [3, 4])
        empty = SqlSource(path, query="SELECT id, amount FROM orders WHERE amount>?", params=(100,))
        self.assertEqual(empty.scan().sampled_rows, 0)
        with ImportSession(empty, "query") as session:
            self.assertEqual(session.run().columns.tolist(), ["id", "amount"])

    def test_sql_errors_duplicates_and_forbidden_operations(self):
        path = self.make_database()
        for query in ("SELECT nope FROM orders", "SELECT * FROM", "SELECT ?", "SELECT 1; SELECT 2",
                      "SELECT o.id, c.id FROM orders o JOIN customers c ON c.id=o.customer_id",
                      "DELETE FROM orders RETURNING id", "CREATE TEMP TABLE x(id)",
                      "ATTACH DATABASE ':memory:' AS other", "PRAGMA user_version=1"):
            with self.subTest(query=query):
                source = SqlSource(path, query=query)
                with self.assertRaises(SourceReadError):
                    source.scan()
                with self.assertRaises(SourceReadError):
                    list(source.iter_chunks())
        with closing(sqlite3.connect(path)) as connection:
            self.assertEqual(connection.execute("SELECT count(*) FROM orders").fetchone()[0], 4)

    def test_sql_options_and_backward_compatibility(self):
        path = self.make_database()
        for options in ({}, {"table": "orders", "query": "SELECT 1"}, {"query": " "},
                        {"table": "orders", "params": ()}, {"query": "SELECT 1", "params": "x"},
                        {"query": "SELECT 1", "sheet_name": ""}):
            with self.subTest(options=options):
                with self.assertRaises(ValueError):
                    SqlSource(path, **options)
        source = SqlSource(path, "orders")
        self.assertEqual(source.list_sheets(), ["orders"])
        self.assertEqual(len(list(source.iter_chunks())[0]), 4)


if __name__ == "__main__":
    unittest.main()
