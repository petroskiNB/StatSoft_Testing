"""python parquet_example.py — создаёт настоящий временный Parquet и читает его."""

from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from table_import import ParquetSource, ImportSession


def main():
    with TemporaryDirectory() as directory:
        path = Path(directory) / "sales.parquet"
        frame = pd.DataFrame({
            "id": pd.Series([1, 2, 3, 4, 5], dtype="Int64"),
            "amount": pd.Series([10, None, 30.5, 40, 50], dtype="Float64"),
        })
        frame.to_parquet(path, engine="pyarrow", index=False, row_group_size=2)

        source = ParquetSource(path)
        print("Листы:", source.list_sheets())
        metadata = source.scan(sheet="sales", sample_rows=3)
        print("Метаданные:", metadata)
        with ImportSession(source, "sales", 2, dtype={"amount": "Float64"}, metadata=metadata) as session:
            print("Первая порция:")
            print(next(session))
            session.pause()
            print("Пауза:", session.progress)
            session.resume()
            print("Результат:")
            print(session.run())
            print(session.progress)


if __name__ == "__main__":
    main()
