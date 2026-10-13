"""python import_example.py — импорт с преобразованием, паузой и прогрессом."""

from pathlib import Path
from tempfile import TemporaryDirectory

from table_import import CsvSource, ImportSession


def show_progress(progress):
    if progress.status == "completed":
        print(f"Завершено: {progress.rows_imported} строк, 100%")
    elif progress.total_rows_exceeded:
        print(f"Принято {progress.rows_imported} строк: ожидаемое число превышено, импорт ещё не завершён")
    elif progress.fraction is not None:
        label = "около " if progress.total_rows_is_estimate else ""
        print(f"{progress.status}: {progress.rows_imported} строк, {label}{progress.fraction:.0%}")
    else:
        print(f"{progress.status}: {progress.rows_imported} строк, процент неизвестен")


def main():
    with TemporaryDirectory() as directory:
        path = Path(directory) / "sales.csv"
        path.write_text("id;amount\n1;10\n2;\n3;30.5\n4;40\n5;50\n", encoding="utf-8")
        source = CsvSource(path, delimiter=";")
        metadata = source.scan(sample_rows=2)
        with ImportSession(
            source, sheet="sales", chunk_size=2,
            dtype={"id": "Int64", "amount": "Float64"},
            metadata=metadata,
        ) as session:
            print("Первая порция:")
            print(next(session))
            session.pause()
            print("Пауза:", session.progress)
            show_progress(session.progress)

            session.resume()
            for chunk in session:
                print("Следующая порция:")
                print(chunk)
                show_progress(session.progress)
            result = session.result()
            print("Завершено:", session.progress)
            show_progress(session.progress)
            print(result)
            print(result.dtypes)


if __name__ == "__main__":
    main()
