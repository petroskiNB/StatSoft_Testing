from contextlib import closing
import pandas as pd
from table_import import CsvSource, SourceReadError

try:
    source = CsvSource("sales.csv", delimiter=";")
    metadata = source.scan(sample_rows=3)

    print("Лист:", metadata.name)
    print("Проверено строк:", metadata.sampled_rows)

    for column in metadata.columns:
        print(column.name, column.dtype, column.missing_count)

    with closing(source.iter_chunks(chunk_size=2)) as chunks:
        result = pd.concat(list(chunks), ignore_index=True)
    print(result)

except SourceReadError as error:
    print("Ошибка:", error)