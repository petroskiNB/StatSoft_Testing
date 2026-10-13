from contextlib import closing
import pandas as pd
from table_import import XlsxSource, SourceReadError
from openpyxl import Workbook

#создание книги для теста
book = Workbook()

sheet = book.active
sheet.title = "Продажи"

sheet.append(["id", "amount"])
sheet.append([1, 10])
sheet.append([2, None])
sheet.append([3, 30])
sheet.append([4, 40])
sheet.append([5, 50])

book.save("sales_example.xlsx")
book.close()

try:
    source = XlsxSource("sales_example.xlsx")
    sheet_names = source.list_sheets()
    print(f'Доступные листы: {sheet_names}')

    if not sheet_names:
        raise ValueError("В книге нет доступных листов")
    
    sheet_name = sheet_names[0]
    print("Выбран лист:", sheet_name)

    metadata = source.scan(
        sheet=sheet_name,
        sample_rows=3,
    )

    print("Лист:", metadata.name)
    print("Проверено строк:", metadata.sampled_rows)

    for column in metadata.columns:
        print(column.name, column.dtype, column.missing_count)

    with closing(
        source.iter_chunks(sheet=sheet_name, chunk_size=2)
    ) as chunks:
        result = pd.concat(list(chunks), ignore_index=True)

    print(result)


except SourceReadError as error:
    print("Ошибка:", error)