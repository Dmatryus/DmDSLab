import unittest

import pandas as pd

from dmdslab.cleaning import (
    drop_almost_const_columns,
    drop_almost_empty_rows,
    drop_duplicates,
)


class TestDataPreprocessing(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame(
            {
                "A": [1, 2, None, 4],
                "B": [1, 1, 1, 1],
                "C": [None, None, None, None],
                "D": [1, 2, 3, 4],
            }
        )

    def test_drop_almost_empty_rows(self):
        # Тестирование удаления почти пустых строк
        result = drop_almost_empty_rows(self.df, threshold=0.2)
        self.assertEqual(len(result), 3)  # Ожидаем удаление одной строки

        result = drop_almost_empty_rows(self.df, threshold=0.9)
        self.assertEqual(len(result), 4)  # Ни одна строка не должна быть удалена

    def test_drop_almost_const_columns(self):
        # Тестирование удаления почти константных столбцов
        result = drop_almost_const_columns(self.df, threshold=0.8)
        self.assertNotIn("B", result.columns)  # Столбец B должен быть удален
        self.assertNotIn("C", result.columns)  # Столбец C должен быть удален

    def test_drop_duplicates(self):
        # Тестирование удаления дубликатов
        df_duplicates = pd.DataFrame({"A": [1, 1, 2], "B": [3, 3, 4]})
        result = drop_duplicates(df_duplicates, mode="rows")
        self.assertEqual(len(result), 2)  # Ожидаем удаление одной дублирующей строки

        result = drop_duplicates(df_duplicates, mode="columns")
        self.assertEqual(len(result.columns), 2)  # Все столбцы уникальны, изменений нет

        result = drop_duplicates(df_duplicates, mode="all")
        self.assertEqual(len(result), 2)  # Удаляются дубликаты строк


if __name__ == "__main__":
    unittest.main()
