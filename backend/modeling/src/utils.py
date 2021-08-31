import pandas as pd
from pandas.tseries.holiday import (AbstractHolidayCalendar, Easter,
                                    EasterMonday, Holiday)
from pandas.tseries.offsets import Day


def build_time_series(data: pd.DataFrame, target: list, window_size: int, forecast_size: int):
    list_x, list_y = [], []
    for i in range(0, data.shape[0] - forecast_size - window_size + 1):
        x = data[i : i + window_size]  # noqa: E203
        y = data[target][i + window_size : i + window_size + forecast_size]  # noqa: E203
        list_x.append(x)
        list_y.append(y)
    print("# Rolling windows :", len(list_x), "of shape :", list_x[0].shape)
    return list_x, list_y


def split_train_val_test(df: pd.DataFrame, split_size_val: float, split_size_test: float):
    n = len(df)
    train_df = df[: int(n * (1 - split_size_val))]
    val_df = df[int(n * (1 - split_size_val)) : int(n * (1 - split_size_test))]  # noqa: E203
    test_df = df[int(n * (1 - split_size_test)) :]  # noqa: E203
    print("Training data shape :", train_df.shape)
    print("Validation data shape :", val_df.shape)
    print("Testing data shape :", test_df.shape, "\n")

    return train_df, val_df, test_df


def plot_test_set(x_test, y_test):
    pass


class FrBusinessCalendar(AbstractHolidayCalendar):
    """Custom Holiday calendar for France based on
      https://en.wikipedia.org/wiki/Public_holidays_in_France
    - 1 January: New Year's Day
    - Moveable: Easter Monday (Monday after Easter Sunday)
    - 1 May: Labour Day
    - 8 May: Victory in Europe Day
    - Moveable Ascension Day (Thursday, 39 days after Easter Sunday)
    - 14 July: Bastille Day
    - 15 August: Assumption of Mary to Heaven
    - 1 November: All Saints' Day
    - 11 November: Armistice Day
    - 25 December: Christmas Day
    """

    rules = [
        Holiday("New Years Day", month=1, day=1),
        EasterMonday,
        Holiday("Labour Day", month=5, day=1),
        Holiday("Victory in Europe Day", month=5, day=8),
        Holiday("Ascension Day", month=1, day=1, offset=[Easter(), Day(39)]),
        Holiday("Bastille Day", month=7, day=14),
        Holiday("Assumption of Mary to Heaven", month=8, day=15),
        Holiday("All Saints Day", month=11, day=1),
        Holiday("Armistice Day", month=11, day=11),
        Holiday("Christmas Day", month=12, day=25),
    ]
