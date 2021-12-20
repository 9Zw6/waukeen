import pandas as pd


def position_for_OKEX_future(df):
    df["pos"] = df["signal"].shift(1)
    df["pos"].fillna(value=0, inplace=True)

    # OKEX future 下午4点清算，无法交易
    condition1 = (df["candle_begin_time"].dt.hour == 16)
    condition2 = (df["candle_begin_time"].dt.minute == 0)
    condition = condition1 & condition2
    df.loc[condition, "pos"] = None
    df["pos"].fillna(method="ffill", inplace=True)

    df.drop(["signal"], axis=1, inplace=True)
    return df
