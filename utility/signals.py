import pandas as pd


def signal_optimized_bolling(df, para=[440, 2.6, 0.05]):
    n = para[0]
    m = para[1]
    l = para[2]

    # 计算bolling上中下轨
    df["median"] = df["close"].rolling(n, min_periods=1).mean()
    df["std"] = df["close"].rolling(n, min_periods=1).std(ddof=0)
    df["upper"] = df["median"] + m * df["std"]
    df["lower"] = df["median"] - m * df["std"]

    # 计算signal long = 1
    condition1 = df["close"] > df["upper"]
    condition2 = df["close"].shift(1) <= df["upper"].shift(1)
    df.loc[condition1 & condition2, "signal_simple_bolling"] = 1

    # 计算signal long = 0
    condition1 = df["close"] < df["median"]
    condition2 = df["close"].shift(1) >= df["median"].shift(1)
    df.loc[condition1 & condition2, "signal_simple_bolling"] = 0

    # 计算signal short = -1
    condition1 = df["close"] < df["lower"]
    condition2 = df["close"].shift(1) >= df["lower"].shift(1)
    df.loc[condition1 & condition2, "signal_simple_bolling"] = -1

    # 计算signal short = 0
    condition1 = df["close"] > df["median"]
    condition2 = df["close"].shift(1) <= df["median"].shift(1)
    df.loc[condition1 & condition2, "signal_simple_bolling"] = 0

    # 补充NaN
    df["signal_simple_bolling"].fillna(method="ffill", inplace=True)
    df["signal_simple_bolling"].fillna(value=0, inplace=True)

    # 计算close偏离median百分比
    df["diff"] = (df["close"] - df["median"]) / df["median"]

    # 优化后的signal long = 1
    condition1 = (1 == df["signal_simple_bolling"])
    condition2 = (df["diff"] <= l) & (df["diff"] > 0)
    df.loc[condition1 & condition2, "signal"] = 1

    # 优化后的signal short = -1
    condition1 = (-1 == df["signal_simple_bolling"])
    condition2 = (df["diff"] >= -l) & (df["diff"] < 0)
    df.loc[condition1 & condition2, "signal"] = -1

    # signal long/short = 0 不变
    condition1 = (0 == df["signal_simple_bolling"])
    df.loc[condition1, "signal"] = 0

    df["signal"].fillna(method="ffill", inplace=True)

    df.drop(["median", "std", "upper", "lower", "signal_simple_bolling", "diff"], axis=1, inplace=True)

    return df
