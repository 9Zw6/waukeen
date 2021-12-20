import pandas as pd


def read_data_file(data_file):
    df = pd.read_hdf(data_file)
    # 清洗数据
    df.sort_values(by=["candle_begin_time"], inplace=True)
    df.drop_duplicates(subset=["candle_begin_time"], inplace=True)
    df.reset_index(inplace=True, drop=True)
    df["candle_begin_time"] = pd.to_datetime(df["candle_begin_time"])

    # 转换成15T周期数据
    rule_type = "15T"
    choose_rule = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    df = df.resample(rule=rule_type, on="candle_begin_time", label="left", closed="left").agg(choose_rule)

    # 筛选数据
    df.dropna(inplace=True)
    # 保留原index
    df.reset_index(inplace=True)
    volume_condition = (df["volume"] > 0)
    date_condition = (df["candle_begin_time"] >= pd.to_datetime("2017-09-01"))
    all_condition = volume_condition & date_condition
    df = df[all_condition]
    # 不保留原index
    df.reset_index(inplace=True, drop=True)

    return df
