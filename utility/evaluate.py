import numpy as np
import pandas as pd


"""
leverage_rate : 杠杆倍数
face_value=0.01 : 合约面值
c_rate=5/10000 : 手续费率
slippage=1/1000 : 滑点
min_margin_ratio=1/100 : 最低保证金率
"""
def equity_curve_for_OKEX_future(df,
                                 leverage_rate=3,
                                 face_value=0.01,
                                 c_rate=5/10000,
                                 slippage=1/1000,
                                 min_margin_ratio=1/100):
    # 找到开仓行
    condition1 = (0 != df["pos"])
    condition2 = (df["pos"] != df["pos"].shift(1))
    open_pos_condition = condition1 & condition2

    # 设置合约开仓时间
    df.loc[open_pos_condition, "open_pos_time"] = df["candle_begin_time"]

    # 设置合约开仓数
    # 初始资金1W
    initial_cash = 10000
    df.loc[open_pos_condition, "contract_num"] = initial_cash * leverage_rate / (face_value * df["open"])
    df["contract_num"] = np.floor(df["contract_num"])

    # 设置实际合约开仓价
    df.loc[open_pos_condition, "open_pos_price"] = df["open"] * (1 + df["pos"] * slippage)

    # 设置实际开仓手续费
    df.loc[open_pos_condition, "open_pos_fee"] = df["open_pos_price"] * face_value * df["contract_num"] * c_rate

    # 设置开仓保证金
    df.loc[open_pos_condition, "cash"] = initial_cash - df["open_pos_fee"]

    # 开仓数据整理
    df.fillna(method="ffill", inplace=True)
    none_pos_condition = (0 == df["pos"])
    cols = ["open_pos_time", "contract_num", "open_pos_price", "open_pos_fee", "cash"]
    df.loc[none_pos_condition, cols] = None

    # 找到平仓行
    condition1 = (0 != df["pos"])
    condition2 = (df["pos"] != df["pos"].shift(-1))
    close_pos_condition = condition1 & condition2

    # 设置合约平仓价
    open_shift = df["open"].shift(-1)
    open_shift.fillna(value=df["close"], inplace=True)
    df.loc[close_pos_condition, "next_open"] = open_shift

    # 设置实际合约平仓价
    df.loc[close_pos_condition, "close_pos_price"] = df["next_open"] * (1 - df["pos"] * slippage)

    # 设置实际合约平仓手续费
    df.loc[close_pos_condition, "close_pos_fee"] = df["close_pos_price"] * face_value * df["contract_num"] * c_rate

    # 设置平仓保证金
    df.loc[close_pos_condition, "cash"] = df["cash"] - df["close_pos_fee"]

    # 设置持仓利润
    # 10 - NaN = NaN
    df["close_profit"] = (df["close"] - df["open_pos_price"]) * face_value * df["contract_num"] * df["pos"]

    # 平仓利润单独计算
    df.loc[close_pos_condition, "close_profit"] = (df["close_pos_price"] - df["open_pos_price"]) * face_value * df["contract_num"] * df["pos"]

    # 设置持仓最小利润
    long_pos_condition = (1 == df["pos"])
    df.loc[long_pos_condition, "min_price"] = df["low"]

    # 防止平多仓时爆仓
    df.loc[long_pos_condition & close_pos_condition, "min_price"] = df[["low", "close_pos_price"]].min(axis=1)
    short_pos_condition = (-1 == df["pos"])
    df.loc[short_pos_condition, "min_price"] = df["high"]

    # 防止平空仓时爆仓
    df.loc[short_pos_condition & close_pos_condition, "min_price"] = df[["high", "close_pos_price"]].max(axis=1)
    df["min_profit"] = (df["min_price"] - df["open_pos_price"]) * face_value * df["contract_num"] * df["pos"]

    # 平仓数据整理
    df.fillna(method="bfill", inplace=True)
    none_pos_condition = (0 == df["pos"])
    cols.extend(["next_open", "close_pos_price", "close_pos_fee", "close_profit", "min_price", "min_profit"])
    df.loc[none_pos_condition, cols] = None

    # 设置净值
    df["net_value"] = df["cash"] + df["close_profit"]
    df["min_net_value"] = df["cash"] + df["min_profit"]

    # 设置最低保证金率
    df["min_margin_ratio"] = df["min_net_value"] / (face_value * df["contract_num"] * df["min_price"])

    # 设置爆仓标记
    closeout_condition = (df["min_margin_ratio"] < (min_margin_ratio + c_rate))
    df.loc[closeout_condition, "closeout"] = 1
    df["closeout"].fillna(method="ffill", inplace=True)
    df.loc[none_pos_condition, "closeout"] = np.NaN

    # 设置爆仓净值
    closeout_condition = (1 == df["closeout"])
    df.loc[closeout_condition, "net_value"] = 0

    # 设置资金变化率
    # NaN的行 pct_change是0, none_pos_condition的行是NaN
    # 0的行   pct_change是NaN, 爆仓会出现 net_value = 0
    df["equity_change"] = df["net_value"].pct_change(1)

    # 开仓日的资金变化率单独计算
    df.loc[open_pos_condition, "equity_change"] = df["net_value"] / initial_cash - 1
    df["equity_change"].fillna(value=0, inplace=True)

    # 设置资金曲线
    df["equity_curve"] = (1 + df["equity_change"]).cumprod()

    # 保留pos
    cols = ["open_pos_time", "contract_num", "open_pos_price", "open_pos_fee", "cash", "next_open",
            "close_pos_price", "close_pos_fee", "close_profit", "min_price", "min_profit", "net_value",
            "min_net_value", "min_margin_ratio", "closeout"]
    df.drop(cols, axis=1, inplace=True)

    return df
