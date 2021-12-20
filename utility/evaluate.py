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

    cols = ["pos", "open_pos_time", "contract_num", "open_pos_price", "open_pos_fee", "cash", "next_open",
            "close_pos_price", "close_pos_fee", "close_profit", "min_price", "min_profit", "net_value",
            "min_net_value", "min_margin_ratio", "closeout", "equity_change"]
    df.drop(cols, axis=1, inplace=True)

    return df


def equity_curve_for_OKEx_USDT_future_next_open(df, slippage=1 / 1000, c_rate=5 / 10000, leverage_rate=3, face_value=0.01,
                                                min_margin_ratio=1/100):
    """
    okex交割合约（usdt本位）资金曲线
    开仓价格是下根K线的开盘价，可以是其他的形式
    相比之前杠杆交易的资金曲线函数，逻辑简单很多：手续费的处理、爆仓的处理等。
    在策略中增加滑点的。滑点的处理和手续费是不同的。
    :param df:
    :param slippage:  滑点 ，可以用百分比，也可以用固定值。建议币圈用百分比，股票用固定值
    :param c_rate:  手续费，commission fees，默认为万分之5。不同市场手续费的收取方法不同，对结果有影响。比如和股票就不一样。
    :param leverage_rate:  杠杆倍数
    :param face_value:  一张合约的面值，0.01BTC
    :param min_margin_ratio: 最低保证金率，低于就会爆仓
    :return:
    """
    # =====下根k线开盘价
    df['next_open'] = df['open'].shift(-1)  # 下根K线的开盘价
    df['next_open'].fillna(value=df['close'], inplace=True)

    # =====找出开仓、平仓的k线
    condition1 = df['pos'] != 0  # 当前周期不为空仓
    condition2 = df['pos'] != df['pos'].shift(1)  # 当前周期和上个周期持仓方向不一样。
    open_pos_condition = condition1 & condition2

    condition1 = df['pos'] != 0  # 当前周期不为空仓
    condition2 = df['pos'] != df['pos'].shift(-1)  # 当前周期和下个周期持仓方向不一样。
    close_pos_condition = condition1 & condition2

    # =====对每次交易进行分组
    df.loc[open_pos_condition, 'start_time'] = df['candle_begin_time']
    df['start_time'].fillna(method='ffill', inplace=True)
    df.loc[df['pos'] == 0, 'start_time'] = pd.NaT

    # =====开始计算资金曲线
    initial_cash = 10000  # 初始资金，默认为10000元
    # ===在开仓时
    # 在open_pos_condition的K线，以开盘价计算买入合约的数量。（当资金量大的时候，可以用5分钟均价）
    df.loc[open_pos_condition, 'contract_num'] = initial_cash * leverage_rate / (face_value * df['open'])
    df['contract_num'] = np.floor(df['contract_num'])  # 对合约张数向下取整
    # 开仓价格：理论开盘价加上相应滑点
    df.loc[open_pos_condition, 'open_pos_price'] = df['open'] * (1 + slippage * df['pos'])
    # 开仓之后剩余的钱，扣除手续费
    df['cash'] = initial_cash - df['open_pos_price'] * face_value * df['contract_num'] * c_rate  # 即保证金

    # ===开仓之后每根K线结束时
    # 买入之后cash，contract_num，open_pos_price不再发生变动
    for _ in ['contract_num', 'open_pos_price', 'cash']:
        df[_].fillna(method='ffill', inplace=True)
    df.loc[df['pos'] == 0, ['contract_num', 'open_pos_price', 'cash']] = None

    # ===在平仓时
    # 平仓价格
    df.loc[close_pos_condition, 'close_pos_price'] = df['next_open'] * (1 - slippage * df['pos'])
    # 平仓之后剩余的钱，扣除手续费
    df.loc[close_pos_condition, 'close_pos_fee'] = df['close_pos_price'] * face_value * df['contract_num'] * c_rate

    # ===计算利润
    # 开仓至今持仓盈亏
    df['profit'] = face_value * df['contract_num'] * (df['close'] - df['open_pos_price']) * df['pos']
    # 平仓时理论额外处理
    df.loc[close_pos_condition, 'profit'] = face_value * df['contract_num'] * (
            df['close_pos_price'] - df['open_pos_price']) * df['pos']
    # 账户净值
    df['net_value'] = df['cash'] + df['profit']

    # ===计算爆仓
    # 至今持仓盈亏最小值
    df.loc[df['pos'] == 1, 'price_min'] = df['low']
    df.loc[df['pos'] == -1, 'price_min'] = df['high']
    df['profit_min'] = face_value * df['contract_num'] * (df['price_min'] - df['open_pos_price']) * df['pos']
    # 账户净值最小值
    df['net_value_min'] = df['cash'] + df['profit_min']
    # 计算保证金率
    df['margin_ratio'] = df['net_value_min'] / (face_value * df['contract_num'] * df['price_min'])
    # 计算是否爆仓
    df.loc[df['margin_ratio'] <= (min_margin_ratio + c_rate), '是否爆仓'] = 1

    # ===平仓时扣除手续费
    df.loc[close_pos_condition, 'net_value'] -= df['close_pos_fee']
    # 应对偶然情况：下一根K线开盘价格价格突变，在平仓的时候爆仓。此处处理有省略，不够精确。
    df.loc[close_pos_condition & (df['net_value'] < 0), '是否爆仓'] = 1

    # ===对爆仓进行处理
    df['是否爆仓'] = df.groupby('start_time')['是否爆仓'].fillna(method='ffill')
    df.loc[df['是否爆仓'] == 1, 'net_value'] = 0

    # =====计算资金曲线
    df['equity_change'] = df['net_value'].pct_change()
    df.loc[open_pos_condition, 'equity_change'] = df.loc[open_pos_condition, 'net_value'] / initial_cash - 1  # 开仓日的收益率
    df['equity_change'].fillna(value=0, inplace=True)
    df['equity_curve'] = (1 + df['equity_change']).cumprod()

    # =====删除不必要的数据，并存储
    df.drop(['next_open', 'contract_num', 'open_pos_price', 'cash', 'close_pos_price', 'close_pos_fee',
             'profit', 'net_value', 'price_min', 'profit_min', 'net_value_min', 'margin_ratio', '是否爆仓'],
            axis=1, inplace=True)

    return df

