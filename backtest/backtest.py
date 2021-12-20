import os
import pandas as pd
from multiprocessing.pool import Pool
from utility import clean_data
from utility import signals
from utility import position
from utility import evaluate

pd.set_option("display.max_rows", 10000)
pd.set_option("expand_frame_repr", False)

current_file_path = os.path.abspath(__file__)
current_file_dir = os.path.dirname(current_file_path)
project_dir = os.path.dirname(current_file_dir)
data_file = project_dir + "/data/binance/spot/BTC-USDT/5m/BTC_USDT_5m.h5"
"""
读取BTC-USDT 5分钟K线数据 hdf文件
示例数据：
          candle_begin_time      open      high       low     close      volume
0       2017-08-17 04:05:00   4261.48   4261.48   4261.48   4261.48    0.000000
1       2017-08-17 04:10:00   4261.48   4261.48   4261.48   4261.48    0.000000
2       2017-08-17 04:15:00   4261.48   4264.88   4261.48   4261.48    0.484666
3       2017-08-17 04:20:00   4264.88   4266.29   4264.88   4266.29    2.328570
4       2017-08-17 04:25:00   4266.29   4270.41   4261.32   4261.45    6.306629
"""
global_df = clean_data.read_data_file(data_file)


def backtest_optimized_bolling(para):
    df = global_df.copy()
    df = signals.signal_optimized_bolling(df, para)
    df = position.position_for_OKEX_future(df)
    df = evaluate.equity_curve_for_OKEX_future(df, leverage_rate=3)
    para.append(df.iloc[-1]["equity_curve"])
    print(para)
    return para


if __name__ == "__main__":
    # 生成参数
    n_list = range(300, 600, 10)
    m_list = [i / 10 for i in range(10, 50, 1)]
    l_list = [i / 100 for i in range(1, 10, 1)]
    para_list = []
    for n in n_list:
        for m in m_list:
            for l in l_list:
                para = [n, m, l]
                para_list.append(para)

    # 多进程加速回测
    with Pool(processes=8) as pool:
        res = pool.map(backtest_optimized_bolling, para_list)
        res_df = pd.DataFrame(res, columns=["n", "m", "l", "equity_curve"])
        res_df.sort_values(by="equity_curve", ascending=False, inplace=True)
        res_file = project_dir + "/data/binance/spot/BTC-USDT/5m/optimized_bolling.csv"
        res_df.to_csv(res_file, index=False)

