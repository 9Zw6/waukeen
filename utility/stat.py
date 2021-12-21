import pandas as pd
import numpy as np
import itertools


def transfer_equity_curve_to_trade(equity_curve):
    condition1 = equity_curve['pos'] != 0
    condition2 = equity_curve['pos'] != equity_curve['pos'].shift(1)
    open_pos_condition = condition1 & condition2

    if 'start_time' not in equity_curve.columns:
        equity_curve.loc[open_pos_condition, 'start_time'] = equity_curve['candle_begin_time']
        equity_curve['start_time'].fillna(method='ffill', inplace=True)
        equity_curve.loc[equity_curve['pos'] == 0, 'start_time'] = pd.NaT

    trade = pd.DataFrame()

    for _index, group in equity_curve.groupby('start_time'):
        trade.loc[_index, 'signal'] = group['pos'].iloc[0]

        if 'leverage_rate' in group:
            trade.loc[_index, 'leverage_rate'] = group['leverage_rate'].iloc[0]

        g = group[group['pos'] != 0]
        trade.loc[_index, 'end_bar'] = g.iloc[-1]['candle_begin_time']
        trade.loc[_index, 'start_price'] = g.iloc[0]['open']
        trade.loc[_index, 'end_price'] = g.iloc[-1]['close']
        trade.loc[_index, 'bar_num'] = g.shape[0]
        trade.loc[_index, 'change'] = (group['equity_change'] + 1).prod() - 1
        trade.loc[_index, 'end_equity_curve'] = g.iloc[-1]['equity_curve']
        trade.loc[_index, 'min_equity_curve'] = g['equity_curve'].min()

    return trade


def strategy_evaluate(equity_curve, trade):
    results = pd.DataFrame()

    results.loc[0, '累积净值'] = round(equity_curve['equity_curve'].iloc[-1], 2)

    annual_return = (equity_curve['equity_curve'].iloc[-1] / equity_curve['equity_curve'].iloc[0]) ** (
        '1 days 00:00:00' / (equity_curve['candle_begin_time'].iloc[-1] - equity_curve['candle_begin_time'].iloc[0]) * 365) - 1
    results.loc[0, '年化收益'] = str(round(annual_return * 100, 2)) + '%'

    equity_curve['max2here'] = equity_curve['equity_curve'].expanding().max()
    equity_curve['dd2here'] = equity_curve['equity_curve'] / equity_curve['max2here'] - 1
    end_date, max_draw_down = tuple(equity_curve.sort_values(by=['dd2here']).iloc[0][['candle_begin_time', 'dd2here']])
    start_date = equity_curve[equity_curve['candle_begin_time'] <= end_date].sort_values(by='equity_curve', ascending=False).iloc[0]['candle_begin_time']
    equity_curve.drop(['max2here', 'dd2here'], axis=1, inplace=True)
    results.loc[0, '最大回撤'] = format(max_draw_down, '.2%')
    results.loc[0, '最大回撤开始时间'] = str(start_date)
    results.loc[0, '最大回撤结束时间'] = str(end_date)

    results.loc[0, '年化收益/回撤比'] = round(abs(annual_return / max_draw_down), 2)

    results.loc[0, '盈利笔数'] = len(trade.loc[trade['change'] > 0])  # 盈利笔数
    results.loc[0, '亏损笔数'] = len(trade.loc[trade['change'] <= 0])  # 亏损笔数
    results.loc[0, '胜率'] = format(results.loc[0, '盈利笔数'] / len(trade), '.2%')  # 胜率

    results.loc[0, '每笔交易平均盈亏'] = format(trade['change'].mean(), '.2%')  # 每笔交易平均盈亏
    results.loc[0, '盈亏收益比'] = round(trade.loc[trade['change'] > 0]['change'].mean() / \
                                    trade.loc[trade['change'] < 0]['change'].mean() * (-1), 2)  # 盈亏比

    results.loc[0, '单笔最大盈利'] = format(trade['change'].max(), '.2%')  # 单笔最大盈利
    results.loc[0, '单笔最大亏损'] = format(trade['change'].min(), '.2%')  # 单笔最大亏损

    trade['持仓时间'] = trade['end_bar'] - trade.index
    max_days, max_seconds = trade['持仓时间'].max().days, trade['持仓时间'].max().seconds
    max_hours = max_seconds // 3600
    max_minute = (max_seconds - max_hours * 3600) // 60
    results.loc[0, '单笔最长持有时间'] = str(max_days) + ' 天 ' + str(max_hours) + ' 小时 ' + str(max_minute) + ' 分钟'  # 单笔最长持有时间

    min_days, min_seconds = trade['持仓时间'].min().days, trade['持仓时间'].min().seconds
    min_hours = min_seconds // 3600
    min_minute = (min_seconds - min_hours * 3600) // 60
    results.loc[0, '单笔最短持有时间'] = str(min_days) + ' 天 ' + str(min_hours) + ' 小时 ' + str(min_minute) + ' 分钟'  # 单笔最短持有时间

    mean_days, mean_seconds = trade['持仓时间'].mean().days, trade['持仓时间'].mean().seconds
    mean_hours = mean_seconds // 3600
    mean_minute = (mean_seconds - mean_hours * 3600) // 60
    results.loc[0, '平均持仓周期'] = str(mean_days) + ' 天 ' + str(mean_hours) + ' 小时 ' + str(mean_minute) + ' 分钟'  # 平均持仓周期

    results.loc[0, '最大连续盈利笔数'] = max(
        [len(list(v)) for k, v in itertools.groupby(np.where(trade['change'] > 0, 1, np.nan))])  # 最大连续盈利笔数
    results.loc[0, '最大连续亏损笔数'] = max(
        [len(list(v)) for k, v in itertools.groupby(np.where(trade['change'] < 0, 1, np.nan))])  # 最大连续亏损笔数

    equity_curve.set_index('candle_begin_time', inplace=True)
    monthly_return = equity_curve[['equity_change']].resample(rule='M').apply(lambda x: (1 + x).prod() - 1)

    return results.T, monthly_return
