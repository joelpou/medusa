import os
import sys
import pandas as pd
import uuid
# import mplfinance as fplt
# import matplotlib.pyplot as plt
import ta


def get_data_interval(data, start_str, end_str):
    start_date = pd.Timestamp(start_str).date()
    end_date = pd.Timestamp(end_str).date()
    start_i, end_i = 0, 0
    for i, row in df.iterrows():
        data_date = pd.Timestamp(row['Date']).date()
        if start_date == data_date:
            start_i = i
        elif end_date == data_date:
            end_i = i
            break
    return data.loc[start_i:end_i]


input_csv = str(sys.argv[1])

df = pd.read_csv(input_csv, parse_dates=True)
print(df.shape)
df = df.drop(['time_period_end', 'time_open', 'time_close', 'trades_count'], axis=1)
df = df.rename({'time_period_start': 'Date', 'price_open': 'Open',
                'price_high': 'High', 'price_low': 'Low', 'price_close': 'Close', 'volume_traded': 'Volume'}, axis=1)
# df.index = pd.DatetimeIndex(df['Date'])

# df = df.tail(500)
print(df.info())
print(df.drop(['Date'], axis=1))

col_keys = df.columns

# df = ta.add_all_ta_features(
#     df, open="Open", high="High", low="Low", close="Close", volume="Volume")
# print(df.shape)
# print(df.info())
#
# df.to_csv('../data/all_ta_features2.csv', index = False)

# print(col_keys)
start = '2014-10-01'
end = '2018-01-01'
df_sliced = get_data_interval(df, start, end)

dates = df_sliced[col_keys[0]]
opens = df_sliced[col_keys[1]]
closes = df_sliced[col_keys[2]]
highs = df_sliced[col_keys[3]]
lows = df_sliced[col_keys[4]]

# Momentum
"""
        window(int): n period.
"""
rsi_day = ta.wrapper.RSIIndicator(closes).rsi()

# Trend
"""
        window_fast(int): n period short-term.
        window_slow(int): n period long-term.
        window_sign(int): n period to signal.
        
        Chartists looking for more sensitivity may try a shorter short-term moving average and a longer long-term 
        moving average. MACD(5,35,5) is more sensitive than MACD(12,26,9) and might be better suited for weekly 
        charts. Chartists looking for less sensitivity may consider lengthening the moving averages. A less sensitive 
        MACD will still oscillate above/below zero, but the centerline crossovers and signal line crossovers will be 
        less frequent. 
"""
# macd = ta.wrapper.MACD(closes).macd()

ppo = ta.wrapper.PercentagePriceOscillator(closes, window_slow=5, window_fast=35, window_sign=5).ppo()
print(ppo.shape)
# print("ppo_signal: {}".format(ppo.ppo_signal()))
# print("ppo_histogram: {}".format(ppo.ppo_hist()))

# Volatility
"""
        window(int): n period.
        window_dev(int): n factor standard deviation
"""
boll_mavg = ta.wrapper.BollingerBands(closes).bollinger_mavg()
boll_hband = ta.wrapper.BollingerBands(closes).bollinger_hband()
boll_lband = ta.wrapper.BollingerBands(closes).bollinger_lband()

df_feats = pd.DataFrame(columns=['Date', 'Close', 'High', 'Low', 'Bol_avg', 'Bol_hb', 'Bol_lb', 'RSI', 'PPO'])
df_feats['Date'] = dates
df_feats['Close'] = closes
df_feats['High'] = highs
df_feats['Low'] = lows
df_feats['Bol_avg'] = boll_mavg
df_feats['Bol_hb'] = boll_hband
df_feats['Bol_lb'] = boll_lband
df_feats['RSI'] = rsi_day
df_feats['PPO'] = ppo

print("Dataframe", df_feats, sep='\n')

df_sliced = get_data_interval(df_feats, '2015-01-01', end)
df_sliced.to_csv('../data/df.csv', index=False)

ta.wrapper.OnBalanceVolumeIndicator
ta.wrapper.MFIIndicator