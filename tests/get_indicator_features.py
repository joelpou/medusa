import os
import sys
import pandas as pd
import uuid
# import mplfinance as fplt
# import matplotlib.pyplot as plt
import ta

input_csv = str(sys.argv[1])

df = pd.read_csv(input_csv, parse_dates=True)
print(df.shape)
df = df.drop(['time_period_end', 'time_open', 'time_close', 'trades_count'], axis=1)
df = df.rename({'time_period_start': 'Date', 'price_open': 'Open' ,
                'price_high': 'High', 'price_low': 'Low', 'price_close': 'Close','volume_traded': 'Volume'}, axis=1)
df.index = pd.DatetimeIndex(df['Date'])

df = df.tail(500)
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
# opens = df[col_keys[1]]
# closes = df[col_keys[2]]
# highs = df[col_keys[3]]
# lows = df[col_keys[4]]
