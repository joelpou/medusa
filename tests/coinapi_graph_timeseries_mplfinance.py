import matplotlib.pyplot as plt
import mplfinance as fplt
import os
import sys

import mplfinance.original_flavor
import pandas as pd
import uuid

input_csv = str(sys.argv[1])

df = pd.read_csv(input_csv, parse_dates=True)

df = df.drop(['time_period_end', 'time_open', 'time_close', 'trades_count'], axis=1)

df = df.tail(4)
print(df.info())
print(df)
print(df.drop(['time_period_start'], axis=1))
print(df.shape)

# print(df.tail(3))
col_keys = df.columns

opens = df[col_keys[1]]
closes = df[col_keys[2]]
highs = df[col_keys[3]]
lows = df[col_keys[4]]

fig = plt.figure(num=1, figsize=(3, 3), dpi=50, facecolor='w', edgecolor='k')
dx = fig.add_subplot(111)
fplt.original_flavor.candlestick2_ochl(dx, opens, closes, highs, lows, width=1, colorup='g', colordown='r')

# fplt.show()

# fplt.plot(smb, color="blue", linewidth=10, alpha=0.5)
# plt.axis('off')
plt.autoscale()

plt.savefig("./" + str(uuid.uuid4())+'.jpg')




# iterate time start 4hrs to time end

# dt_range = pd.date_range(start="2012-01-01T04:00:00.0000000Z", end="2012-01-01T08:00:00.0000000Z")
# df = df[df.index.isin(dt_range)]
# df.head()

#
# fplt.plot(
#             df,
#             type='candle',
#             style='yahoo',
#             title='Apple, March - 2020',
#             ylabel='Price ($)'
#         )

# apple_df = pd.read_csv('../data/AAPL.csv', index_col=0, parse_dates=True)
# print(apple_df.head())
# print(apple_df.tail())
# print(apple_df.shape)

# dt_range = pd.date_range(start="2020-09-09", end="2020-10-09")
# apple_df = apple_df[apple_df.index.isin(dt_range)]
# apple_df.head()

# fplt.plot(
#             apple_df,
#             type='candle',
#             style='yahoo',
#             title='Apple, March - 2020',
#             ylabel='Price ($)'
#         )

