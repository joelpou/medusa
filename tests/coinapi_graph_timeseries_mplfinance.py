import matplotlib.pyplot as plt
import mplfinance as fplt
import os
import sys

import mplfinance.original_flavor
import pandas as pd

input_csv = str(sys.argv[1])

df = pd.read_csv(input_csv, index_col=0, parse_dates=True)

print(df.head())
print(df.tail())
print(df.shape)

# iterate time start 4hrs to time end

dt_range = pd.date_range(start="2012-01-01T04:00:00.0000000Z", end="2012-01-01T08:00:00.0000000Z")
df = df[df.index.isin(dt_range)]
df.head()

mplfinance.original_flavor.candlestick_ochl()

fplt.plot(
            df,
            type='candle',
            style='yahoo',
            title='Apple, March - 2020',
            ylabel='Price ($)'
        )

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

