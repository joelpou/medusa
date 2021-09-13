import os
import sys
import pandas as pd
import uuid
import mplfinance as fplt
import matplotlib.pyplot as plt

input_csv = str(sys.argv[1])

df = pd.read_csv(input_csv, parse_dates=True)
print(df.shape)
df = df.drop(['time_period_end', 'time_open', 'time_close', 'trades_count'], axis=1)
df = df.rename({'time_period_start': 'Date', 'price_open': 'Open' ,
                'price_high': 'High', 'price_low': 'Low', 'price_close': 'Close','volume_traded': 'Volume'}, axis=1)
df.index = pd.DatetimeIndex(df['Date'])

df = df.tail(6)
print(df.info())
print(df.drop(['Date'], axis=1))

col_keys = df.columns
# print(col_keys)
# opens = df[col_keys[1]]
# closes = df[col_keys[2]]
# highs = df[col_keys[3]]
# lows = df[col_keys[4]]

figname = '../images/' + str(uuid.uuid4())+'.png'
# fig = fplt.figure(num=1, figsize=(2, 2), dpi=50)
# dx = fig.add_subplot(111)
# dx.axis('off')

# mc = fplt.make_marketcolors(
#                             up='tab:green',down='tab:red',
#                             edge='inherit',
#                             wick={'up':'green','down':'red'},
#                             alpha=0.5,
#                             volume='tab:blue'
#                            )
#
# s  = fplt.make_mpf_style(marketcolors=mc)

fplt.plot(
    df,
    # ax= dx,
    axisoff=True,
    figscale=0.2,
    type='line',
    # style=s,
    style='charles',
    volume=True,
    # savefig=dict(fname=figname,dpi=100,pad_inches=0.25)
)


# plt.autoscale()
fplt.show()
# plt.savefig("../images/" + str(uuid.uuid4())+'.png')