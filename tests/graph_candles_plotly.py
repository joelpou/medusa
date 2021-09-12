import plotly.graph_objects as go
import pandas as pd
from datetime import datetime
import os
import sys
import uuid

input_csv = str(sys.argv[1])

df = pd.read_csv(input_csv)

print(df.head())
print(df.tail())

fig = go.Figure(data=[go.Candlestick(x=df['time_period_start'],
                open=df['price_open'],
                high=df['price_high'],
                low=df['price_low'],
                close=df['price_close'])])

fig.show()
# fig.write_image("./" + str(uuid.uuid4())+'.png')