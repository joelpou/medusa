import json
import requests
import pandas as pd
import os
import sys

output_path = str(sys.argv[1])

if not os.path.exists(output_path):
    os.makedirs(output_path)

# https://docs.coinapi.io/#ohlcv

url = 'https://rest.coinapi.io/v1/ohlcv/BITSTAMP_SPOT_BTC_USD/history?period_id=15MIN&time_start=2018-06-16T07:30:00&time_end=2021-01-01T00:00:00&limit=100000'
# # url = 'https://rest.coinapi.io/v1/ohlcv/BITSTAMP_SPOT_BTC_USD/latest?period_id=1MIN'

headers = {'X-CoinAPI-Key': '8F93BAD3-962F-4DC4-BA92-ABF45169BDE8'}
# headers = {'X-CoinAPI-Key': 'F88AD855-A3E8-49CA-8D93-CB1980815DCA'}
response = requests.get(url, headers=headers)

ohlcv_historical = response.json()

df = pd.DataFrame(ohlcv_historical)

print(df.shape)

# json.dump(df.to_dict(), outfile, indent=4, sort_keys=True)

df.to_csv(os.path.join(output_path, "coinapi_histdata_btc_usd_15MIN_2018-2021.csv"), index=False, header=True)