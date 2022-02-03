import os
import sys
import random
import yfinance as yf
import mplfinance as fplt
import pandas as pd
import ta

output_path = sys.argv[1]
os.makedirs(output_path, exist_ok=True)


def label_candle_data(data, last_list):
    # print(data)
    trend = ''
    current_highs = data['High'].tolist()
    current_highest = max(current_highs)
    current_lows = data['Low'].tolist()
    current_lowest = min(current_lows)
    current_closes = data['Close'].tolist()
    current_close = current_closes[-1]

    if current_close > last_list[0]:
        trend = 'UP'
    elif current_close < last_list[1]:
        trend = 'DOWN'
    elif last_list[0] >= current_close >= last_list[1]:
        trend = 'SIDE'

    return [current_highest, current_lowest, current_close], trend


def save_plot(chunk, output):
    # https://github.com/matplotlib/mplfinance/blob/master/examples/styles.ipynb
    mc = fplt.make_marketcolors(up='g', down='r', inherit=True)
    s = fplt.make_mpf_style(base_mpf_style='nightclouds', marketcolors=mc)
    fplt.plot(
        chunk,
        axisoff=True,
        figscale=0.2,
        figratio=(1.00, 1.00),
        type='candle',
        # style='yahoo',
        style=s,
        # volume=True,
        savefig=dict(fname=output, dpi=35)
    )


def main():
    cnt = 0
    window = 14
    offset = 2
    # table = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
    # df = table[0]
    # tickers = df['Symbol'].to_list()
    # print(tickers)
    # full_stock_data = yf.download(stockdata, '2010-01-01', '2021-03-03')
    # print(full_stock_data['Volume'])
    tickers = ['TSLA', 'AAPL', 'NFLX', 'GOOG', 'AMZN', 'FB', 'EBAY', 'ETSY', 'F', 'IBM', 'ENPH', 'COST']
    for ticker in tickers:
        print(ticker)
        ticker_data = yf.Ticker(ticker)
        ticker_df = ticker_data.history(period='1d', start='2012-12-01', end='2021-01-01')
        ticker_df = ticker_df.drop(['Dividends', 'Stock Splits'], axis=1)
        ticker_df.reset_index(inplace=True)  # Convert the Datetimeindex to 'Date' column
        ticker_df.index = pd.DatetimeIndex(ticker_df['Date'])
        ticker_volume = ticker_df["Volume"]

        ticker_volume_sma = ta.wrapper.SMAIndicator(ticker_volume, window).sma_indicator()

        last_highest = max(ticker_df[:window]['High'].tolist())
        last_lowest = min(ticker_df[:window]['Low'].tolist())
        last_close = ticker_df[:window]['Close'].tolist()[-1]

        last_list = [last_highest, last_lowest, last_close]

        ticker_df = ticker_df[window:]
        ticker_volume_sma = ticker_volume_sma[window:]

        print(ticker_df.info())
        cnt_up, cnt_down, cnt_side = 0, 0, 0

        for i, row in enumerate(ticker_df.iterrows()):
            if i % (window - offset) == 0:
                chunk = ticker_df[i:i + window - offset]
                vol_avg = float(chunk['Volume'].mean())
                vol_sma = float(ticker_volume_sma[i])
                if vol_avg > vol_sma:
                    cnt = cnt + 1
                    print('count: ' + str(cnt))
                    last_list, trend = label_candle_data(chunk, last_list)

                    if trend == 'UP':
                        cnt_up = cnt_up + 1
                    elif trend == 'DOWN':
                        cnt_down = cnt_down + 1
                    elif trend == 'SIDE':
                        cnt_side = cnt_side + 1

                    dates = pd.to_datetime(chunk['Date']).dt.date
                    date_interval = str(dates[0]) + '_' + str(dates[-1])
                    output_dir = output_path + '/' + trend
                    os.makedirs(output_dir, exist_ok=True)
                    output = output_dir + '/' + ticker + '_' + date_interval + '_' + str(len(chunk)) + '_' + trend + '.png'
                    save_plot(chunk, output)

                    offset = random.randint(2, 8)

                else:
                    print('skip...')
    print('Total candle images labeled: ' + str(cnt))


if '__main__':
    main()
