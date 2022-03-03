import os
import sys
import random
import yfinance as yf
import matplotlib.pyplot
import mplfinance as fplt

import pandas as pd
import ta
from statistics import mean

output_path = sys.argv[1]
os.makedirs(output_path, exist_ok=True)


def show_plot(chunk):
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
    )
    # disable the "show plots in tool window" option under settings > tools > Python Scientific on PyCharm
    fplt.show()


def label_candle_data(data):
    trend = ''
    current_highs = data['High'].tolist()
    current_highest = max(current_highs)
    # i_highest = current_highs.index(current_highest)

    current_lows = data['Low'].tolist()
    current_lowest = min(current_lows)
    # i_lowest = current_lows.index(current_lowest)

    current_closes = data['Close'].tolist()
    # first_close = current_closes[0]
    last_close = current_closes[-1]

    # h_diff = current_highest - last_close
    # l_diff = current_lowest - last_close
    threshold = 0.22
    avg_closes = mean(current_closes)
    diff = last_close - avg_closes

    # if last_close > current_highest:
    if diff > threshold:
        trend = 'UP'
    # if last_close < lowest_highest:
    elif diff < -threshold:
        trend = 'DOWN'
    # current_highest >= current_close >= lowest_highest:
    else:
        trend = 'SIDE'

    assert trend

    # print(trend)
    # show_plot(data)

    return trend


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
        savefig=dict(fname=output, dpi=64)
    )


def main():
    cnt = 0
    window = 14
    offset = 2
    samples = 6
    classes = 3

    table = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
    df = table[0]
    tickers = df['Symbol'].to_list()
    # ticker_bad = ['BF.B', 'BRK.B', 'OGN', 'OTIS', 'CARR']  # Data doesn't exist for error
    #
    # for t_bad in ticker_bad:
    #     i = tickers.index(t_bad)
    #     tickers.pop(i)

    random.shuffle(tickers)
    # print(tickers)
    # full_stock_data = yf.download(stockdata, '2010-01-01', '2021-03-03')
    # print(full_stock_data['Volume'])
    # tickers = ['TSLA', 'AAPL', 'NFLX', 'GOOG', 'AMZN', 'FB', 'EBAY', 'ETSY', 'F', 'IBM', 'ENPH', 'COST']
    for ticker in tickers:
        lcnt_up, lcnt_down, lcnt_side = 0, 0, 0
        print(ticker)
        ticker_data = yf.Ticker(ticker)
        ticker_df = ticker_data.history(period='1d', start='2012-12-01', end='2019-01-01')

        if len(ticker_df) == 0:
            continue

        ticker_df = ticker_df.drop(['Dividends', 'Stock Splits'], axis=1)
        ticker_df.reset_index(inplace=True)  # Convert the Datetimeindex to 'Date' column
        ticker_df.index = pd.DatetimeIndex(ticker_df['Date'])
        ticker_volume = ticker_df["Volume"]

        ticker_volume_sma = ta.wrapper.SMAIndicator(ticker_volume, window).sma_indicator()

        # last_highest = max(ticker_df[:window]['High'].tolist())
        # last_lowest = min(ticker_df[:window]['Low'].tolist())
        # last_close = ticker_df[:window]['Close'].tolist()[-1]
        #
        # last_list = [last_highest, last_lowest, last_close]

        ticker_df = ticker_df[window:]
        ticker_volume_sma = ticker_volume_sma[window:]

        print('# Candles: ' + str(len(ticker_df)))

        # print(ticker_df.info())

        for i, row in enumerate(ticker_df.iterrows()):
            if i % (window - offset) == 0:
                chunk = ticker_df[i:i + window - offset]
                vol_avg = float(chunk['Volume'].mean())
                vol_sma = float(ticker_volume_sma[i])
                # capture only patterns that have the volume above 14-days simple moving average
                if vol_avg > vol_sma:
                    trend = label_candle_data(chunk)

                    if trend == '':
                        continue

                    if trend == 'UP':
                        if lcnt_up == samples / classes:  # split classes uniformly
                            continue
                        lcnt_up += 1
                    elif trend == 'DOWN':
                        if lcnt_down == samples / classes:
                            continue
                        lcnt_down += 1
                    elif trend == 'SIDE':
                        if lcnt_side == samples / classes:
                            continue
                        lcnt_side += 1

                    cnt = cnt + 1

                    dates = pd.to_datetime(chunk['Date']).dt.date
                    date_interval = str(dates[0]) + '_' + str(dates[-1])
                    output_dir = output_path + '/' + trend
                    os.makedirs(output_dir, exist_ok=True)
                    output = output_dir + '/' + ticker + '_' + date_interval + '_' + str(
                        len(chunk)) + '_' + trend + '.png'
                    save_plot(chunk, output)
                    print('Saved: ' + output)

                    offset = random.randint(2, 8)

                    local_instances = lcnt_up + lcnt_down + lcnt_side

                    if local_instances == samples:
                        break

    print('Total candle images labeled: ' + str(cnt))


if '__main__':
    main()
