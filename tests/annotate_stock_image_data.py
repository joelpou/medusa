import os
import sys
import glob
import yfinance as yf


input_dir = str(sys.argv[1])  # input dir where candlestick images are stored
output_dir = str(sys.argv[2])  # output dir to save csv annotations file
os.makedirs(output_dir, exist_ok=True)
# s&p500_candle_annotations.csv

for f in glob.glob(os.path.join(input_dir, "**/*.png"), recursive=True):
    print(f)
    fsplit = f.rsplit("/")
    name = fsplit[-1]
    fsplit = name.rsplit("_")
    ticker = fsplit[0]
    date_from = fsplit[1]
    date_to = fsplit[2]
    print(ticker)
    #print(date_from)
    ticker_data = yf.Ticker(ticker)
    ticker_df = ticker_data.history(period='1d', start=date_from, end=date_to)
    print(len(ticker_df))
    print(ticker_df)