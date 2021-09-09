import pandas as pd
pd.options.mode.chained_assignment = None
import matplotlib.pyplot as plt


if __name__ == '__main__':
    raw_data = pd.read_csv("../data/bitstampUSD_1-min_data_2012-01-01_to_2021-03-31.csv")
    raw_data.head()
    raw_data.tail()
    df = raw_data.dropna()
    df.reset_index(inplace=True, drop=True)
    print(df.head())

    df.loc[:, 'Timestamp'] = pd.to_datetime(df.loc[:, 'Timestamp'], unit='s')
    print(df.head())

    df.drop([0, 1, 2, 3], inplace=True)
    df.reset_index(inplace=True, drop=True)
    print(df.head())
    print(df.shape)
    #
    df.plot(x="Timestamp", y="Weighted_Price")
    plt.show()
