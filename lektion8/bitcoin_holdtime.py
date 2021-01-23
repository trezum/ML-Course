import pandas as pd
import os

# The analysis could be more of a worst case view if High -> Low was used instead of Close -> Close

# Dataset source:
# https://www.kaggle.com/mczielinski/bitcoin-historical-data

# use your path
path = r'C:\Datasets\BitcoinPrice\bitstampUSD_1-min_data_2012-01-01_to_2020-12-31.csv'
cleaned_path = r'C:\Datasets\BitcoinPrice\bitstampUSD_cleaned.csv'
resampled_path = r'C:\Datasets\BitcoinPrice\bitstampUSD_resampled.csv'

# Removing some collumns to reduce the amount of data
if not os.path.isfile(cleaned_path):
    df = pd.read_csv(path)
    df.drop('Weighted_Price', axis=1, inplace=True)
    df.drop('Open', axis=1, inplace=True)
    df.drop('Volume_(Currency)', axis=1, inplace=True)
    df.drop('Volume_(BTC)', axis=1, inplace=True)
    df.drop('High', axis=1, inplace=True)
    df.drop('Low', axis=1, inplace=True)
    df = df.dropna(axis='rows')
    df.to_csv(cleaned_path)

else:
    df = pd.read_csv(cleaned_path)

# Downsampling, sacreficing some resolution to make the calculation faster.
if not os.path.isfile(resampled_path):
    print(df.head())
    print(df.info())

    time_col_name = 'Timestamp'

    # convert the column (it's a unix timestamp) to datetime type
    datetime_series = pd.to_datetime(df[time_col_name], unit='s', origin='unix')

    datetime_index = pd.DatetimeIndex(datetime_series.values)
    df2 = df.set_index(datetime_index)

    # we don't need the column anymore
    df2.drop(time_col_name, axis=1, inplace=True)

    print(df2.index)
    print(df2.head())
    print(df2.info())

    df3 = df2.resample('1D').max()

    print(df3.info())
    print(df3.head())

    df3.to_csv(resampled_path)
else:
    df3 = pd.read_csv(resampled_path)

found = False

for i in range(1200, len(df3.index)):
    if not found:
        # Move the window along the rows
        for index, row in df3.iterrows():
            if (index + i) <= len(df3.index) - 1:
                first = df3['Close'][index]
                last = df3['Close'][index + i]

                if first > last:
                    print(str(i) + ' days is not enough ' + str(index) + '-' + str(index + i) + ' : ' + str(first) + '-' + str(last))
                    # Break if we find a window where the result is lower.
                    break

                # break if we get to the end of the data, this means we have found a window without loss.
                if index + i == len(df3.index) - 1:
                    found = True
                    break
    else:
        # Print the required hold window
        print(str(i-1) + ' days is enough!')
        break


