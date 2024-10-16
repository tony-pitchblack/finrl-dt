# Imports
import pandas as pd
import matplotlib.pyplot as plt
from finrl.meta.preprocessor.yahoodownloader import YahooDownloader

TRAIN_START_DATE = '2009-01-01'
TRAIN_END_DATE = '2020-07-01'
TRADE_START_DATE = '2020-07-01'
TRADE_END_DATE = '2021-10-29'

# Part 4: DJIA index for training period
df_dji_train = YahooDownloader(start_date=TRAIN_START_DATE,
                               end_date=TRAIN_END_DATE,
                               ticker_list=['dji']).fetch_data()

df_dji_train = df_dji_train[['date', 'close']]
train_fst_day = df_dji_train['close'].iloc[0]
dji_train = pd.merge(df_dji_train['date'], 
                     df_dji_train['close'].div(train_fst_day).mul(1000000), 
                     how='outer', left_index=True, right_index=True).set_index('date')

# Part 4: DJIA index for trading period
df_dji_trade = YahooDownloader(start_date=TRADE_START_DATE,
                               end_date=TRADE_END_DATE,
                               ticker_list=['dji']).fetch_data()

df_dji_trade = df_dji_trade[['date', 'close']]
trade_fst_day = df_dji_trade['close'].iloc[0]
dji_trade = pd.merge(df_dji_trade['date'], 
                     df_dji_trade['close'].div(trade_fst_day).mul(1000000), 
                     how='outer', left_index=True, right_index=True).set_index('date')

# # When merging results, you can now use either dji_train or dji_trade as needed
# result_train = pd.merge(result_train, dji_train, how='outer', left_index=True, right_index=True).fillna(method='bfill')
# result_trade = pd.merge(result_trade, dji_trade, how='outer', left_index=True, right_index=True).fillna(method='bfill')

# When plotting, you can now plot training and trading results separately
plt.figure(figsize=(12, 6))
df_dji_trade.plot(title='Training Period Performance')
plt.figure(figsize=(12, 6))
df_dji_train.plot(title='Trading Period Performance')

plt.show()