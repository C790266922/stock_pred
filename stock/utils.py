import os
import time
import pandas as pd
import numpy as np
import talib as tb
import pywt


def load_data(path):
    '''
    load all data in the given path,
    if more than half rows volume == 0, ignore it
    date, time, open, high, low, close, volume, amount
    '''

    files = sorted(os.listdir(path))
    df = pd.read_csv(path + '/' + files[0], header = None)
    df = norm(df)
    df = label_df(df)

    print('num of files: ' + str(len(files)))
    dropped = 0
    for i in range(1, len(files)):
        temp = pd.read_csv(path + '/' + files[i], header = None)
        # count nonzero volume
        nonzero_vol = temp.iloc[:, 6].astype(bool).sum()
        if nonzero_vol > len(temp) / 2:
            temp = norm(temp)
            temp = label_df(temp)
            df = pd.concat([df, temp], ignore_index = True)
        else:
            dropped += 1
        if i % 100 == 0:
            print('progress: %d/%d' % (i, len(files)))

    print(str(dropped) + ' files dropped')
	# drop inf and nan
    df.replace([np.inf, -np.inf], np.nan)
    df.dropna(how = 'any', inplace = True)

    df.reset_index(inplace = True)
    return df


def norm(df, cols = ['open', 'high', 'low', 'close', 'volume', 'amount']):
    '''
    normalize dataframe
    '''
    for i in cols:
        min_val = df.iloc[:, i].min()
        max_val = df.iloc[:, i].max()
        range_ = max_val - min_val
        df[i] = (df[i] - min_val) / range_

    return df

def label_df(df, col = 'close'):
    '''
    add 2 label columns
    for classification, price_up:1, price_no_change:0, price_down:1
    for regression, each row's label is the next row's close price
    '''
    up_or_down = []
    next_close = []
    for i in range(len(df)):
        if i == len(df) - 1:
            up_or_down.append(0)
            next_close.append(df.iloc[i][col])
        else:
            if df.iloc[i + 1][col] > df.iloc[i][col]:
                up_or_down.append(1)
            elif df.iloc[i + 1][col] < df.iloc[i][col]:
                up_or_down.append(-1)
            else:
                up_or_down.append(0)

            next_close.append(df.iloc[i + 1][col])

    df.insert(len(df.columns), 'up_or_down', np.array(up_or_down))
    df.insert(len(df.columns), 'next_close', np.array(next_close))
    return df

def onehot_encoder(label):
	'''
	one-hot encoder for label column
	'''
	ret = np.zeros([len(label), 3])
	for i in range(len(label)):
		if label[i] == 0:
			ret[i, 1] = 1
		elif label[i] == -1:
			ret[i, 0] = 1
		else:
			ret[i, 2] = 1
	return ret

def stack_data(arr, time_steps, is_label = False):
	'''
	stack data into shape (time_steps, feature_length)
	'''
	ret = []
	if is_label:
		return arr[time_steps:]

	for i in range(len(arr) -  time_steps):
		ret.append(arr[i : i + time_steps])

	return np.array(ret)

class Dataset:
	'''
	get next batch
	'''
	def __init__(self, x, y, shuffle = True):
		self._index_in_epoch = 0
		self._x = x
		self._y = y
		self._size = len(x)
		self._perm_arr = np.arange(self._size)
		self._shuffle = shuffle
		if shuffle:
			np.random.shuffle(self._perm_arr)

	def next_batch(self, batch_size):
		start = self._index_in_epoch
		end = start + batch_size
		self._index_in_epoch = end

		if end > self._size:
			start = 0
			end = batch_size
			self._index_in_epoch = end
			if self._shuffle:
				np.random.shuffle(self._perm_arr)

		batch_x = self._x[start : end]
		batch_y = self._y[start : end]
		return batch_x, batch_y

def add_features(df):
    # add MACD, CCI, ATR, BOLL, EMA20, MA5, MA10, MOM6, MOM12, ROC, RSI, WR, KDJ
    open_ = df['open'].values
    high = df['high'].values
    low = df['low'].values
    close = df['close'].values

    macd, macdsignal, macdhist = tb.MACD(close, fastperiod = 12, slowperiod = 26, signalperiod = 9)
    cci = tb.CCI(high, low, close, timeperiod = 14)
    atr = tb.ATR(high, low, close, timeperiod = 14)
    boll_up, boll_mid, boll_low = tb.BBANDS(close, timeperiod = 5, nbdevup = 2, nbdevdn = 2, matype = 0)
    ema20 = tb.EMA(close, timeperiod = 20)
    ma5 = tb.SMA(close, timeperiod = 5)
    ma10 = tb.SMA(close, timeperiod = 10)
    mom6 = tb.MOM(close, timeperiod = 6)
    mom12 = tb.MOM(close, timeperiod = 12)
    roc = tb.ROC(close, timeperiod = 14)
    rsi_k, rsi_d = tb.STOCHRSI(close, timeperiod = 14, fastk_period = 5, fastd_period = 3, fastd_matype = 0)
    wr = tb.WILLR(high, low, close, timeperiod = 14)
    kdj_j = 3 * rsi_k - 2 * rsi_d
	
    feat_list = [macd, macdsignal, macdhist, cci, atr, boll_up, boll_mid, boll_low, ema20, ma5, ma10, mom6, mom12, roc, rsi_k, rsi_d, wr, kdj_j]
	
    handle_nan(feat_list)

    df.insert(len(df.columns), 'macd', macd)
    df.insert(len(df.columns), 'macdsignal', macdsignal)
    df.insert(len(df.columns), 'macdhist', macdhist)
    df.insert(len(df.columns), 'cci', cci)
    df.insert(len(df.columns), 'atr', atr)
    df.insert(len(df.columns), 'boll_up', boll_up)
    df.insert(len(df.columns), 'boll_mid', boll_mid)
    df.insert(len(df.columns), 'boll_low', boll_low)
    df.insert(len(df.columns), 'ema20', ema20)
    df.insert(len(df.columns), 'ma5', ma5)
    df.insert(len(df.columns), 'ma10', ma10)
    df.insert(len(df.columns), 'mom6', mom6)
    df.insert(len(df.columns), 'mom12', mom12)
    df.insert(len(df.columns), 'roc', roc)
    df.insert(len(df.columns), 'rsi_k', rsi_k)
    df.insert(len(df.columns), 'rsi_d', rsi_d)
    df.insert(len(df.columns), 'wr', wr)
    df.insert(len(df.columns), 'kdj_j', kdj_j)

    return df

def handle_nan(feat_list):
    for arr in feat_list:
        arr[np.isnan(arr)] = 0

def dwt_decompose(df, level = 5, columns = ['close']):
	for col in columns:
		# coefs: (level + 1, len(df))
		coefs = pywt.wavedec(df[col].values, 'db4', level = level)
		for i in range(len(coefs)):
			if i == 0:
				df.insert(len(df.columns), 'dwt_a1', coefs[0])
			else:
				df.insert(len(df.columns), 'dwt_d' + str(i), coefs[i])

	return df


if __name__ == '__main__':

    '''
    load data and store them in one file
    so next time we can just use pd.read_csv to load the 1 file
    this can save us a lot of IO time
    '''
    feat_list = ['close', 'macd', 'macdsignal', 'macdhist', 'cci', 'atr', 'boll_up', 'boll_mid', 'boll_low', 'ema20', 'ma5', 'ma10', 'mom6', 'mom12', 'roc', 'rsi_k', 'rsi_d', 'wr', 'kdj_j']
	
    if os.path.exists('./stock_data/day.csv'):
        day_df = pd.read_csv('./stock_data/day.csv')
    else:
        day_df = load_data('./stock_data/day')
    day_df = add_features(day_df)
    # day_df = dwt_decompose(day_df, level = 5, columns = feat_list)
    day_df.to_csv('./stock_data/feat_day.csv', index = False)
    print('1 day data processed')
    exit()

    if os.path.exist('./stock_data/min.csv'):
        min_df = pd.read_csv('./stock_data/min.csv')
    else:
        min_df = load_data('./stock_data/min')
    min_df = add_features(min_df)
    min_df.to_csv('./stock_data/min.csv', index = False)
    print('1 min data processed')
