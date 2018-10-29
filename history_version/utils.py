import os
import time
import pandas as pd
import numpy as np


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


def norm(df, cols = [2, 3, 4, 5, 6, 7]):
    '''
    normalize dataframe
    '''
    for i in cols:
        min_val = df.iloc[:, i].min()
        max_val = df.iloc[:, i].max()
        range_ = max_val - min_val
        df[i] = (df[i] - min_val) / range_

    return df

def label_df(df, col = 5):
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
            next_close.append(df.iloc[i, col])
        else:
            if df.iloc[i + 1, col] > df.iloc[i, col]:
                up_or_down.append(1)
            elif df.iloc[i + 1, col] < df.iloc[i, col]:
                up_or_down.append(-1)
            else:
                up_or_down.append(0)

            next_close.append(df.iloc[i + 1, col])
	
    df.insert(len(df.columns), len(df.columns), np.array(up_or_down))
    df.insert(len(df.columns), len(df.columns), np.array(next_close))
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


if __name__ == '__main__':
    
    '''
    load data and store them in one file
    so next time we can just use pd.read_csv to load the 1 file
    this can save us a lot of IO time
    '''
    day_df = load_data('./stock_data/day')
    day_df.to_csv('./stock_data/day.csv', index = False, header = False)
    print('1 day data processed')

    min_df = load_data('./stock_data/min')
    min_df.to_csv('./stock_data/min.csv', index = False, header = False)
    print('1 min data processed')
