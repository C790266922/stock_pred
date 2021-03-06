import time
import numpy as np
import pandas as pd
import tensorflow as tf
from utils import *
from model import *


data_path = './stock_data/feat_day.csv'
ckpt_path = './ckpt/featured_data'

df = pd.read_csv(data_path)
if 'day.csv' not in data_path:
	# 30min.csv is unlabeled
	df = add_features(df)
	df = label_df(df, col = 'close')
	df.dropna(how = 'any', inplace = True)

train_size = int(len(df) * 0.7)
header = list(df.columns[1:])
feat_list = [t for t in header if t not in ['up_or_down', 'next_close']]
label_list = ['next_close']
this_close_index = feat_list.index('close') 

train_x = df[feat_list][:train_size].values
train_y = df[label_list][:train_size].values
test_x = df[feat_list][train_size:].values
test_y = df[label_list][train_size:].values

# train_y = onehot_encoder(train_y)
# test_y = onehot_encoder(test_y)

time_steps = 5
train_x = stack_data(train_x, time_steps)
train_y = stack_data(train_y, time_steps, is_label = True)

test_x = stack_data(test_x, time_steps)
test_y = stack_data(test_y, time_steps, is_label = True)

dataset = Dataset(train_x, train_y)

# model parameters
input_size = len(feat_list)
hidden_size = 100
hidden_layers = 5
keep_prob = 0.7
output_size = len(label_list)
learning_rate = 1e-5


x = tf.placeholder(tf.float32, shape = [None, time_steps, input_size], name = 'input_x')
y = tf.placeholder(tf.float32, shape = [None, output_size], name = 'label_y')

model = LSTM(input_size,
		hidden_size,
		hidden_layers,
		output_size,
		time_steps,
		x,
		this_close_index,
		y,
		is_training = True,
		learning_rate = learning_rate,
		keep_prob = keep_prob)

init = tf.global_variables_initializer()

epochs = 1000
batch_size = 200
display_step = 500

saver = tf.train.Saver()

with tf.Session(config = tf.ConfigProto(log_device_placement = True)) as sess:
# with tf.Session() as sess:
	sess.run(init)

	ckpt = tf.train.get_checkpoint_state(ckpt_path)
	if ckpt and ckpt.model_checkpoint_path:
		saver.restore(sess, ckpt.model_checkpoint_path)
		print('model restored...')

	for epoch in range(epochs):
		for i in range(len(train_x) // batch_size):
			batch_x, batch_y = dataset.next_batch(batch_size)

			_, train_loss = sess.run([model.train_op, model.loss], feed_dict = {x: batch_x, y: batch_y})

			if i % display_step == 0:
				test_loss, acc = sess.run([model.loss, model.accuracy], feed_dict = {x: test_x, y: test_y})
				print('Epoch %d, step %d: train_loss = %f, test_loss = %f, test_accuracy = %f' % (epoch + 1, i, train_loss, test_loss, acc))
				saver.save(sess, ckpt_path + '/lstm')
