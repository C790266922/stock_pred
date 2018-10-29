import numpy as np
import pandas as pd
import tensorflow as tf
from utils import *
import time


df = pd.read_csv('./stock_data/30min.csv', header = None)
train_size = int(len(df) * 0.7)

# feat_list = [2, 3, 4, 5, 6, 7]
# label_list = [8]
feat_list = [1, 2, 3, 4, 5, 6]
label_list = [7]

df = label_df(df, col = 4)

train_x = df[feat_list][:train_size].values
train_y = df[label_list][:train_size].values
test_x = df[feat_list][train_size:].values
test_y = df[label_list][train_size:].values

train_y = onehot_encoder(train_y)
test_y = onehot_encoder(test_y)

time_steps = 5
train_x = stack_data(train_x, time_steps)
train_y = stack_data(train_y, time_steps, is_label = True)

test_x = stack_data(test_x, time_steps)
test_y = stack_data(test_y, time_steps, is_label = True)

index_in_epoch = 0
perm_arr = np.arange(len(train_x))
np.random.shuffle(perm_arr)
def next_batch(x, y, batch_size):
	global index_in_epoch, perm_arr
	start = index_in_epoch
	end = index_in_epoch + batch_size
	index_in_epoch = end

	if end > len(train_x):
		np.random.shuffle(perm_arr)
		start = 0
		end = batch_size
		index_in_epoch = batch_size
	
	batch_x = x[perm_arr[start : end]]
	batch_y = y[perm_arr[start : end]]
	return batch_x, batch_y


input_size = 6
hidden_size = 100
hidden_layers = 5
keep_prob = 1.0
output_size = 3

lr = 1e-3

W = {
		'in': tf.Variable(tf.random_normal([input_size, hidden_size]), trainable = True),
		'out': tf.Variable(tf.random_normal([hidden_size, output_size]), trainable = True) 
}

b = {
		'in': tf.Variable(tf.random_normal([hidden_size]), trainable = True),
		'out': tf.Variable(tf.random_normal([output_size]), trainable = True)
}

x = tf.placeholder(tf.float32, shape = [None, input_size], name = 'input_x')
y = tf.placeholder(tf.float32, shape = [None, output_size], name = 'input_y')

# input layer
x_in = tf.matmul(x, W['in']) + b['in']
x_in = tf.reshape(x_in, [-1, time_steps, hidden_size])

# lstm cell
lstm_cell = tf.contrib.rnn.BasicLSTMCell(hidden_size)
lstm_cell = tf.contrib.rnn.DropoutWrapper(lstm_cell, input_keep_prob = 1.0, output_keep_prob = keep_prob)
stacked_lstm = [lstm_cell for _ in range(hidden_layers)]
stacked_lstm = tf.contrib.rnn.MultiRNNCell(stacked_lstm)
outputs, states = tf.nn.dynamic_rnn(stacked_lstm, x_in, dtype = tf.float32)

# output layer
logits = tf.matmul(outputs[:, -1, :], W['out']) + b['out']
pred = tf.nn.softmax(logits)

# loss and optimizer
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = pred, labels = y))
optimizer = tf.train.AdamOptimizer(learning_rate = lr)
train_op = optimizer.minimize(loss)

correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
init = tf.global_variables_initializer()

epochs = 1000
batch_size = 10
display_step = 50

saver = tf.train.Saver()
tf.add_to_collection('pred', pred)
tf.add_to_collection('loss', loss)
tf.add_to_collection('correct_pred', correct_pred)
tf.add_to_collection('accuracy', accuracy)


with tf.Session(config = tf.ConfigProto(log_device_placement = True)) as sess:
	sess.run(init)

   #  ckpt = tf.train.get_checkpoint_state('./ckpt')
	# if ckpt and ckpt.model_checkpoint_path:
		# saver.restore(sess, ckpt.model_checkpoint_path)
		# print('model restored...')

	for epoch in range(epochs):
		for i in range(len(train_x) // batch_size):
			batch_x, batch_y = next_batch(train_x, train_y, batch_size)
			batch_x = np.reshape(batch_x, [-1, input_size])

			_, train_loss = sess.run([train_op, loss], feed_dict = {x: batch_x, y: batch_y})

			if i % display_step == 0:
				test_loss, acc = sess.run([loss, accuracy], feed_dict = {x: test_x.reshape([-1, input_size]), y: test_y})
				print('Epoch %d, step %d: train_loss = %f, test_loss = %f, test_accuracy = %f' % (epoch + 1, i + 1, train_loss, test_loss, acc))
				saver.save(sess, './ckpt2/lstm')
