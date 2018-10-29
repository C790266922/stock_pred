import tensorflow as tf

class LSTM:

	def __init__(self,
			input_size,
			hidden_size,
			hidden_layers,
			output_size,
			time_steps,
			input_x,	# (batch_size, time_steps, input_size)
			idx,		# index of this_close, 4
			label_y,	# for regression, next_close
			is_training,
			learning_rate,
			keep_prob = 1.0):

		# input layer
		w_in = tf.Variable(tf.random_normal([input_size, hidden_size]))
		b_in = tf.Variable(tf.random_normal([hidden_size]))

		x = tf.reshape(input_x, [-1, input_size])
		x = tf.matmul(x, w_in) + b_in
		x = tf.reshape(x, [-1, time_steps, hidden_size])

		# stacked lstm
		lstm_cell = tf.contrib.rnn.BasicLSTMCell(hidden_size)
		if is_training and keep_prob < 1.0:
			lstm_cell = tf.contrib.rnn.DropoutWrapper(lstm_cell, input_keep_prob = 1.0, output_keep_prob = keep_prob)
		lstm_cells = [lstm_cell for _ in range(hidden_layers)]
		stacked_lstm = tf.contrib.rnn.MultiRNNCell(lstm_cells)

		outputs, states = tf.nn.dynamic_rnn(stacked_lstm, x, dtype = tf.float32)

		# output layer
		w_out = tf.Variable(tf.random_normal([hidden_size, output_size]))
		b_out = tf.Variable(tf.random_normal([output_size]))

		self.pred_y = tf.matmul(outputs[:, -1, :], w_out) + b_out
		tf.add_to_collection('pred', self.pred_y)

		# loss and optimizer
        # loss for regression
		self.loss = tf.reduce_mean((self.pred_y - label_y) * (self.pred_y - label_y))
		self.optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)
		self.train_op = self.optimizer.minimize(self.loss)

		# up or down accuracy
		# input_x should be rank 3
		if len(input_x.shape) != 3:
			input_x = input_x.reshape([-1, time_steps, input_size])

		self.this_close = tf.map_fn(lambda x: x[-1, idx], input_x)
		self.this_close = tf.reshape(self.this_close, [-1, 1])

		self.truth = tf.cast(label_y > self.this_close, tf.float32)
		self.pred = tf.cast(self.pred_y > self.this_close, tf.float32)
		self.accuracy = tf.reduce_mean(tf.abs(self.truth - self.pred))
		tf.add_to_collection('accuracy', self.accuracy)
