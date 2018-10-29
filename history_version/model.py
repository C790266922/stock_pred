import tensorflow as tf

class LSTM:

	def __init__(self,
			input_size,
			hidden_size,
			hidden_layers,
			output_size,
			time_steps,
			input_x,	# (batch_size, time_steps, input_size)
			label_y,	# (batch_size, output_size)
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

		self.logits = tf.matmul(outputs[:, -1, :], w_out) + b_out

		self.pred = tf.nn.softmax(self.logits)
		
		# loss and optimizer
		self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits = self.pred, labels = label_y))
		self.optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)
		self.train_op = self.optimizer.minimize(self.loss)

		self.correct_pred = tf.equal(tf.argmax(self.pred, 1), tf.argmax(label_y, 1))
		self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32))
		tf.add_to_collection('accuracy', self.accuracy)
