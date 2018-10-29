import numpy as np
import pandas as pd
import tensorflow as tf
import time
from tensorflow.python import pywrap_tensorflow
from tensorflow.python.tools import inspect_checkpoint
from utils import *

file_path = './stock_data/30min.csv'
ckpt_path = './ckpt/featured_data'

test_df = pd.read_csv(file_path)
if '30min.csv' in file_path:
	test_df = add_features(test_df)
	test_df = label_df(test_df, col = 'close')

header = list(test_df.columns[1:])
feat_list = [t for t in header if t not in ['up_or_down', 'next_close']]
label_list = ['next_close']

test_x = test_df[feat_list].values
test_y = test_df[label_list].values
# test_y = onehot_encoder(test_y)

time_steps = 5
test_x = stack_data(test_x, 5)
test_y = stack_data(test_y, 5, is_label = True)

input_size = len(feat_list)
output_size = len(label_list) 

ckpt = tf.train.get_checkpoint_state(ckpt_path)
saver = tf.train.import_meta_graph(ckpt.model_checkpoint_path + '.meta')

reader = pywrap_tensorflow.NewCheckpointReader(ckpt.model_checkpoint_path)
var_to_shape_map = reader.get_variable_to_shape_map()
# print tensor names
for key in var_to_shape_map:
	print("tensor_name: ", key)
	print(reader.get_tensor(key))

with tf.Session() as sess:
	if ckpt and ckpt.model_checkpoint_path:
		saver.restore(sess, ckpt.model_checkpoint_path)
		print('model restored...')
	else:
		print('model not found')

	i = 1
	accuracy = tf.get_collection('accuracy')[0]
	# accuracy = tf.get_default_graph().get_operation_by_name('accuracy').outputs[0]
	x = tf.get_default_graph().get_operation_by_name('input_x').outputs[0]
	y = tf.get_default_graph().get_operation_by_name('label_y').outputs[0]

	overall_acc = 0
	count = 0
	try:
		for a, b in zip(test_x, test_y):
			acc = sess.run(accuracy, feed_dict = {x: a.reshape([-1, time_steps, input_size]), y: b.reshape([-1, output_size])})
			overall_acc += acc
			count += 1
			if i % 5000 == 0:
				print('progress: %d/%d, accuracy: %f' % (i, len(test_x), overall_acc / count))

			i += 1
	except(KeyboardInterrupt):
		print('total test case: %d' % count)
		print('overall accuracy: %f' % (overall_acc / count))

	print('total test case: %d' % count)
	print('overall accuracy: %f' % (overall_acc / count))
