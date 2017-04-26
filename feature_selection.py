from RBM import RBM
import tensorflow as tf
import scipy.io as sio
from scipy.stats import rankdata
import numpy as np

Epoch = 100

dataset = sio.loadmat("glioma1.mat")
MA_data = dataset['data']
MA_label = dataset['ylab']

#Normalize for unit variance and zero mean
MA_data = (MA_data - np.mean(MA_data))/np.std(MA_data)

n_items = np.shape(MA_data)[0]
n_feature = np.shape(MA_data)[1]
percentage = 0.5
n_hidden1 = int(percentage * n_feature)
n_hidden2 = int(percentage * n_hidden1)

with tf.device("/gpu:0"):
	rbm1 = RBM(n_hidden=n_hidden1, n_visible=n_feature, alpha=0.0001, datatype="gaussian")
	rbm2 = RBM(n_hidden=n_hidden2, n_visible=n_hidden1, alpha=0.0001, datatype="binary")

	for num in range(Epoch):
		new_w, new_hb, new_vb, ReconErr = rbm1.train(MA_data)
		print("Epoch: {}, Reconstruction Error: {}".format(num,ReconErr))

	for num in range(Epoch):
		batch_xs = rbm1.passThrough(MA_data)
		new_w, new_hb, new_vb, ReconErr = rbm2.train(batch_xs)
		print("Epoch: {}, Reconstruction Error: {}".format(num,ReconErr))

	temp1 = rbm1.passThrough(MA_data)
	temp2 = rbm2.inference(temp1)
	REC = rb1.passBack(temp2)

	print(np.shape(REC))
	RMSE = np.sqrt(np.mean(np.square(MA_data - REC), axis=0)/float(np.shape(MA_data)[0]))
	print("Shape of RMSE: {}, RMSE {}".format(np.shape(RMSE), RMSE))
	RMSE = np.sort(RMSE)
	for _ in range(n_feature):
		print("Feature {}, ReconError:{}".format(_, RMSE[_]))



	# saver = tf.train.Saver()
	# save_path = saver.save(sess, "/tmp/model.ckpt")

	# with tf.Session() as sess:
	# 	saver.restore(sess, "/tmp/model.ckpt")
	# 	pre_sigmoid_h1, h1_probability, h1_sample = rbm.sample_h_given_v(batch_xs)
	# 	pre_sigmoid_v1, v1_probability, v1_sample = rbm.sample_v_given_h(h1_probability)
	# 	print("Shape of v1_sample is {}".format(np.shape(v1_sample)))
	# 	RMSE = sess.run(tf.sqrt(tf.reduce_mean(tf.square(MA_data - v1_sample),0)/tf.to_float(tf.shape(MA_data)[0])))
	# 	print("Shape of RMSE: {}, RMSE {}".format(np.shape(RMSE), RMSE))
	# 	for _ in range(n_feature):
	# 		print("Feature {}, ReconError:{}".format(_, RMSE[_]))


