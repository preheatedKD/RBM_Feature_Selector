from RBM import RBM
import tensorflow as tf
import scipy.io as sio
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
n_hidden = tf.to_int32(percentage * n_feature)
rbm = RBM(n_hidden=n_hidden, n_visible=n_feature, alpha=0.0001, datatype="gaussian")
rbm.build_model()

for num in range(Epoch):
	batch_xs = MA_data
	new_w, new_hb, new_vb, ReconErr = rbm.train(batch_xs)
	print("Epoch: {}, Reconstruction Error: {}".format(num,ReconErr))

pre_sigmoid_h1, h1_probability, h1_sample = rbm.sample_h_given_v(batch_xs)
pre_sigmoid_v1, v1_probability, v1_sample = rbm.sample_v_given_h(h1_probability)

print("Shape of v1_sample is {}".format(np.shape(v1_sample)))
RMSE = tf.sqrt(tf.reduce_mean(tf.square(MA_data - v1_sample),0)/tf.to_float(tf.shape(MA_data)[0]))
print("Shape of RMSE: {}".format(RMSE))
for _ in range(n_feature):
	print("Feature {}, ReconError:{}".format(_, RMSE[_]))


