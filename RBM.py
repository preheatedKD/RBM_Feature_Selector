import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.examples.tutorials.mnist import input_data
from utils import tile_raster_images

class RBM(object):
	def __init__(self, n_hidden = 100, n_visible = 784, alpha=0.001, gibbs_sampling_steps=10, layer_names=['weight','v_bias','h_bias'], datatype="binary"):
		self.layer_names = layer_names
		self.datatype = datatype
		self.n_visible = n_visible
		self.n_hidden = n_hidden
		self.alpha = alpha

		network_weights = self._initialize_weights()
		self.weights = network_weights

		self.input = tf.placeholder(tf.float32, [None, n_visible])
		self.weights = tf.Variable(tf.truncated_normal([n_visible,n_hidden], stddev=1.0), name=self.layer_names[0])
		self.v_bias = tf.Variable(tf.zeros([n_visible], dtype=tf.float32),  name=self.layer_names[1])
		self.h_bias = tf.Variable(tf.random_uniform([n_hidden], dtype=tf.float32), name=self.layer_names[2])

		self.params = [self.weights, self.h_bias, self.v_bias]
		self.gibbs_sampling_steps = gibbs_sampling_steps

		_, h0_probability, h0_sample = self.sample_h_given_v(self.input)

		#1 Step Gibbs Sampling
		# _, v_probability = self.propagate_h2v(h0_probability)
		# _, h_probability = self.propagate_v2h(v1_probability)

		v1_probability, h1_probability = self.gibbs_sampling(h0_probability)
		self.w_positive = tf.matmul(tf.transpose(self.input), h0_sample)		
		self.w_negative = tf.matmul(tf.transpose(v1_probability), h1_probability)

		lamda = 0.001
		self.update_w = self.weights*(1 - lamda) + self.alpha * (self.w_positive - self.w_negative)#self.weights.assign_add(self.alpha * (self.w_positive - self.w_negative))#/tf.to_float(tf.shape(self.input)[0]))
		self.update_vb = self.v_bias*(1 - lamda) + self.alpha * tf.reduce_mean(self.input - v1_probability, 0)#self.v_bias.assign_add(self.alpha * tf.reduce_mean(self.input - v1_probability, 0))
		self.update_hb = self.h_bias*(1 - lamda) + self.alpha * tf.reduce_mean(h1_probability - h0_probability,0)#self.h_bias.assign_add(self.alpha * tf.reduce_mean(h1_probability - h0_probability,0))
		# with tf.variable_scope('loss'):
		_, self.h_sample_prob, _ = self.sample_h_given_v(self.input)
		_, _, self.v_sample_prob = self.sample_v_given_h(self.h_sample_prob)

		self.RMSE = tf.sqrt(tf.reduce_mean(tf.square(self.input - self.v_sample_prob))/tf.to_float(tf.shape(self.input)[0]))

		self.updates = [self.update_w, self.update_hb, self.update_vb]

		init = tf.global_variables_initializer()
		self.sess = tf.Session()
		self.sess.run(init)

	def propagate_v2h(self, vis):
		if self.datatype == "gaussian":
			vis = tf.cast(vis, tf.float32)
			
		pre_sigmoid = tf.add(self.h_bias,tf.matmul(vis,self.weights))
		return [pre_sigmoid, tf.sigmoid(pre_sigmoid)]

	def propagate_h2v(self, hid):
		pre_sigmoid = tf.add(self.v_bias,tf.matmul(hid,tf.transpose(self.weights)))
		return [pre_sigmoid, tf.sigmoid(pre_sigmoid)]

	def sample_h_given_v(self, v0_probability):
		pre_sigmoid_h0, h0_probability = self.propagate_v2h(v0_probability)
		# h0_sample = tf.contrib.distributions.Bernoulli(p=h0_probability).sample()
		h0_sample = tf.nn.relu(tf.sign(h0_probability - tf.random_uniform(tf.shape(h0_probability))))
		return [pre_sigmoid_h0, h0_probability, tf.cast(h0_sample,tf.float32)]

	def sample_v_given_h(self, h0_probability):
		pre_sigmoid_v1, v1_probability = self.propagate_h2v(h0_probability)

		if self.datatype == "gaussian":
			v1_sample = tf.contrib.distributions.Normal(mu=v1_probability, sigma=1.).sample()
		
		elif self.datatype == "binary":
			# v1_sample = tf.contrib.distributions.Bernoulli(p=v1_probability).sample()
			v1_sample = tf.nn.relu(tf.sign(v1_probability - tf.random_uniform(tf.shape(v1_probability))))		

		return [pre_sigmoid_v1, v1_probability, tf.cast(v1_sample,tf.float32)]

	def gibbs_step(self, h0_probability):
		#Hidden to Visible
		pre_sigmoid_v1, v1_probability, v1_sample = self.sample_v_given_h(h0_probability)
		#Visible to Hidden
		pre_sigmoid_h1, h1_probability, h1_sample = self.sample_h_given_v(v1_probability)
		
		return [v1_probability, v1_sample,
				h1_probability, h1_sample]

	def gibbs_sampling(self, h0_probability):
		v1_probability = 0.0
		h1_probability = 0.0

		for step in range(self.gibbs_sampling_steps - 1):
			v1_probability, v1_sample, h1_probability, h1_sample = self.gibbs_step(h0_probability)
			h0_probability = h1_probability

		return [v1_probability, h1_probability]

	def inference(self, vis):
		return self.sess.run(self.v_sample_prob, feed_dict={self.input:vis})

	def passThrough(self, vis):
		return self.sess.run(self.h_sample_prob, feed_dict={self.input:vis})

	def passBack(self, hid):
		return self.sess.run(self.v_sample_prob, feed_dict={self.h_sample_prob:hid})

	def build_model(self):
		_, h0_probability, h0_sample = self.sample_h_given_v(self.input)
		
		#1 Step Gibbs Sampling
		# _, v_probability = self.propagate_h2v(h0_probability)
		# _, h_probability = self.propagate_v2h(v1_probability)

		v1_probability, h1_probability = self.gibbs_sampling(h0_probability)

		self.w_positive = tf.matmul(tf.transpose(self.input), h0_sample)		
		self.w_negative = tf.matmul(tf.transpose(v1_probability), h1_probability)

		self.update_w = self.weights.assign_add(self.alpha * (self.w_positive - self.w_negative))#/tf.to_float(tf.shape(self.input)[0]))
		self.update_vb = self.v_bias.assign_add(self.alpha * tf.reduce_mean(self.input - v1_probability, 0))
		self.update_hb = self.h_bias.assign_add(self.alpha * tf.reduce_mean(h1_probability - h0_probability,0))
		# with tf.variable_scope('loss'):
		_, h_sample_prob, _ = self.sample_h_given_v(self.input)
		_, _, v_sample_prob = self.sample_v_given_h(h_sample_prob)

		self.RMSE = tf.sqrt(tf.reduce_mean(tf.square(self.input - v_sample_prob))/tf.to_float(tf.shape(self.input)[0]))

		self.updates = [self.update_w, self.update_hb, self.update_vb]

	def train(self, train_input):
		n_w, n_hb, n_vb = self.sess.run(self.updates, feed_dict={self.input:train_input})
		ReconErr = self.sess.run(self.RMSE, feed_dict={self.input:train_input})
		return  [n_w, n_hb, n_vb, ReconErr]

	def _initialize_weights(self):
		# These weights are only for storing and loading model for tensorflow Saver.
		all_weights = dict()
		all_weights['w'] = tf.Variable(tf.random_normal([self.n_visible, self.n_hidden], stddev=0.01, dtype=tf.float32),
									   name=self.layer_names[0])
		all_weights['vb'] = tf.Variable(tf.zeros([self.n_visible], dtype=tf.float32), name=self.layer_names[1])
		all_weights['hb'] = tf.Variable(tf.random_uniform([self.n_hidden], dtype=tf.float32), name=self.layer_names[2])
		return all_weights

	def save_weights(self, path):
		self.sess.run(self.weights['w'].assign(self.weights))
		self.sess.run(self.weights['vb'].assign(self.v_bias))
		self.sess.run(self.weights['hb'].assign(self.h_bias))
		saver = tf.train.Saver({self.layer_names[0]: self.weights['w'],
								self.layer_names[1]: self.weights['vb'],
								self.layer_names[2]: self.weights['hb']})
		save_path = saver.save(self.sess, path)

	def restore_weights(self, path):
		saver = tf.train.Saver({self.layer_names[0]: self.weights['w'],
								self.layer_names[1]: self.weights['vb'],
								self.layer_names[2]: self.weights['hb']})

		saver.restore(self.sess, path)

		self.weights = self.weights['w'].eval(self.sess)
		self.v_bias = self.weights['vb'].eval(self.sess)
		self.h_bias = self.weights['hb'].eval(self.sess)

	def mnist(self, n_hidden=500):
		self.W = tf.Variable(tf.truncated_normal([n_hidden,10], stddev=1.0), name='weights_mnist')
		self.b = tf.Variable(tf.zeros([10]))

		self.y_ = tf.placeholder(tf.float32, [None, 10])
		
		self.build_model()
		
		init = tf.global_variables_initializer()
		self.sess.run(init)

		y = tf.nn.softmax(tf.matmul(self.h_probability, self.W) + self.b)
		self.mnist_cross_entropy = tf.reduce_mean(-tf.reduce_sum(self.y_ * tf.log(y), reduction_indices=[1]))

		self.train_step = tf.train.GradientDescentOptimizer(0.5).minimize(self.mnist_cross_entropy)

	
	def mnist_train(self, batch_xs, batch_ys):
		self.sess.run(self.train_step, feed_dict={self.input: batch_xs,self.y_: batch_ys})

	def mnist_test(self, batch_xs, batch_ys):
		_,probability,_ = self.sample_h_given_v(batch_xs)
		y = tf.nn.softmax(tf.matmul(self.h_probability, self.W) + self.b)		
		correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(self.y_, 1))
		self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
		print(sess.run(self.accuracy, feed_dict={self.input: batch_xs, self.y_: batch_ys}))

if __name__ == '__main__':
	BATCH = 100
	Epoch = 20

	mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
	sess = tf.Session()
	rbm = RBM()
	rbm.build_model()
	# rbm.mnist()
	new_w = np.zeros([784, 100], np.float32)
	
	for num in range(Epoch):
		for _ in range(len(mnist.train.images)/BATCH):
			batch_xs, batch_ys = mnist.train.next_batch(BATCH)
			new_w, new_hb, new_vb, ReconErr = rbm.train(batch_xs)
			# rbm.mnist_train(batch_xs, batch_ys)
			print("Epoch: {}, Iteration: {}, Reconstruction Error: {}".format(num,_,ReconErr))
		image = Image.fromarray(tile_raster_images(X=new_w.T,img_shape=(28, 28),tile_shape=(10, 10),tile_spacing=(1, 1)))
		image.save("rbm_{}.png".format(num))
		# rbm.mnist_test(mnist.test.images, mnist.test.labels)
		#Testing
		# correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
		# accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
		# print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))

		

	
