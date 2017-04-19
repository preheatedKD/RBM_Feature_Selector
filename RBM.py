import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.examples.tutorials.mnist import input_data
from utils import tile_raster_images

class RBM(object):
	def __init__(self, n_hidden = 500, n_visible = 784, gibbs_sampling_steps=10, alpha=0.01):
		self.alpha = alpha
		self.input = tf.placeholder(tf.float32, [None, n_visible])
		self.weights = tf.Variable(tf.truncated_normal([n_visible,n_hidden], stddev=1.0), name='weights_')
		self.v_bias = tf.Variable(tf.zeros([n_visible]), name='v_bias')
		self.h_bias = tf.Variable(tf.zeros([n_hidden]),	name='h_bias')

		self.params = [self.weights, self.h_bias, self.v_bias]
		self.gibbs_sampling_steps = gibbs_sampling_steps

		# self.update_w = tf.Variable(tf.truncated_normal([n_visible,n_hidden],stddev=1.0), name='updates_weights')
		# self.update_hb = tf.Variable(tf.zeros([n_hidden]), name='updates_h_bias')
		# self.update_vb = tf.Variable(tf.zeros([n_visible]), name='updates_v_bias')

		init = tf.global_variables_initializer()
		self.sess = tf.Session()
		self.sess.run(init)

	def propagate_v2h(self, vis):
		pre_sigmoid = tf.add(self.h_bias,tf.matmul(vis,self.weights))
		return [pre_sigmoid, tf.sigmoid(pre_sigmoid)]

	def propagate_h2v(self, hid):
		pre_sigmoid = tf.add(self.v_bias,tf.matmul(hid,tf.transpose(self.weights)))
		return [pre_sigmoid, tf.sigmoid(pre_sigmoid)]

	def sample_h_given_v(self, v0_sample):
		pre_sigmoid_h0, h0_probability = self.propagate_v2h(v0_sample)
		h0_sample = tf.contrib.distributions.Bernoulli(p=h0_probability).sample()
		return [pre_sigmoid_h0, h0_probability, tf.cast(h0_sample,tf.float32)]

	def sample_v_given_h(self, h0_sample):
		pre_sigmoid_v1, v1_probability = self.propagate_h2v(h0_sample)
		v1_sample = tf.contrib.distributions.Bernoulli(p=v1_probability).sample()
		return [pre_sigmoid_v1, v1_probability, tf.cast(v1_sample,tf.float32)]

	def gibbs_step(self, v0_sample):
		#Visible to Hidden
		pre_sigmoid_h0, h0_probability, h0_sample = self.sample_h_given_v(v0_sample)
		#Hidden to Visible
		pre_sigmoid_v1, v1_probability, v1_sample = self.sample_v_given_h(h0_sample)
		#Visible to Hidden again
		pre_sigmoid_h1, h1_probability, h1_sample = self.sample_h_given_v(v1_sample)

		return [h0_probability, h0_sample, 
				v1_probability, v1_sample,
				h1_probability, h1_sample]

	def gibbs_vhv(self, v0_sample):
		pre_sigmoid_h0, h0_probability, h0_sample = self.sample_h_given_v(v0_sample)
		pre_sigmoid_v1, v1_probability, v1_sample = self.sample_v_given_h(h0_sample)
		return [pre_sigmoid_h0, h0_probability, h0_sample, 
				pre_sigmoid_v1, v1_probability, v1_sample]

	def gibbs_hvh(self, h0_sample):
		pre_sigmoid_v1, v1_probability, v1_sample = self.sample_v_given_h(h0_sample)
		pre_sigmoid_h1, h1_probability, h1_sample = self.sample_h_given_v(v1_sample)
		return [pre_sigmoid_v1, v1_probability, v1_sample,
				pre_sigmoid_h1, h1_probability, h1_sample]		

	def build_model(self):
		h0_probability, h0_sample, v1_probability, v1_sample, h1_probability, h1_sample = self.gibbs_step(self.input)
		nn_input = v1_probability
		h_sample = h0_sample
		h_probability = h0_probability
		
		#Positive gradient
		self.w_positive = tf.matmul(tf.transpose(self.input), h0_sample)

		for step in range(self.gibbs_sampling_steps - 1):
			h0_probability, h0_sample, v1_probability, v1_sample,h1_probability, h1_sample = self.gibbs_step(nn_input)
			nn_input = v1_probability

		self.h_probability = h1_probability
		#Negative gradient
		self.w_negative = tf.matmul(tf.transpose(v1_sample), h1_sample)

		self.update_w = self.weights.assign_add(self.alpha * (self.w_positive - self.w_negative))#/tf.to_float(tf.shape(self.input)[0]))
		self.update_vb = self.v_bias.assign_add(self.alpha * tf.reduce_mean(self.input - v1_probability, 0))
		self.update_hb = self.h_bias.assign_add(self.alpha * tf.reduce_mean(h_probability - h1_probability,0))
		
		# with tf.variable_scope('loss'):
		self.RMSE = tf.sqrt(tf.reduce_mean(tf.square(self.input - v1_sample)))

		self.updates = [self.update_w, self.update_hb, self.update_vb]

	def train(self, train_input):
		n_w, n_hb, n_vb = self.sess.run(self.updates, feed_dict={self.input:train_input})
		ReconErr = self.sess.run(self.RMSE, feed_dict={self.input:train_input})
		return  [n_w, n_hb, n_vb, ReconErr]
	# def cost(self, batch):
	# 	return self.sess.run(self.loss_function, feed_dict={self.input:batch})
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
	new_w = np.zeros([784, 500], np.float32)
	
	for num in range(Epoch):
		for _ in range(len(mnist.train.images)/BATCH):
			batch_xs, batch_ys = mnist.train.next_batch(BATCH)
			new_w, new_hb, new_vb, ReconErr = rbm.train(batch_xs)
			# rbm.mnist_train(batch_xs, batch_ys)
			print("Epoch: {}, Iteration: {}, Reconstruction Error: {}".format(num,_,ReconErr))
		image = Image.fromarray(tile_raster_images(X=new_w.T,img_shape=(28, 28),tile_shape=(25, 20),tile_spacing=(1, 1)))
		image.save("rbm_{}.png".format(num))
		# rbm.mnist_test(mnist.test.images, mnist.test.labels)
		#Testing
		# correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
		# accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
		# print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))

		

	
