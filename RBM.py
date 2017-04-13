import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

class RBM(object):
	def __init__(self, n_hidden = 20, n_visible = 784, gibbs_sampling_steps=5):
		self.input = tf.placeholder(tf.float32, [None, n_visible])
		self.weights = tf.Variable(tf.truncated_normal([n_visible,n_hidden], stddev=1.0), name='weights')
		self.v_bias = tf.Variable(tf.zeros([n_visible]), name='v_bias')
		self.h_bias = tf.Variable(tf.zeros([n_hidden]),	name='h_bias')

		self.params = [self.weights, self.h_bias, self.v_bias]
		self.gibbs_sampling_steps = gibbs_sampling_steps

		self.update_w = tf.Variable(tf.truncated_normal([n_visible,n_hidden],stddev=1.0), name='updates_weights')
		self.update_hb = tf.Variable(tf.zeros([n_hidden]), name='updates_h_bias')
		self.update_vb = tf.Variable(tf.zeros([n_visible]), name='updates_v_bias')

		# self.new_w = np.zeros([n_visible, n_hidden], np.float32)
		# self.update_hb = np.zeros([n_hidden], np.float32)
		# self.update_vb = np.zeros([n_visible], np.float32)

		init = tf.global_variables_initializer()
		self.sess = tf.Session()
		self.sess.run(init)

	def propagate_v2h(self, vis):
		pre_sigmoid = tf.add(self.v_bias,tf.matmul(vis,self.weights))
		return [pre_sigmoid, tf.sigmoid(pre_sigmoid)]

	def propagate_h2v(self, hid):
		pre_sigmoid = tf.add(self.h_bias,tf.matmul(hid,self.weights))
		return [pre_sigmoid, tf.sigmoid(pre_sigmoid)]

	def sample_h_given_v(self, v0_sample):
		pre_sigmoid_h0, h0_probability = self.propagate_v2h(v0_sample)
		h0_sample = tf.contrib.distributions.Bernoulli(p=h0_probability)
		return [pre_sigmoid_h0, h0_probability, h0_sample]

	def sample_v_given_h(self, h0_sample):
		pre_sigmoid_v1, v1_probability = self.propagate_h2v(h0_sample)
		v1_sample = tf.contrib.distributions.Bernoulli(p=v1_probability)
		return [pre_sigmoid_v1, v1_probability, v1_sample]

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

	def build_model(self, alpha=0.1):
		h0_probability, h0_sample, 
		v1_probability, v1_sample,
		h1_probability, h1_sample = gibbs_step(self.input)
		
		nn_input = v1_probability

		#Positive gradient
		self.w_positive = tf.matmul(tf.transpose(self.input), h0_sample)

		for step in range(self.gibbs_sampling_steps - 1):
			h0_probability, h0_sample, v1_probability, v1_sample,h1_probability, h1_sample = self.gibbs_step(nn_input)
			nn_input = v1_probability

		#Negative gradient
		self.w_negative = tf.matmul(tf.transpose(v1_probability), h1_probability)

		self.update_w = self.weights.assign_add(alpha * (self.w_positive - self.w_negative))
		self.update_vb = self.v_bias.assign_add(alpha * tf.reduce_mean(self.input - v1_probability, 0))
		self.update_hb = self.h_bias.assign_add(alpha * tf.reduce_mean(h0_probability - h1_probability,0))

		self.loss_function = tf.sqrt(tf.reduce_mean(tf.square(self.input - v1_probability)))
		_ = tf.scalar_summary("cost", self.loss_function)

	def train(self, train_input):
		updates = [self.update_w, self.update_hb, self.update_vb]
		# self.new_w, self.new_hb, self.new_vb = 
		self.sess.run(updates, feed_dict={self.input:train_input})

if __name__ == '__main__':
	mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
	#Models
	sess = tf.Session()
	rbm = RBM()
	for _ in range(100):
		batch_xs, batch_ys = mnist.train.next_batch(100)
		rbm.train(batch_xs)
		

	