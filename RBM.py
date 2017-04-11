import tensorflow as tf

BATCH_SIZE = 128
n_hidden = 150
n_visible = 784

class RBM():
	def __init__(self, n):
		self.weights = tf.Variable(tf.truncated_normal(
									[n_visible,n_hidden],
									stddev=1.0),
							name='weights')
		
		self.v_bias = tf.Variable(tf.zeros([1]),
							name='v_bias')

		self.h_bias = tf.Variable(tf.zeros([1]),
							name='h_bias')

	def propagate_v2h(self, vis):
		pre_sigmoid = tf.add(self.v_bias,tf.matmul(vis,self.weights))
		return [pre_sigmoid, tf.sigmoid(pre_sigmoid)]

	def propagate_h2v(self, hid):
		pre_sigmoid = tf.add(self.v_bias,tf.matmul(hid,self.weights))
		return [pre_sigmoid, tf.sigmoid(pre_sigmoid)]

	def sample_h_given_v(self, v0_sample):
		pre_sigmoid_h1, h1_probability = self.propagate_v2h(v0_sample)
		h1_sample = tf.contrib.distributions.Bernoulli(p=h1_probability)
		return [pre_sigmoid_h1, h1_probability, h1_sample]

	def sample_v_given_h(self, h0_sample):
		pre_sigmoid_v1, v1_probability = self.propagate_h2v(h0_sample)
		v1_sample = tf.contrib.distributions.Bernoulli(p=v1_probability)
		return [pre_sigmoid_v1, v1_probability, v1_sample]

	def gibbs_vhv(self, v0_sample):
		pre_sigmoid_h1, h1_probability, h1_sample = self.sample_h_given_v(v0_sample)
		pre_sigmoid_v1, v1_probability, v1_sample = self.sample_v_given_h(h1_sample)
		return [pre_sigmoid_h1, h1_probability, h1_sample, 
				pre_sigmoid_v1, v1_probability, v1_sample]

	def gibbs_hvh(self, h0_sample):
		pre_sigmoid_v1, v1_probability, v1_sample = self.sample_v_given_h(h0_sample)
		pre_sigmoid_h1, h1_probability, h1_sample = self.sample_h_given_v(v1_sample)
		return [pre_sigmoid_v1, v1_probability, v1_sample,
				pre_sigmoid_h1, h1_probability, h1_sample]		

	def free_energy():
		pass
		