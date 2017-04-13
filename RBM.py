import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

class RBM():
	def __init__(self, n_hidden = 20, n_visible = 784):
		self.input = tf.placeholder(tf.float32, [None, n_visible])
		self.weights = tf.Variable(tf.truncated_normal(
									[n_visible,n_hidden],
									stddev=1.0),
							name='weights')
		
		self.v_bias = tf.Variable(tf.zeros([n_visible]),
							name='v_bias')

		self.h_bias = tf.Variable(tf.zeros([n_hidden]),
							name='h_bias')

		self.params = [self.weights, self.h_bias, self.v_bias]

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

	def free_energy(self, v_sample):
		vBiasTerm = tf.matmul(self.v_bias,v_sample)
		hWvTerm = tf.reduce_sum(tf.log(1 + tf.exp(1 + tf.add(tf.matmul(self.weights,v_sample),self.h_bias))))
		return - vBiasTerm - hWvTerm

	def CD_learning(self, lr=0.1, k=1):
		pre_sigmoid_h0, h0_probability, h0_sample, pre_sigmoid_v1, v1_probability, v1_sample = gibbs_vhv(self.input)
		pre_sigmoid_h1, h1_probability, h1_sample = sample_h_given_v(v1_sample)
		
		#Positive gradient
		self.w_positive = tf.matmul(tf.transpose(self.input), h0_sample)
		#Negative gradient
		self.w_negative = tf.matmul(tf.transpose(v1_sample), h1_sample)

		self.update_w = self.weights - alpha * (self.w_positive - self.w_negative) / (tf.shape(self.input)[0])
		self.update_vb = self.v_bias - alpha * tf.reduce_mean(self.input - v1_sample, 0)
		self.update_hb = self.h_bias - alpha * tf.reduce_mean(h0_sample - h1_sample,0)

		#Sampling functions
        _, _, self.h_sample = sample_h_given_v(self.input)
        _, _, self.v_sample = sample_v_given_h(self.h_sample)

        #Cost
        self.err_sum = tf.reduce_mean(tf.square(self.input - self.v_sample))
		
if __name__ == '__main__':
	mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
	tf.global_variables_initializer().run()