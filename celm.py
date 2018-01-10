import tensorflow as tf
import numpy as np


# CHECK : Constants
# omega = 100000.


class CELM(object):
	def __init__(self, sess, batch_size, input_len, hidden_num, output_len, omega=1.):
		'''
		Args:
		  sess : TensorFlow session.
		  batch_size : The batch size (N)
		  input_len : The length of input. (L)
		  hidden_num : The number of hidden node. (K)
		  output_len : The length of output. (O)
		'''

		self._sess = sess
		self._batch_size = batch_size
		self._input_len = input_len
		self._hidden_num = hidden_num
		self._output_len = output_len

		# for train
		self._x0 = tf.placeholder(tf.float32, [self._batch_size, self._input_len])
		self._t0 = tf.placeholder(tf.float32, [self._batch_size, self._output_len])

		# for test
		self._x1 = tf.placeholder(tf.float32, [None, self._input_len])
		self._t1 = tf.placeholder(tf.float32, [None, self._output_len])

		self._W = tf.Variable(tf.zeros([self._input_len, self._hidden_num]), trainable=False, dtype=tf.float32)
		self._b = tf.Variable(tf.zeros([self._hidden_num]), trainable=False, dtype=tf.float32)
		self._beta = tf.Variable(tf.zeros([self._hidden_num, self._output_len]), trainable=False, dtype=tf.float32)
		self._var_list = [self._W, self._b, self._beta]

		self.H0 = tf.nn.relu(tf.matmul(self._x0, self._W) + self._b)  # N x K
		# self.H0 = tf.sigmoid(tf.matmul(self._x0, self._W) + self._b)  # N x K
		self.H0_T = tf.transpose(self.H0)

		self.H1 = tf.nn.relu(tf.matmul(self._x1, self._W) + self._b)  # N x K
		# self.H1 = tf.sigmoid(tf.matmul(self._x1, self._W) + self._b)  # N x K
		self.H1_T = tf.transpose(self.H1)

		# beta analytic solution : self._beta_s (K x O)
		if self._hidden_num < self._batch_size:  # L < K
			identity = tf.constant(np.identity(self._hidden_num), dtype=tf.float32)
			self._beta_s = tf.matmul(
				tf.matmul(tf.matrix_inverse(tf.matmul(self.H0_T, self.H0) + identity / omega), self.H0_T),
				self._t0)  # _beta_s = (H_T*H + I/om)^(-1)*H_T*T
		else:
			identity = tf.constant(np.identity(self._batch_size), dtype=tf.float32)
			self._beta_s = tf.matmul(
				tf.matmul(self.H0_T, tf.matrix_inverse(tf.matmul(self.H0, self.H0_T) + identity / omega)),
				self._t0)  # _beta_s = H_T*(H*H_T + I/om)^(-1)*T

		self._assign_beta = self._beta.assign(self._beta_s)
		self._fx0 = tf.matmul(self.H0, self._beta)
		self._fx1 = tf.matmul(self.H1, self._beta)

		# self._cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = self._fx0, labels=self._t0))
		self._train_correct_prediction = tf.equal(tf.argmax(self._fx0, 1), tf.argmax(self._t0, 1))
		self._train_accuracy = tf.reduce_mean(tf.cast(self._train_correct_prediction, tf.float32))

		self._init = False
		self._feed = False

		# for the mnist test
		self._correct_prediction = tf.equal(tf.argmax(self._fx1, 1), tf.argmax(self._t1, 1))
		self._accuracy = tf.reduce_mean(tf.cast(self._correct_prediction, tf.float32))

	def feed(self, x, t):
		'''
		Args :
		  x : input array (N x L)
		  t : label array (N x O)
		'''

		if not self._init: self.init()
		flag, i = 1, 0
		weight = np.zeros([self._hidden_num, self._input_len], float)
		bias = np.zeros([self._hidden_num], float)
		while i < self._hidden_num:
			Class1_label = Class2_label = 1
			while Class1_label == Class2_label:
				Class1 = np.random.randint(self._batch_size)
				Class2 = np.random.randint(self._batch_size)
				Class1_label, Class2_label = np.argmax(t[Class1]), np.argmax(t[Class2])
			if Class1_label > Class2_label:
				Class1, Class2 = Class2, Class1
			weight[i] = x[Class1] - x[Class2]
			normweight = np.dot(weight[i], weight[i])
			if normweight < 1. / (flag * self._output_len):
				flag += 1
				continue
			weight[i] = weight[i] / (normweight / 2)
			# if i < self._hidden_num /2:
			# 	weight[i][:900]=0
			# else:weight[i][900:]=0
			bias[i] = np.dot((x[Class2] + x[Class1]).T, (x[Class2] - x[Class1])) / normweight
			# bias[i] = np.abs(bias[i])
			i += 1
		print bias[self._hidden_num - 1]

		self._assign_W = self._W.assign(weight.T)
		self._assign_bias = self._b.assign(bias)

		self._sess.run(self._assign_W, {self._x0: x})
		self._sess.run(self._assign_bias, {self._x0: x})
		self._sess.run(self._assign_beta, {self._x0: x, self._t0: t})

		self._feed = True
		return self._sess.run(self._train_accuracy, {self._x0: x, self._t0: t})

	def feedcs(self, x, t):
		'''
		Constrained Sum ELM
		'''
		if not self._init: self.init()
		i = 0
		weight = np.zeros([self._hidden_num, self._input_len], float)
		while i < self._hidden_num:
			Class1 = Class2 = 0
			Class1_label, Class2_label = 1, 0
			while Class1_label != Class2_label:
				Class1 = np.random.randint(self._batch_size)
				Class2 = np.random.randint(self._batch_size)
				Class1_label, Class2_label = np.argmax(t[Class1]), np.argmax(t[Class2])
			weight[i] = x[Class1] + x[Class2]
			normweight = np.dot(weight[i], weight[i])
			weight[i] = weight[i] / (normweight)
			i += 1

		self._assign_W = self._W.assign(weight.T)
		self._assign_bias = self._b.assign(np.random.random([self._hidden_num]))

		self._sess.run(self._assign_W, {self._x0: x})
		self._sess.run(self._assign_bias, {self._x0: x})
		self._sess.run(self._assign_beta, {self._x0: x, self._t0: t})
		self._feed = True

	def init(self):
		self._sess.run(tf.variables_initializer(self._var_list))
		self._init = True

	def test(self, x, t=None, trained=False):
		if trained: self._feed = True
		if not self._feed: exit("Not feed-forward trained")
		if t is not None:
			print("Accuracy: {:.9f}".format(self._sess.run(self._accuracy, {self._x1: x, self._t1: t})))
		else:
			return self._sess.run(self._fx1, {self._x1: x})
