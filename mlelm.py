# coding: utf-8
import tensorflow as tf
import numpy as np


class ELM_AE(object):
	'''extreme learning machine based anto-encoder'''

	def __init__(self, sess, batch_size, input_len, hidden_num, omega=1.):
		self._sess = sess
		self._batch_size = batch_size
		self._input_len = input_len
		self._hidden_num = hidden_num
		self._output_len = input_len
		self._omega = omega

		# for train
		self._x0 = tf.placeholder(tf.float32, [self._batch_size, self._input_len])
		self._t0 = tf.placeholder(tf.float32, [self._batch_size, self._output_len])

		# for test
		self._x1 = tf.placeholder(tf.float32, [None, self._input_len])
		self._t1 = tf.placeholder(tf.float32, [None, self._output_len])

		orthogonal = tf.orthogonal_initializer(seed=2017)
		W = orthogonal([self._input_len, self._hidden_num], dtype=tf.float32)
		b = orthogonal([1, self._hidden_num], dtype=tf.float32)
		self._W = tf.Variable(self._sess.run(W), trainable=False, dtype=tf.float32)
		self._b = tf.Variable(self._sess.run(b), trainable=False, dtype=tf.float32)
		self._beta = tf.Variable(tf.zeros([self._hidden_num, self._output_len]), trainable=False, dtype=tf.float32)
		self._var_list = [self._W, self._b, self._beta]

		self.H0 = tf.nn.relu(tf.matmul(self._x0, self._W) + self._b)
		# self.H0 = tf.sigmoid(tf.matmul(self._x0, self._W) + self._b)  # N x K
		self.H0_T = tf.transpose(self.H0)

		self.H1 = tf.nn.relu(tf.matmul(self._x1, self._W) + self._b)
		# self.H1 = tf.sigmoid(tf.matmul(self._x1, self._W) + self._b)  # N x K
		self.H1_T = tf.transpose(self.H1)

		# beta analytic solution : self._beta_s (K x O)
		if self._input_len == self._hidden_num:
			s, u, v = tf.svd(tf.matmul(self.H0_T, self._t0))
			self._beta_s = tf.matmul(u, tf.transpose(v))
		elif self._hidden_num < self._batch_size:  # L < K
			identity = tf.constant(np.identity(self._hidden_num), dtype=tf.float32)
			self._beta_s = tf.matmul(
				tf.matmul(tf.matrix_inverse(tf.matmul(self.H0_T, self.H0) + identity / self._omega), self.H0_T),
				self._t0)
		# _beta_s = (H_T*H + I/om)^(-1)*H_T*T
		else:
			identity = tf.constant(np.identity(self._batch_size), dtype=tf.float32)
			self._beta_s = tf.matmul(
				tf.matmul(self.H0_T, tf.matrix_inverse(tf.matmul(self.H0, self.H0_T) + identity / self._omega)),
				self._t0)
		# _beta_s = H_T*(H*H_T + I/om)^(-1)*T

		self._assign_beta = self._beta.assign(self._beta_s)
		self._fx0 = tf.matmul(self.H0, self._beta)
		self._fx1 = tf.matmul(self.H1, self._beta)

		self._cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self._fx0, labels=self._t0))

		self._init = False
		self._feed = False

		# for the mnist test
		# self._correct_prediction = tf.equal(tf.argmax(self._fx1, 1), tf.argmax(self._t1, 1))
		# self._accuracy = tf.reduce_mean(tf.cast(self._correct_prediction, tf.float32))
		self._accuracy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self._fx1, labels=self._t1))

	def feed(self, x, corrtype='none', corrfrac=0.0):
		'''
		Args :
		  x : input array (N x L)
		  t : label array (N x O)
		'''

		if not self._init: self.init()
		corruption_ratio = np.round(corrfrac * x.shape[1]).astype(np.int)

		if corrtype == 'none':
			pass

		if corrfrac > 0.0:
			if corrtype == 'masking':
				x = masking_noise(x, self._sess, corrfrac)

			elif corrtype == 'salt_and_pepper':
				x = salt_and_pepper_noise(x, corruption_ratio)
		else:
			pass
		self._sess.run(self._assign_beta, {self._x0: x, self._t0: x})
		self._feed = True

	def init(self):
		self._sess.run(tf.variables_initializer(self._var_list))
		self._init = True

	def test(self, x):
		if not self._feed: exit("Not feed-forward trained")
		print("Accuracy: {:.9f}".format(self._sess.run(self._accuracy, {self._x1: x, self._t1: x})))
		return self._sess.run(self._fx1, {self._x1: x})

	def getbeta(self):
		return self._beta


class MLELM(object):
	'''
	multilayer extreme learning machine based anto-encoder
	three hidden layers
	'''

	def __init__(self, sess, batch_size, input_len, hidden1_num, hidden2_num, hidden3_num, output_len, omega=1.):
		self._sess = sess
		self._batch_size = batch_size
		self._input_len = input_len
		self._hidden1_num = hidden1_num
		self._hidden2_num = hidden2_num
		self._hidden3_num = hidden3_num
		self._output_len = output_len
		self._omega = omega

		# for train
		self._x0 = tf.placeholder(tf.float32, [self._batch_size, self._input_len])
		self._t0 = tf.placeholder(tf.float32, [self._batch_size, self._output_len])

		# for test
		self._x1 = tf.placeholder(tf.float32, [None, self._input_len])
		self._t1 = tf.placeholder(tf.float32, [None, self._output_len])

		self._W1 = tf.Variable(
			tf.zeros([self._input_len, self._hidden1_num]),
			trainable=False, dtype=tf.float32)
		self._W2 = tf.Variable(
			tf.zeros([self._hidden1_num, self._hidden2_num]),
			trainable=False, dtype=tf.float32)
		self._W3 = tf.Variable(
			tf.zeros([self._hidden2_num, self._hidden3_num]),
			trainable=False, dtype=tf.float32)
		self._W4 = tf.Variable(
			tf.zeros([self._hidden3_num, self._output_len]),
			trainable=False, dtype=tf.float32)
		self._var_list = [self._W1, self._W2, self._W3, self._W4]

		self.H1 = tf.nn.relu(tf.matmul(self._x0, self._W1))
		# self.H1 = tf.sigmoid(tf.matmul(self._x0, self._W1))  # N x K
		self.H1_T = tf.transpose(self.H1)
		self.HT1 = tf.nn.relu(tf.matmul(self._x1, self._W1))
		# self.HT1=tf.sigmoid(tf.matmul(self._x1,self._W1))
		self.HT1_T = tf.transpose(self.HT1)

		if self._hidden1_num == self._hidden2_num:
			self.H2 = tf.matmul(self.H1, self._W2)
			self.HT2 = tf.matmul(self.HT1, self._W2)
		else:
			self.H2 = tf.nn.relu(tf.matmul(self.H1, self._W2))  # N x K
			self.HT2 = tf.nn.relu(tf.matmul(self.HT1,
											self._W2))  # self.H2 = tf.sigmoid(tf.matmul(self.H1, self._W2))  # N x K  # self.HT2=tf.sigmoid(tf.matmul(self.HT1,self._W2))
		self.H2_T = tf.transpose(self.H2)
		self.HT2_T = tf.transpose(self.HT2)

		if self._hidden2_num == self._hidden3_num:
			self.H3 = tf.matmul(self.H2, self._W3)
			self.HT3 = tf.matmul(self.HT2, self._W3)
		else:
			self.H3 = tf.nn.relu(tf.matmul(self.H2, self._W3))  # N x K
			self.HT3 = tf.nn.relu(tf.matmul(self.HT2,
											self._W3))  # self.H3 = tf.sigmoid(tf.matmul(self.H2, self._W3))  # N x K  # self.HT3=tf.sigmoid(tf.matmul(self.HT2,self._W3))
		self.H3_T = tf.transpose(self.H3)
		self.HT3_T = tf.transpose(self.HT3)

		# beta analytic solution : self._beta_s (K x O)
		if self._hidden3_num < self._batch_size:  # L < K
			identity = tf.constant(np.identity(self._hidden3_num), dtype=tf.float32)
			self._W_s = tf.matmul(
				tf.matmul(tf.matrix_inverse(tf.matmul(self.H3_T, self.H3) + identity / self._omega), self.H3_T),
				self._t0)  # _beta_s = (H_T*H + I/om)^(-1)*H_T*T
		else:
			identity = tf.constant(np.identity(self._batch_size), dtype=tf.float32)
			self._W_s = tf.matmul(
				tf.matmul(self.H3_T, tf.matrix_inverse(tf.matmul(self.H3, self.H3_T) + identity / self._omega)),
				self._t0)  # _beta_s = H_T*(H*H_T + I/om)^(-1)*T

		self._assign_W = self._W4.assign(self._W_s)
		self._fx0 = tf.matmul(self.H3, self._W4)
		self._fx1 = tf.matmul(self.HT3, self._W4)

		# self._cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self._fx0, labels=self._t0))
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
		elm1 = ELM_AE(self._sess, self._batch_size, self._input_len, self._hidden1_num, 1)
		elm1.feed(x, corrtype='masking', corrfrac=0.5)
		W1_s = elm1.getbeta()
		self._assign_W1 = self._W1.assign(tf.transpose(W1_s))
		self._sess.run(self._assign_W1, {self._x0: x})
		H1 = self._sess.run(self.H1, {self._x0: x})
		elm2 = ELM_AE(self._sess, self._batch_size, self._hidden1_num, self._hidden2_num, 0.1)
		elm2.feed(H1, corrtype='masking', corrfrac=0.4)
		W2_s = elm2.getbeta()
		self._assign_W2 = self._W2.assign(tf.transpose(W2_s))
		self._sess.run(self._assign_W2, {self._x0: x})
		H2 = self._sess.run(self.H2, {self._x0: x})
		elm3 = ELM_AE(self._sess, self._batch_size, self._hidden2_num, self._hidden3_num, 0.1)
		elm3.feed(H2, corrtype='masking', corrfrac=0.3)
		W3_s = elm3.getbeta()
		self._assign_W3 = self._W3.assign(tf.transpose(W3_s))
		self._sess.run(self._assign_W3, {self._x0: x})
		self._sess.run(self._assign_W, {self._x0: x, self._t0: t})
		self._feed = True
		return self._sess.run(self._train_accuracy,{self._x0:x,self._t0:t})

	def init(self):
		self._sess.run(tf.variables_initializer(self._var_list))
		self._init = True

	def test(self, x, t=None,is_trained=False):
		if is_trained:self._feed=True
		if not self._feed: exit("Not feed-forward trained")
		if t is not None:
			print("Accuracy: {:.9f}".format(self._sess.run(self._accuracy, {self._x1: x, self._t1: t})))
		else:
			return self._sess.run(self._fx1, {self._x1: x})


def masking_noise(data, sess, v):
	"""Apply masking noise to data in X.

	In other words a fraction v of elements of X
	(chosen at random) is forced to zero.
	:param data: array_like, Input data
	:param sess: TensorFlow session
	:param v: fraction of elements to distort, float
	:return: transformed data
	"""
	data_noise = data.copy()
	rand = tf.random_uniform(data.shape)
	data_noise[sess.run(tf.nn.relu(tf.sign(v - rand))).astype(np.bool)] = 0

	return data_noise


def salt_and_pepper_noise(X, v):
	"""Apply salt and pepper noise to data in X.

	In other words a fraction v of elements of X
	(chosen at random) is set to its maximum or minimum value according to a
	fair coin flip.
	If minimum or maximum are not given, the min (max) value in X is taken.
	:param X: array_like, Input data
	:param v: int, fraction of elements to distort
	:return: transformed data
	"""
	X_noise = X.copy()
	n_features = X.shape[1]

	mn = X.min()
	mx = X.max()

	for i, sample in enumerate(X):
		mask = np.random.randint(0, n_features, v)

		for m in mask:

			if np.random.random() < 0.5:
				X_noise[i][m] = mn
			else:
				X_noise[i][m] = mx

	return X_noise
