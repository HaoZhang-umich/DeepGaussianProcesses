import tensorflow as tf
import numpy as np
from utils import GP_prediction, Layer


class DGP(object):

    def __init__(self, X, Y, n_inducing, kernels, likelihood, window_size,
                 lr=0.01, epsilon=0.01, mdecay=0.05):
        self.n_inducing = n_inducing
        self.kernels = kernels
        self.likelihood = likelihood
        self.window_size = window_size

        n_layers = len(kernels)
        n = X.shape[0]

        self.layers = []
        for l in range(n_layers):
            d_out = self.kernels[l+1].d if l+1 < n_layers else Y.shape[1]
            self.layers.append(Layer(self.kernels[l], d_out, n_inducing, X))

        self.X_placeholder = tf.placeholder(tf.float64, shape=[None, X.shape[1]])
        self.Y_placeholder = tf.placeholder(tf.float64, shape=[None, Y.shape[1]])
        self.X = X
        self.Y = Y
        self.n = X.shape[0]
        self.vars = [l.U for l in self.layers]
        self.data_iter = 0
        self.window_size = window_size
        self.window = []
        self.samples = []
        self.sample_opt = None
        self.burn_in_opt = None

        self.f, self.fmeans, self.fvars = self.feed_forward(self.X_placeholder)
        self.y_mean, self.y_var = self.fmeans[-1], self.fvars[-1]+self.likelihood.variance

        self.log_prior = tf.add_n([l.log_prior() for l in self.layers])
        self.log_likelihood = self.likelihood.log_pdf(self.Y_placeholder, self.y_mean, self.y_var)

        self.obj = - tf.reduce_sum(self.log_likelihood) / tf.cast(tf.shape(self.X_placeholder)[0], tf.float64) \
                   - (self.log_prior / n)

        self.epsilon = epsilon
        burn_in_updates = []
        sample_updates = []

        grads = tf.gradients(self.obj, self.vars)

        for theta, grad in zip(self.vars, grads):
            xi = tf.Variable(tf.ones_like(theta), dtype=tf.float64, trainable=False)
            g = tf.Variable(tf.ones_like(theta), dtype=tf.float64, trainable=False)
            g2 = tf.Variable(tf.ones_like(theta), dtype=tf.float64, trainable=False)
            p = tf.Variable(tf.zeros_like(theta), dtype=tf.float64, trainable=False)

            r_t = 1. / (xi + 1.)
            g_t = (1. - r_t) * g + r_t * grad
            g2_t = (1. - r_t) * g2 + r_t * grad ** 2
            xi_t = 1. + xi * (1. - g * g / (g2 + 1e-16))
            Minv = 1. / (tf.sqrt(g2 + 1e-16) + 1e-16)

            burn_in_updates.append((xi, xi_t))
            burn_in_updates.append((g, g_t))
            burn_in_updates.append((g2, g2_t))

            epsilon_scaled = epsilon / tf.sqrt(tf.cast(self.n, tf.float64))
            noise_scale = 2. * epsilon_scaled ** 2 * mdecay * Minv
            sigma = tf.sqrt(tf.maximum(noise_scale, 1e-16))
            sample_t = tf.random_normal(tf.shape(theta), dtype=tf.float64) * sigma
            p_t = p - epsilon ** 2 * Minv * grad - mdecay * p + sample_t
            theta_t = theta + p_t

            sample_updates.append((theta, theta_t))
            sample_updates.append((p, p_t))

        self.sample_opt = [tf.assign(var, var_t) for var, var_t in sample_updates]
        self.burn_in_opt = [tf.assign(var, var_t) for var, var_t in burn_in_updates + sample_updates]

        self.adam = tf.train.AdamOptimizer(lr)
        self.hyper_opt = self.adam.minimize(self.obj)

        self.session = tf.Session()
        init_op = tf.global_variables_initializer()
        self.session.run(init_op)

    def sampling(self, num, spacing):
        self.samples = []
        for i in range(num):
            for j in range(spacing):
                feed_dict = {self.X_placeholder: self.X, self.Y_placeholder: self.Y}
                self.session.run((self.sample_opt), feed_dict=feed_dict)

            values = self.session.run((self.vars))
            sample = {}
            for var, value in zip(self.vars, values):
                sample[var] = value
            self.samples.append(sample)

    def train(self):
        feed_dict = {self.X_placeholder: self.X, self.Y_placeholder: self.Y}
        self.session.run(self.burn_in_opt, feed_dict=feed_dict)
        values = self.session.run((self.vars))
        sample = {}
        for var, value in zip(self.vars, values):
            sample[var] = value
        self.window.append(sample)
        if len(self.window) > self.window_size:
            self.window = self.window[-self.window_size:]

        feed_dict = {self.X_placeholder: self.X, self.Y_placeholder: self.Y}
        feed_dict.update(self.window[np.random.randint(len(self.window))])
        self.session.run(self.hyper_opt, feed_dict=feed_dict)

    def print_MLL(self):
        feed_dict = {self.X_placeholder: self.X, self.Y_placeholder: self.Y}
        mll = np.mean(self.session.run((self.log_likelihood), feed_dict=feed_dict), 0)
        print('Mean log likelihood: {}'.format(mll))

    def feed_forward(self, X):
        Fs = [X, ]
        Fmeans, Fvars = [], []

        for layer in self.layers:
            mean, var = layer.GP(Fs[-1])
            eps = tf.random_normal(tf.shape(mean), dtype=tf.float64)
            F = mean + eps * tf.sqrt(var)
            Fs.append(F)
            Fmeans.append(mean)
            Fvars.append(var)

        return Fs[1:], Fmeans, Fvars

    def predict(self, X, S):
        assert S <= len(self.samples)
        ms, vs = [], []
        for i in range(S):
            feed_dict = {self.X_placeholder: X}
            feed_dict.update(self.samples[i])
            m, v = self.session.run((self.y_mean, self.y_var), feed_dict=feed_dict)
            ms.append(m)
            vs.append(v)
        return ms, vs