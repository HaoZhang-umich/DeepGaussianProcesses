import tensorflow as tf
import numpy as np
from scipy.cluster.vq import kmeans2


class Normal(object):

    def __init__(self, variance=0.1**2):
        self.variance = tf.Variable(variance)

    def log_pdf(self, x, loc, var):
        return - tf.log(2 * np.pi * var) / 2 - (x-loc)**2 / var / 2.0


class RBF(object):

    def __init__(self, d, width=10, scale=1.0):
        self.d = int(d)
        self.width = tf.exp(tf.Variable(np.log(width), dtype=tf.float64))
        self.scale = tf.Variable(scale, dtype=tf.float64)

    def K(self, X1, X2=None):
        if X2 is None: X2 = X1
        X1_squared = tf.reduce_sum(tf.square(X1), axis=-1, keepdims=True)
        X2_squared = tf.reduce_sum(tf.square(X2), axis=-1, keepdims=True)
        tmp = X1_squared - 2 * tf.matmul(X1, tf.transpose(X2)) + tf.transpose(X2_squared)
        tmp /= self.scale**2
        return tf.exp(-tmp / 2) / self.width**2

    def K_diag(self, X):
        return tf.ones(tf.shape(X)[:-1], dtype=tf.float64) / self.width**2        


def GP_prediction(X_test, X, f, kernel):

    n = tf.shape(X)[0]
    d = tf.shape(f)[1]

    Kx = kernel.K(X) + tf.eye(n, dtype=tf.float64) * 1e-6
    Kxx_test = kernel.K(X, X_test)
    Kx_test = kernel.K_diag(X_test)

    S = tf.linalg.triangular_solve(tf.cholesky(Kx), Kxx_test)

    mean = tf.matmul(tf.transpose(S), f)
    var = Kx_test - tf.transpose(tf.reduce_sum(tf.square(S), 0))
    var = tf.tile(var[:,None], [1, d])

    return mean, var

class Layer(object):
    def __init__(self, kernel, d_out, n_inducing, X):
        self.d_in, self.d_out = kernel.d, d_out
        self.kernel = kernel
        self.n_inducing= n_inducing

        self.Z = tf.Variable(kmeans2(X, self.n_inducing)[0], dtype=tf.float64)

        self.mean = np.zeros((self.d_in, self.d_out))
        for i in range(min(self.d_in, self.d_out)): self.mean[i,i] = 1

        self.U = tf.Variable(np.zeros((self.n_inducing, self.d_out)), dtype=tf.float64, trainable=False)

    def GP(self, X):
        mean, var = GP_prediction(X, self.Z, self.U, self.kernel)
        return mean, var

    def log_prior(self):
        return -tf.reduce_sum(tf.square(self.U)) / 2.0    