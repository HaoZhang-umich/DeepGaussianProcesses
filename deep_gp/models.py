import numpy as np
from scipy.stats import norm
from scipy.special import logsumexp
from utils import Normal, RBF
from deep_gp import DGP

class DGPModel(object):
    def __init__(self, n_layer=4, n_inducing=100, n_iter=1000, n_sample=100):
        self.n_inducing = n_inducing
        self.n_iter = n_iter
        self.n_sample = n_sample
        self.n_layer = n_layer
        print("n inducing:", self.n_inducing)
        print("max n_iter:", self.n_iter)
        self.window_size = 100
        self.sample_space = 50
        self.model = None

    def fit(self, X, Y):
        if len(Y.shape) == 1:
            Y = Y.reshape(-1,1)
        likelihood = Normal(np.var(Y, 0))
        n, d = X.shape[0], X.shape[1]
        layers_kernel = []
        for i_layer in range(self.n_layer):
            layers_kernel.append(RBF(d, scale=np.sqrt(d)))


        self.model = DGP(X, Y, self.n_inducing, layers_kernel, likelihood,
                         window_size=self.window_size)

        for i_iter in range(self.n_iter):
            self.model.train()
            if i_iter % 100 == 0:
                print('Iter {}'.format(i_iter))
                self.model.print_MLL()
            if i_iter % 1000 == 0:
                self.model.sampling(self.n_sample, self.sample_space)

    def _predict(self, X_test, S):
        ms, vs = [], []
        n = max(len(X_test) / 100, 1)  
        for x_test in np.array_split(X_test, n):
            m, v = self.model.predict(x_test, S)
            ms.append(m)
            vs.append(v)

        return np.concatenate(ms, 1), np.concatenate(vs, 1)

    def predict(self, X_test):
        ms, vs = self._predict(X_test, self.n_sample)
        m = np.mean(ms, 0)
        v = np.mean(vs + ms**2, 0) - m**2
        return m, v

    def pdf(self, X_test, Y_test):
        ms, vs = self._predict(X_test, self.n_sample)
        logps = norm.logpdf(np.repeat(Y_test[None, :, :], self.n_sample, axis=0), ms, np.sqrt(vs))
        return logsumexp(logps, axis=0) - np.log(self.n_sample)