import numpy as np
from scipy.stats import norm
import pandas
import tensorflow
import models
import pandas as pd
concrete = './data/concrete/Concrete_Data.xls'
concrete = pd.read_excel(concrete)
y = concrete.iloc[:,-1]
x = concrete.iloc[:,:-1]

print("data loaded sucessfully")

N = x.shape[0]
n = int(N * 0.8)
ind = np.arange(N)

np.random.shuffle(ind)
train_ind = ind[:n]
test_ind = ind[n:]

X = x.iloc[train_ind] 
Xs = x.iloc[test_ind]
Y = y.iloc[train_ind]
Ys = y.iloc[test_ind]
X = X.values
Xs = Xs.values
Y = (Y.values).reshape(-1,1)
Ys = (Ys.values).reshape(-1,1)

print(X.shape, Y.shape)

X_mean = np.mean(X, 0)
X_std = np.std(X, 0)
X = (X - X_mean) / X_std
Xs = (Xs - X_mean) / X_std
Y_mean = np.mean(Y, 0)
Y = (Y - Y_mean)
Ys = (Ys - Y_mean)

print("data sphered")

model = models.DGPModel(n_layer=4, n_inducing=100, n_iter=10000, n_sample=10)
print("init model")
model.fit(X, Y)
print("model trained!")

m, v = model.predict(Xs)
print('MSE', np.mean(np.square(Ys - m)))
print('MLL', np.mean(model.pdf(Xs, Ys)))
