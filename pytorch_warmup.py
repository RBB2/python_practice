import torch
import numpy as np
import matplotlib

import gzip, pickle
import time
import random

matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
with gzip.open("mnist.pkl.gz") as f:
        train_set, valid_set, test_set = pickle.load(f, encoding="bytes")

# Initialize dataset
X = train_set[0]
Y = train_set[1]

X_dev = valid_set[0]
Y_dev = valid_set[1]

X_test = test_set[0]
Y_test = test_set[1]

n = X.shape[0]

X_torch = torch.from_numpy(X).float()
Y_torch = torch.from_numpy(Y).long()

X_torch_dev = torch.from_numpy(X_dev).float()
Y_torch_dev = torch.from_numpy(Y_dev).long()

X_torch_test = torch.from_numpy(X_test).float()
Y_torch_test = torch.from_numpy(Y_test).long()


d = X.shape[1]
W = torch.zeros((d,10), requires_grad=True)
b = torch.zeros(10, requires_grad=True)

def model(X):
	return X @ W + b
loss_fn = torch.nn.CrossEntropyLoss()

optim = torch.optim.SGD([W, b], lr = 0.1)

def misclass_err(X,Y,y_hat,w,b):
    Yhat_labels = np.argmax(y_hat, axis=0)
    errors = (Y == Yhat_labels)
    return 100 * np.sum(errors)/(y_hat.shape[0]*1.0)


iterations = []
train_loss = []
dev_loss = []
mis_class_err = []
for i in range(1):
	y_hat = model(X_torch)
	#loss = loss_fn(y_hat, Y_torch)
	#y_hat_dev = model(X_torch_dev)
	#loss_dev = loss_fn(y_hat_dev, Y_torch_dev)
	miss_err = misclass_err(X_test,Y_test,y_hat,W,b)

	if i % 20 == 0:
		print('i', i, 'loss', miss_err)
		iterations.append(i)
		#train_loss.append(loss)
		#dev_loss.append(loss_dev)
		mis_class_err.append(miss_err)

	optim.zero_grad()
	loss.backward()
	optim.step()


#plt.plot(iterations, train_loss, label='train log loss')
plt.plot(iterations, mis_class_err, label='dev')
plt.legend(bbox_to_anchor=(0.4,0.9),loc=2,borderaxespad=0.)
plt.xlabel('iterations')
plt.ylabel('Log loss error')
plt.title('Train and Dev set log loss')
plt.show()


