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

def compute_yhat(X):
	w = W.detach().numpy()
	B = b.detach().numpy()
	return X @ w + B

loss_fn = torch.nn.CrossEntropyLoss()

optim = torch.optim.SGD([W, b], lr = 0.1)

def misclass_err(X,Y,y_hat,w,b):
	#print(np.shape(y_hat))
	Yhat_labels = np.argmax(y_hat, axis=1)
	errors = np.sum(Y != Yhat_labels)
	#print(errors)
	#print(y_hat.shape[0])
	return 100 * errors/(y_hat.shape[0]*1.0)


iterations = []
train_loss = []
dev_loss = []
mis_class_err = []
min_error = 100
for i in range(10000):
	y_hat_np = compute_yhat(X_test)
	miss_err = misclass_err(X_test,Y_test,y_hat_np,W,b)
	if (miss_err < min_error):
		min_error = miss_err

	y_hat_torch = model(X_torch)
	loss = loss_fn(y_hat_torch, Y_torch)
	#y_hat_dev = model(X_torch_dev)
	#loss_dev = loss_fn(y_hat_dev, Y_torch_dev)
	

	if i % 500 == 0 and i > 200:
		print('i', i, 'loss', loss.item(), 'miss err', miss_err)
		iterations.append(i)
		#train_loss.append(loss)
		#dev_loss.append(loss_dev)
		mis_class_err.append(miss_err)

	optim.zero_grad()
	loss.backward()
	optim.step()

print(min_error)


#plt.plot(iterations, train_loss, label='train log loss')
plt.plot(iterations, mis_class_err, label='test')
plt.legend(bbox_to_anchor=(0.4,0.9),loc=2,borderaxespad=0.)
plt.xlabel('iterations')
plt.ylabel('misclassification error')
plt.title('Test set misclassification error')
plt.show()

