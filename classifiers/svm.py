from importData import *
import numpy as np


class SVM:
	def __init__(self, train_data, train_labels, test_data, test_labels,
			kernel_choice = "rbf", eps = 1e-05, epochs = 1000, lr=0.01, margin=10000,
			seed=True, validate=True, kparam=0.2):
		self.kernel_choice = kernel_choice
		if kernel_choice=="rbf":
			self.kernel = self.rbf_kernel(kparam)
		if kernel_choice=="poly":
			self.kernel = self.poly_kernel(kparam)
		self.eps = eps
		self.lr = lr
		self.margin = margin
		self.seed = seed
		self.validate = validate
		self.kparam = kparam
		self.weights = None
		self.epochs = epochs
		self.train_data = train_data
		self.train_labels = train_labels
		self.test_data = test_data
		self.test_labels = test_labels
		self.trained_weights = None

	def predict(self, phase):
		if phase == "train":
			data = self.train_data
			labels = self.train_labels
		elif phase == "test":
			data = self.test_data
			labels = self.test_labels
		K = self.kernel(data, data)
		K1 = self.kernel(data, self.train_data)
		K1 = np.insert(K1, K1.shape[1], 1, axis=1)
		# trained_weights = np.expand_dims(self.weights, -1)
		trained_weights = self.weights
		
		prediction = np.sign(np.dot(K1, trained_weights))
		prediction = prediction == labels
		acc = np.sum(prediction)/K1.shape[0]
		print("Accuracy on phase: " + phase + " ,is: " + str(acc))

	def fit(self):
		self.train_data.reshape(-1, train_data.shape[-1])
		self.train_labels = np.array([-1]*train_data.shape[0] + [1]*train_data.shape[1])
		self.train_labels = self.train_labels[:, None]

		self.K = self.kernel(self.train_data, self.train_data)
		self.K = np.concatenate((self.K, np.ones((self.K.shape[0], 1))), axis = 1)
		self.gradient_descent(self.K, self.train_labels)		
		final_weights = np.expand_dims(self.weights, -1)
		self.prediction = np.sign(np.dot(self.K, self.weights))
		if self.validate:
			self.predict("train")
			self.predict("test")

	def rbf_kernel(self, param):
		sigma = 1/(param ** 2)
		kernel = lambda X, y: np.exp(sigma * np.square(X[:, None] - y).sum(axis=2))
		return kernel

	def poly_kernel(self, param):
		powr = param
		kernel = lambda X, Y: np.power((np.matmul(X, y.T) + 1), powr)
		return kernel

	def compute_grad(self, X, Y):
		x = np.expand_dims(X, 0)
		y = np.expand_dims(Y, 0)
		print("xshape" ,x.shape)
		print("weights shape", self.weights.shape)
		print("yshape", y.shape)
		distance = 1 - (y*np.dot(x, self.weights))
		print("dist shape", distance.shape)
		distance[distance < self.eps] = 0
		dw = np.zeros((self.weights.shape[0],1))
		idx = np.where(distance > 0)[0]
		print("idx shape", len(idx))
		print(dw.shape, self.weights.shape)
		if(len(idx) == 0):
			dw += self.weights
		else:
			print((y*x).shape)
			dw += self.weights - self.margin*((y*x)).squeeze()

		gradient = dw/len(self.weights)
		return gradient		
		

	def gradient_descent(self, X, labels):
		self.weights = np.random.rand(X.shape[1],1)
		print('k shape', X.shape)
		print('labels shape', labels.shape)
		
		for epoch in range(self.epochs):
			idx = np.random.permutation(X.shape[0])
			X_shuffled = X[idx]
			labels_shuffled = labels[idx]
			weights = self.weights

			gradient = self.compute_grad(X_shuffled, labels_shuffled)
			self.weights = self.weights - self.lr*gradient
			if np.linalg.norm(gradient) <= self.eps and epoch > 10:
				print('Updates too smallL {np.linalg.norm(gradient)}, Exiting training')
				return

		if((epoch +1)%1000 == 0 or (epoch + 1)%5000 == 0) and (self.validate):
			self.get_accuracy("train")
			self.get_accuracy("test")

							

if __name__=="__main__":
	preprocess = "pca"
	dsplit = 0.8
	threshold = 0.8
	train_data, train_labels, test_data, test_labels = importMyData("dD", 2, preprocess, threshold, dsplit)
	model = SVM(train_data, train_labels, test_data, test_labels, eps = 1e-05, epochs = 10000, lr=0.01, margin=10000,
			seed=True, validate=True, kparam=0.2)
	model.fit()
	model.predict("train")
	model.predict("test")