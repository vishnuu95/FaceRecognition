from importData import importMyData
import numpy as np
from abc import ABC
from scipy.stats import multivariate_normal
from sklearn.covariance import LedoitWolf

from mainFile import *

class Bayes(Classifier):
	gauss = {}
	def __init__(self, train_data, train_labels, n_classes, n_samples, n_features):
		self.trainData = train_data.reshape(n_classes, n_samples, n_features)
		self.trainLabels = train_labels.reshape(n_classes, n_samples)
		for i in range(n_classes):
			mu = np.mean(self.trainData[i,:,:],axis=0)[None,:]
			x_mu = self.trainData[i,:,:].T - mu.T
			cov = np.matmul(x_mu, x_mu.T)/n_samples
			# print(cov.shape)
			X = np.random.multivariate_normal(mean = mu[0,:], cov = cov, size= n_features)
			# print(X[None, :])
			# input()
			covLW = LedoitWolf().fit(X)
			cov = covLW.covariance_
			# print(cov.shape)
			self.gauss[i] = [mu, cov]
			# print(np.matmul(self.trainData[i,:,:].T, self.trainData[i,:,:]))
			# input()
		# print(self.trainData.shape, self.trainLabels.shape)
		# for i in range(train_labels):
		# 	if i in gauss:
		# 		gauss[i] = 	
		# 	else:
		# 		gauss[i] = [train_data[], ]	

	def validate(self, sample, label):
		maxprob = 0
		idx = -1
		probs = np.array([])
		for i in self.gauss:
			# print(self.gauss[i][0][0,:].shape, self.gauss[i][1].shape)
			# print(multivariate_normal.pdf(sample, self.gauss[i][0][0,:], self.gauss[i][1]))
			# if (multivariate_normal.pdf(sample, self.gauss[i][0][0,:], self.gauss[i][1]) > maxprob):
			# 	maxprob = multivariate_normal.pdf(sample, self.gauss[i][0][0,:], self.gauss[i][1])
			# 	idx = i
			probs = np.append(probs, multivariate_normal.pdf(sample, self.gauss[i][0][0,:], self.gauss[i][1]))
		# print(np.sum(probs))	
		probs = probs/np.sum(probs)
		print(probs)
		maxprob = 0
		for i in range(probs.shape[0]):
			if probs[i] > maxprob:
				maxprob = probs[i]
				idx = i	

			# cov = self.gauss[i][1]
			# x = sample
			# mean = self.gauss[i][0]
			# k = cov.shape[0]
			# # print(np.linalg.det(cov))
			# det = np.linalg.det(cov)
			# const = (2 * np.pi) ** (-k/2) * det ** (-1/2)
			# x = np.expand_dims(x, -1)
			# mean = np.expand_dims(mean, -1)
			# diff = x - mean
			# power = (-1/2) * np.dot(np.dot(diff.T, np.linalg.inv(cov)), diff)
		print(maxprob, idx, label)	
		if idx == -1:
			print(" All probs were 0!")
			return False
		elif idx == label:
			return True	


if __name__=="__main__":
	data_params = {1:	
					{'dD':
						{'classes':200,
						 'samples':3
						},
					 'iD':
						{'classes':68,
						 'samples':21
						},
					 'pD':
						{
						 'classes':68,
						 'samples':13
						}
					},
				   2:
				   	{'dD':
						{'classes':2,
						 'samples':200
						}
				  	}
				  }


	dsplit = 0.85
	threshold = 0.7
	task = 1 # 1 - person idenfitication, 2 - expression vs nuetral
	data_choice = 'dD' # dD, iD, pD
	preprocess = 'pca'
	data_train, labels_train, data_test, labels_test = importMyData(data_choice, task, preprocess, threshold, dsplit)
	model = Bayes(data_train, labels_train, data_params[task][data_choice]['classes'],
							int(data_train.shape[0]/data_params[task][data_choice]['classes']), data_train.shape[-1])

	correct_preds = 0
	for i in range(data_test.shape[0]):
		# print(data_test[i].shape , labels_test[i].shape)
		# input()
		if(model.validate(data_test[i],labels_test[i])):
			correct_preds += 1
	print(" Bayes accuracy: ", correct_preds/data_test.shape[0])		