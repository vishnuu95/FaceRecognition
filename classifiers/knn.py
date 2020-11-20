from importData import importMyData
import numpy as np
from abc import ABC 

from mainFile import *

class KNN(Classifier):
	knn = 0
	def __init__(self, data, labels, knn):
		self.trainData = data
		self.trainLabels = labels
		self.knn = knn

	def validate(self, sample, label):	
		neighbors = self.closestKNeighbors(sample)
		neighbors = np.sort(neighbors)  
	    # find the max frequency using 
	    # linear traversal 
		max_count = 1
		res = neighbors[0] 
		curr_count = 1
		n = neighbors.shape[0] 
		for i in range(1, neighbors.shape[0]):  
		    if (neighbors[i] == neighbors[i - 1]): 
		        curr_count += 1
		          
		    else: 
		        if (curr_count > max_count):  
		            max_count = curr_count 
		            res = neighbors[i - 1] 
		          
		        curr_count = 1
		  
		# If last element is most frequent 
		if (curr_count > max_count): 
		  
		    max_count = curr_count 
		    res = neighbors[n - 1]
		print(label, neighbors)     
		if (res == label):		
			return True
		else:
			return False	

	def closestKNeighbors(self, sample):
		# print((self.trainData - sample).shape)	
		errors_feature_vec = np.linalg.norm(self.trainData - sample, axis=1)
		# print(errors_feature_vec)
		idx = np.argsort(errors_feature_vec)
		# idx = idx[::-1]
		return self.trainLabels[idx][0:self.knn]


if __name__=="__main__":
	np.random.seed(0)
	dsplit = 0.8
	knn = 1
	threshold = 0.9
	task = 1 # 1 - person idenfitication, 2 - expression vs nuetral
	data_choice = 'dD' # dD, iD, pD
	preprocess = 'pca'
	data_train, labels_train, data_test, labels_test = importMyData(data_choice, task, preprocess, threshold, dsplit)
	# print(data_train.shape, labels_train.shape, data_test.shape, labels_test.shape)

	model = KNN(data_train, labels_train, knn)

	correct_preds = 0
	for i in range(data_test.shape[0]):
		# print(data_test[i].shape , labels_test[i].shape)
		# input()
		if(model.validate(data_test[i],labels_test[i])):
			correct_preds += 1

	print(" KNN accuracy: ", correct_preds/data_test.shape[0])