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
		          
		    else : 
		        if (curr_count > max_count):  
		            max_count = curr_count 
		            res = neighbors[i - 1] 
		          
		        curr_count = 1
		  
		# If last element is most frequent 
		if (curr_count > max_count): 
		  
		    max_count = curr_count 
		    res = neighbors[n - 1] 
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

def performPCA(data, threshold):
	data_flat = data.reshape((-1, data.shape[-1]))
	data_flatT = data_flat.T
	for i in range(data_flatT.shape[1]):
		data_flatT[:,i] = data_flatT[:,i] - np.mean(data_flatT[:,i])

	data_covarinace = np.matmul(data_flatT.T, data_flatT)
	P, D, Pinv = np.linalg.svd(data_covarinace)
	Dsum = np.sum(D)
	csum = D[0]/Dsum
	idx = 0
	var_contrib_cumm = np.array([csum])
	for i in range(1,D.shape[0]):
		if(csum > threshold):
			idx = i
			break
		csum += D[i]/Dsum
		var_contrib_cumm = np.append(var_contrib_cumm, [csum])
	new_data = np.matmul(data_flatT, P[:,0:i])
	# print(data.shape)
	return new_data
	
if __name__=="__main__":
	np.random.seed(0)
	dsplit = 0.8
	knn = 3
	dD, iD, pD = importMyData()
	threshold = 0.1

	data_choice = 'iD' # dD, iD, pD
	task = 1 # 1 - person idenfitication, 2 - expression vs nuetral
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

	nclasses = data_params[task][data_choice]['classes']
	nsamples = data_params[task][data_choice]['samples']			
	data = None
	if data_choice == 'dD':
		data = performPCA(dD, threshold)
	# 	nclasses = data_params[task]['dD']['classes']
	# 	nsamples = data_params[task]['dD']['samples']
	if data_choice == 'iD':
		data = performPCA(iD, threshold)
	# 	nclasses = data_params[task]['iD']['classes']
	# 	nsamples = data_params[task]['iD']['samples']
	if data_choice == 'pD':
		data = perfromPCA(pD, threshold)
	# 	nclasses = data_params[task]['pD']['classes']
	# 	nsamples = data_params[task]['pD']['samples']

	if task == 1:
		data = data.reshape((nclasses, nsamples, data.shape[-1]))
		labels = np.zeros((nclasses,nsamples))
		shuffle_data = np.empty_like(data)
		# train_label = np.empty_like(labels)
		for i in range(nclasses):
			labels[i,:] = np.array(nsamples*[i])
			idx = np.random.permutation(nsamples)
			shuffle_data[i, :, :] = data[i, idx, :]
		# print(shuffle_data.shape)
		data_train = shuffle_data[:,0:int(0.67*nsamples),:]
		data_train = data_train.reshape((-1, data_train.shape[-1]))
		print(data_train.shape)
		labels_train = labels[:,0:int(0.67*nsamples)].reshape(-1)
		print(labels_train.shape)
		data_test = shuffle_data[:,int(0.67*nsamples):,:]
		data_test = data_test.reshape(-1,data_train.shape[-1])
		print(data_test.shape)
		labels_test = labels[:,int(0.67*nsamples):]
		print(labels_test.shape)
		knn = 3
		# data_flat= data.reshape((24,21,600))
		# labels_flat = labels.reshape((24,21,600))
		# # idx = np.random.permutation(data_flat.shape[2])

		# data_flat = data_flat[idx]
		# labels_flat = labels_flat[idx]
		# training_data = data_flat[:dsplit*data.shape[2]]
		# training_label = data_flat[:dsplit*data.shape[2]]
		# test_data = data_flat[dsplit*data.shape[2],:]
		# test_labels = labels_cl[dsplit*data.shape[2],:]
	if task == 2:
		data = dD.reshape((dD.shape[0],dD.shape[1],3,200))
		labels = np.zeros((3,200))
		for i in range(3):
			labels[i,:] = np.array(200*[i])
		data_flat = data[:,:,0:2,:].reshape((-1,400))
		data_flat = data_flat.T
		
		labels_flat = labels[0:2,:].reshape((-1, 400))
		labels_flat = labels_flat.T
		
		idx = np.random.permutation(data_flat.shape[0])
		data_train = data_flat[:int(dsplit*data_flat.shape[0]),:]
		labels_train = labels_flat[:int(dsplit*data_flat.shape[0])]
		data_test = data_flat[int(dsplit*data_flat.shape[0]):,:]
		labels_test = labels_flat[int(dsplit*data_flat.shape[0]):]
		print(data_train.shape, labels_train.shape, data_test.shape, labels_test.shape)

	model = KNN(data_train, labels_train, knn)

	correct_preds = 0
	for i in range(data_test.shape[0]):
		# print(data_test[i].shape , labels_test[i].shape)
		if(model.validate(data_test[i],labels_test[i])):
			correct_preds += 1

	print(" KNN accuracy: ", correct_preds/data_test.shape[0])