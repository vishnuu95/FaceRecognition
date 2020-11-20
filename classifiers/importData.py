import scipy.io
import numpy as np
from sklearn.decomposition import PCA

def performPCA(data, threshold):
	data_flat = data.reshape((-1, data.shape[-1]))
	print(data_flat.shape) # 504x600
	pca = PCA(threshold).fit(data_flat.T)
	# print(pca.components_.shape)
	pcaToolKitData = pca.transform(data_flat.T)
	print(pcaToolKitData.shape)
	new_data = pcaToolKitData
	# # input()
	# print("data_flat shape", data_flat.shape)
	# data_flatT = data_flat.T # 600x504
	# data_norm = np.empty_like(data_flatT)
	# for i in range(data_flatT.shape[1]):
	# 	mu = np.mean(data_flatT[:,i])
	# 	std = np.std(data_flatT[:,i])
	# 	data_norm[:,i] = (data_flatT[:,i] - mu)/std
	# data_covariance = np.matmul(data_norm.T, data_norm)
	# # data_covariance = np.matmul(data_flatT.T, data_flatT)
	# # for i in range(data_covariance.shape[1]):
	# # 	data_covariance[:,i] = data_covariance[:,i] - np.mean(data_covariance[:,i])

	# # data_covariance = np.matmul(data_flatT.T, data_flatT)
	# P, D, Pinv = np.linalg.svd(data_covariance)
	# # print(D)
	# # print(P.shape, Pinv.shape)
	# Dsum = np.sum(D)
	# csum = D[0]/Dsum
	# idx = 0
	# var_contrib_cumm = np.array([csum])
	# for i in range(1,D.shape[0]):
	# 	if(csum > threshold):
	# 		idx = i
	# 		break
	# 	csum += D[i]/Dsum
	# 	var_contrib_cumm = np.append(var_contrib_cumm, [csum])
	# new_data = np.matmul(data_flatT, P[:,0:i])
	# print(new_data.shape)
	# input()
	return new_data

def importMyData(data_choice, task, preprocess = "pca", threshold = 0.6, dsplit=0.8):
	np.random.seed(21)
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

	datasetData = scipy.io.loadmat('../datasets/data.mat')
	illumData = scipy.io.loadmat('../datasets/illumination.mat')
	poseData = scipy.io.loadmat('../datasets/pose.mat')
	# print(illumData)
	dD = datasetData['face']
	iD = illumData['illum']
	pD = poseData['pose']

	nclasses = data_params[task][data_choice]['classes']
	nsamples = data_params[task][data_choice]['samples']	

	if data_choice == 'dD':
		if preprocess == "pca":
			data = performPCA(dD, threshold)
	# 	# nclasses = data_params[task]['dD']['classes']
	# # # 	nsamples = data_params[task]['dD']['samples']
	if data_choice == 'iD':
		iD = iD.reshape(-1,nsamples*nclasses)
		if preprocess == "pca":
			data = performPCA(iD, threshold)
	# # # 	nclasses = data_params[task]['iD']['classes']
	# # # 	nsamples = data_params[task]['iD']['samples']
	if data_choice == 'pD':
		pD = pD.reshape(-1,nsamples*nclasses)
		if preprocess == "pca":
			data = performPCA(pD, threshold)

	if task == 1:
		data = data.reshape((nclasses, nsamples, data.shape[-1]))
		print("1", data.shape)
		labels = np.zeros((nclasses,nsamples))
		shuffle_data = np.empty_like(data)
		# train_label = np.empty_like(labels)
		for i in range(nclasses):
			labels[i,:] = np.array(nsamples*[i])
			idx = np.random.permutation(nsamples)
			shuffle_data[i, :, :] = data[i, idx, :]
		# print("shuffle data shape", shuffle_data.shape)
		data_train = shuffle_data[:,0:int(0.67*nsamples),:]
		data_train = data_train.reshape((-1, data_train.shape[-1]))
		# print(data_train.shape)
		labels_train = labels[:,0:int(0.67*nsamples)].reshape(-1)
		# print(labels_train.shape)
		data_test = shuffle_data[:,int(0.67*nsamples):,:]
		data_test = data_test.reshape(-1,data_test.shape[-1])
		# print(data_test.shape)
		labels_test = labels[:,int(0.67*nsamples):].reshape(-1)
		# print(labels_test.shape)
		return data_train, labels_train, data_test, labels_test


	if task == 2:
		data = data.reshape((3, nsamples, data.shape[-1]))
		labels = np.zeros((3, nsamples))
		for i in range(3):
			labels[i,:] = np.array(nsamples*[i])
		data_flat = data[0:2,:,:].reshape((-1, data.shape[-1]))
		
		labels_flat = labels[0:2, :].reshape(-1)
		print(labels_flat)
		idx = np.random.permutation(data_flat.shape[0])
		# print(idx)
		data_flat = data_flat[idx, :]
		labels_flat = labels_flat[idx]
		# print(labels_flat)
		data_train = data_flat[:int(dsplit*data_flat.shape[0]),:]
		labels_train = labels_flat[:int(dsplit*data_flat.shape[0])]
		data_test = data_flat[int(dsplit*data_flat.shape[0]):,:]
		labels_test = labels_flat[int(dsplit*data_flat.shape[0]):]
		print(data_train.shape, labels_train.shape, data_test.shape, labels_test.shape)	

		return data_train, labels_train, data_test, labels_test

if __name__=="__main__":
	dD, iD, pD = importMyData()