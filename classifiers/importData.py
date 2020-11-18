import scipy.io


def importMyData():
	datasetData = scipy.io.loadmat('../datasets/data.mat')
	illumData = scipy.io.loadmat('../datasets/illumination.mat')
	poseData = scipy.io.loadmat('../datasets/pose.mat')
	# print(illumData)
	datasetData = datasetData['face']
	illumData = illumData['illum']
	poseData = poseData['pose']
	return datasetData, illumData, poseData

if __name__=="__main__":
	dD, iD, pD = importMyData()