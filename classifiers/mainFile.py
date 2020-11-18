from abc import ABC 

class Classifier(ABC):
	trainData = None
	trainLabels = None
	testData = None
	testLabels = None
	
	def __init__(train_data = None, train_labels = None):
		self.trainData = train_data
		self.trainLabels = train_labels

	def setTrainData(train_data, train_labels):
		self.trainData = train_data
		self.trainLabels = train_labels

	def setTestData(test_data, test_labels):
		self.testData = test_data
		self.testLabels = test_labels

	def train():
		pass

	# def validate():
	# 	pass