**********************************CECS 550 PATTERN RECONITION PROJECT-1****************************************
                                       GROUP MEMBERS - Kiran M Gowda (018761559) 
													   Karthik Ganduri (018779902)

import csv
import random
import math
import pandas as pd
import numpy as np

class Diabetes:

	def separateByClass(self, dataset):
		separated = {}
		for i in range(len(dataset)):
			vector = dataset[i]
			if (vector[-1] not in separated):
				separated[vector[-1]] = []
			separated[vector[-1]].append(vector)
		return separated
	
	def mean(self,numbers):
		return sum(numbers)/float(len(numbers))

	def stdev(self,numbers):
		avg = self.mean(numbers)
		variance = sum([pow(x-avg,2) for x in numbers])/float(len(numbers)-1)
		return math.sqrt(variance)
		
	def summarize(self, dataset):
		summaries = [(self.mean(attribute), self.stdev(attribute)) for attribute in zip(*dataset)]
		del summaries[-1]
		return summaries
		
	def summarizeByClass(self,dataset):
		separated = self.separateByClass(dataset)
		summaries = {}
		for classValue, instances in separated.items():
			summaries[classValue] = self.summarize(instances)
		return summaries
		
	
	def calculateProbability(self,x, mean, stdev):
		exponent = math.exp(-(math.pow(x-mean,2)/(2*math.pow(stdev,2))))
		return (1 / (math.sqrt(2*math.pi) * stdev)) * exponent
		
	def calculateClassProbabilities(self,summaries, inputVector):
		probabilities = {}
		for classValue, classSummaries in summaries.items():
			probabilities[classValue] = 1
			for i in range(len(classSummaries)):
				mean, stdev = classSummaries[i]
				x = inputVector[i]
				probabilities[classValue] *= self.calculateProbability(x, mean, stdev)
		return probabilities
		
	def predict(self,summaries, inputVector):
		probabilities = self.calculateClassProbabilities(summaries, inputVector)
		bestLabel, bestProb = None, -1
		for classValue, probability in probabilities.items():
			if bestLabel is None or probability > bestProb:
				bestProb = probability
				bestLabel = classValue
		return bestLabel
		
	def getPredictions(self,summaries, testSet):
		predictions = []
		for i in range(len(testSet)):
			result = self.predict(summaries, testSet[i])
			predictions.append(result)
		return predictions
		
	def getAccuracy(self,testSet, predictions):
		correct = 0
		for x in range(len(testSet)):
			if testSet[x][-1] == predictions[x]:
				correct += 1
		return (correct/float(len(testSet))) * 100.0
		
	def confusion_matrix(self, true, pred):
		k = len(np.unique(true))
		res = np.zeros((k,k))
		for i in range(len(true)):
			j = true[i]
			k = pred[i]
			res[j][k]+=i 
		return res
		
if __name__ =="__main__":
	db1 = Diabetes()
	table_header = ['pregnant', 'glucose', 'bp', 'skin', 'insulin', 'bmi', 'pedigree', 'age','is_diabetes']
   
	testing = pd.read_csv(r'C:\Users\KiranGM\Desktop\CECS 550 Projects\Project1Data\data\test.csv', header=None, names=table_header)
	training = pd.read_csv(r'C:\Users\KiranGM\Desktop\CECS 550 Projects\Project1Data\data\train.csv', header = None, names = table_header)
	
	y = list(map(int, testing.is_diabetes))
	
	testing = testing.values.tolist()
	training = training.values.tolist()
	
	print('training = {0} rows and testing = {1} rows'.format(len(training), len(testing)))
	
	summaries = db1.summarizeByClass(training)
	predictions = db1.getPredictions(summaries, testing)
	accuracy = db1.getAccuracy(testing, predictions)
	print('Accuracy by counting classified and misclassified points: {0}%'.format(accuracy))
	
	
	t = pd.Series(y, name = 'Original')
	p = pd.Series(predictions, name = 'Prediction')
	
	confusion = pd.crosstab(t, p, rownames = ['Original'], colnames = ['Prediction'], margins = True)
	print(confusion)
	
	confusion = db1.confusion_matrix(y,list(map(int, predictions)))
	
	TP = confusion[1, 1]
	TN = confusion[0, 0]
	FP = confusion[0, 1]
	FN = confusion[1, 0]
	
	Accuracy = ((TP + TN) / (TP + FP + TN +FN)*100)
	
	print('Accuracy by using confusion matrix: {0}%'.format(Accuracy))
	
	Error = ((FP + FN) / (TP + FP + TN +FN)*100)
	
	print('Classifier Error: {0}%'.format(Error))
    
	Sensitivity = (TP / (FN + TP)* 100)
	
	print('Classifier Sensitivity: {0}%'.format(Sensitivity))
	
	
	
	
	
 

	
	
		
	
