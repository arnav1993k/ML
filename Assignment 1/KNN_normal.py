#!/usr/bin/env python3
import os
import numpy as np
from matplotlib import pyplot as plt
from KNN import KNN
import sklearn.metrics as metrics
from Utilities import Utilities
from sklearn.decomposition import PCA
from collections import Counter
def findError(X,y,x_test,y_test,label,knn,k):
	c_out=np.zeros([len(y_test),(k+1)])
	# classifier=neighbors.KNeighborsClassifier(n_neighbors=k)
	# classifier.fit(X,y.ravel())
	# score=classifier.score(x_test,y_test.ravel())
	i=0
	for test in x_test:
		classes=Counter()
		distances=knn.get_distances(X,y,test)
		for j in range(1,k+1):
			classes=knn.get_classes(distances,y,j)
			c_out[i][j]=classes.most_common(1)[0][0]
		i+=1
	#print(c_out)
	y_test=y_test.astype(int).ravel()
	total_accuracy=np.ones(k+1)
	for i in range(1,k+1):
		c=np.array(c_out[:,i],dtype=np.int16).ravel()
		total_accuracy[i]=metrics.accuracy_score(y_test,c)
	#total_accuracy=np.array(total_accuracy)
	total_accuracy=total_accuracy[1:11]
	print(label+" error= "+str(1-total_accuracy))
	#percentage_error=total_error/y_test.shape[0]
	#print("The "+label+" error is "+str()+" percent.")
	return (1-total_accuracy)*100
def main():
	desktop_path=os.path.join(os.path.expanduser('~'), 'Desktop')
	training_path=desktop_path+"/trainingDigits"
	testing_path=desktop_path+"/testDigits"
	util=Utilities()
	features,outputs=util.converttoVector(training_path)
	test_feature,test_output=util.converttoVector(testing_path)
	knn=KNN()
	training_error=findError(features,outputs,features,outputs,"Training",knn,10)
	testing_error=findError(features,outputs,test_feature,test_output,"Testing",knn,10)
	# print(training_error)
	# print(testing_error)
	test=plt.plot(np.linspace(1,11,10),testing_error,'b',label="Testing Error")
	train=plt.plot(np.linspace(1,11,10),training_error,'r',label="Training Error")
	plt.legend()
	plt.xlabel("Least k distances")
	plt.ylabel("Percentage error")
	plt.show()
#main
if __name__ == '__main__':
	main()