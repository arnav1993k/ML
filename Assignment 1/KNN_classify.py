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
	c_out=[]
	# classifier=neighbors.KNeighborsClassifier(n_neighbors=k)
	# classifier.fit(X,y.ravel())
	# score=classifier.score(x_test,y_test.ravel())
	
	for test in x_test:
		classes=Counter()
		predictions=Counter()
		distances=knn.get_distances(X,y,test)
		print("Applying model averaging for k = 1 to "+str(k))
		for j in range(1,k+1):
			classes=knn.get_classes(distances,y,j)
			#implementing model averaging
			predictions[classes.most_common(1)[0][0]]+=1
		c_out+=[predictions.most_common(1)[0][0]]
		#print(predictions)
	# df=pd.DataFrame()
	# df['Actual']=y_test.astype(int).ravel()
	# df['Predicted']=np.array(c_out,dtype=np.int16).ravel()
	# df['Error']=df['Actual']-df['Predicted']
	# total_error=(df['Error']!=0).sum()
	#df.to_csv(desktop_path+"/"+label+"_knn.csv")
	y_test=y_test.astype(int).ravel()
	c=np.array(c_out,dtype=np.int16).ravel()
	total_accuracy=metrics.accuracy_score(y_test,c)

	#print(total_accuracy,score)
	#percentage_error=total_error/y_test.shape[0]
	#print("The "+label+" error is "+str()+" percent.")
	return total_accuracy*100
def main():
	desktop_path=os.path.join(os.path.expanduser('~'), 'Desktop')
	training_path=desktop_path+"/trainingDigits"
	testing_path=desktop_path+"/testDigits"
	util=Utilities()
	features,outputs=util.converttoVector(training_path)
	test_feature,test_output=util.converttoVector(testing_path)
	knn=KNN()
	errors_train=[]
	errors_test=[]
	errors_train+=[findError(features,outputs,features,outputs,"Training",knn,10)]
	errors_test+=[findError(features,outputs,test_feature,test_output,"Testing",knn,10)]
	print(errors_train)
	print(errors_test)
	# plt.plot(errors_train)
	# plt.plot(errors_test)
	# plt.show()
#main
if __name__ == '__main__':
	main()