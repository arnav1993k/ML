#!/usr/bin/env python3
import os
import numpy as np
from matplotlib import pyplot as plt
from NBC import NBClassifier
from Gaussian_NB import Gaussian_NBC
import pandas as pd
import sklearn.metrics as metrics
from Utilities import Utilities
from sklearn.decomposition import PCA
def findError(X,y,label,nbc):
	c_out=[]
	for test in X:
		c_out+=[nbc.predict(test)]
	# df=pd.DataFrame()
	# df['Actual']=y.astype(int).ravel()
	# df['Predicted']=np.array(c_out,dtype=np.int16).ravel()
	# df['Error']=df['Actual']-df['Predicted']
	# total_error=(df['Error']!=0).sum()
	# df.to_csv(desktop_path+"/"+label+".csv")
	#percentage_error=total_error/(df.count+1)
	y=y.astype(int).ravel()
	c=np.array(c_out,dtype=np.int16).ravel()
	accuracy_score=metrics.accuracy_score(y,c)
	print("The "+label+" accuracy is "+str(accuracy_score)+" percent.")
	return accuracy_score*100

def main():
	desktop_path=os.path.join(os.path.expanduser('~'), 'Desktop')
	training_path=desktop_path+"/trainingDigits"
	testing_path=desktop_path+"/testDigits"
	util=Utilities()
	features,outputs=util.converttoVector(training_path)
	test_feature,test_output=util.converttoVector(testing_path)
	# nb=NBClassifier(0.000001)
	# nb.fit(features,outputs)

	# findError(features,outputs,"Training",nb)
	# findError(test_feature,test_output,"Testing",nb)
	testing_accuracy=[]
	training_accuracy=[]
	for k in range(1,10):
		gnb=Gaussian_NBC()
		pca=PCA(n_components=k)
		X=pca.fit_transform(features)
		gnb.fit(X,outputs)
		X_test=np.matmul(test_feature,np.transpose(pca.components_))
		training_accuracy+=[findError(X,outputs,"Training",gnb)]
		testing_accuracy+=[findError(X_test,test_output,"Testing",gnb)]
	plt.plot(testing_accuracy,'r')
	plt.plot(training_accuracy,'b')
	plt.show()
#main
if __name__ == '__main__':
	main()