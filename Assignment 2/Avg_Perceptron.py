#!/usr/bin/env python3
import os
import numpy as np
import Utilities
import math
from matplotlib import pyplot as plt
class Perceptron:
	"""docstring for ClassName"""
	def __init__(self,w,alpha):
		self.weights=np.zeros(w)
		self.alpha=alpha
	def fit(self,X,y,e):
		err=1
		j=0
		error_rate=[]
		cached_weights=self.weights
		counter=1
		first=0
		while(j<2000):
			err=0
			for i in range(len(X)):
				p=np.dot(X[i],self.weights).ravel()
				if p>=0:
					z=1
				else:
					z=-1
				if y[i] != z:
					d=X[i]*(y[i])
					err+=1
					self.weights=self.weights+self.alpha*d
					cached_weights=cached_weights+self.alpha*d*counter
				counter+=1
			j+=1
			error_rate+=[[j,err]]
			if err==0 and first==0:
				first=j
		self.weights-=cached_weights/counter

		return first,np.array(error_rate)
	def predict(self,X):
		y=np.matmul(X,self.weights)
		y[y>=0]=1
		y[y<0]=-1
		return y
	def find_error(self,X,y):
		z=self.predict(X)
		err=np.sum(np.absolute(y-z))/(2*len(y))
		return err

def __main__():
	desktop_path=os.path.join(os.path.expanduser('~'), 'Desktop')
	#Declare new perceptron
	p1=Perceptron(35,1)
	#fetch data in form of array
	y_train,X_train=Utilities.file_to_Array("bclass-train",desktop_path+"/ecen765/Assignment 3/")
	y_test,X_test=Utilities.file_to_Array("bclass-test",desktop_path+"/ecen765/Assignment 3/")
	#make adjust for bias
	o_train=np.ones((len(y_train),1))
	X_train=np.concatenate((o_train,X_train),axis=1)
	o_test=np.ones((len(y_test),1))
	X_test=np.concatenate((o_test,X_test),axis=1)
	#fit perceptron to data for non normalized case
	iterations,error_rate1=p1.fit(X_train,y_train,0.01)
	print("Training convergence time is "+str(iterations)+" iterations.")
	print("Testing error is "+str(p1.find_error(X_test,y_test)))
	plt.plot(error_rate1[:,0],error_rate1[:,1],label="Non normalized "+str(iterations)+" iterations")
	plt.legend()
	#L1 norm
	p2=Perceptron(35,1)
	X_norm1_train=Utilities.norm(X_train,1)
	X_norm1_test=Utilities.norm(X_test,1)
	iterations,error_rate2=p2.fit(X_norm1_train,y_train,0.01)
	print("Training convergence time is "+str(iterations)+" iterations.")
	print("Testing error is "+str(p2.find_error(X_norm1_test,y_test)))
	plt.plot(error_rate2[:,0],error_rate2[:,1],label="1D Norm "+str(iterations)+" iterations")

	# #L2 norm
	p3=Perceptron(35,1)
	X_norm2_train=Utilities.norm(X_train,2)
	X_norm2_test=Utilities.norm(X_test,2)
	iterations,error_rate3=p3.fit(X_norm2_train,y_train,0.01)
	print("Training convergence time is "+str(iterations)+" iterations.")
	print("Testing error is "+str(p3.find_error(X_norm2_test,y_test)))
	plt.plot(error_rate3[:,0],error_rate3[:,1],label="2D Norm "+str(iterations)+" iterations")
	plt.legend()
	plt.show()
if __name__=="__main__":
	__main__()

			
