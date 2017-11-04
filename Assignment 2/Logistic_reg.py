#!/usr/bin/env python3
import os
import numpy as np
import Utilities
import math
from matplotlib import pyplot as plt
class Logistic_Regression(object):
	"""docstring for Logistic_Regression"""
	def __init__(self,w,alpha):
		self.theta=np.zeros(w)
		self.alpha=alpha
	def cost(self,X,y,mode="normal"):
		p=np.dot(X,self.theta)
		z=Utilities.sigmoid(p)
		if mode=="discrete":
			z[z>=0.5]=1
			z[z<0.5]=0
		err=y-z
		return err
	def gradient(self,X,y):
		err=self.cost(X,y)
		gradient=np.dot(X.T,err)
		self.theta+=self.alpha*gradient
		print(self.theta)
		# print(self.theta)
	def fit(self,X,y,e):
		i=0
		error_rate=[]
		total_error=1
		first=0
		while(i<15000):
			self.gradient(X,y)
			err=self.cost(X,y,"discrete")
			i+=1
			total_error=(np.sum(np.absolute(err))/len(y))
			error_rate+=[[i,total_error]]
			if (total_error==0 or i==15000) and first==0 :
				first=i
		# print(self.alpha,self.theta)
		return first,np.array(error_rate)
	def predict(self,X):
		p=np.dot(X,self.theta)
		z=Utilities.sigmoid(p)
		z[z>=0.5]=1
		z[z<0.5]=0
		return z
	def find_error(self,X,y):
		z=self.predict(X)
		err=y-z
		return (np.sum(np.absolute(err))/len(y))
def __main__():
	desktop_path=os.path.join(os.path.expanduser('~'), 'Desktop')
	#fetch data in form of array
	y_train,X_train=Utilities.file_to_Array("bclass-train",desktop_path+"/ecen765/Assignment 3/",1)
	y_test,X_test=Utilities.file_to_Array("bclass-test",desktop_path+"/ecen765/Assignment 3/",1)
	#make adjust for bias
	o_train=np.ones((len(y_train),1))
	X_train=np.concatenate((o_train,X_train),axis=1)
	o_test=np.ones((len(y_test),1))
	X_test=np.concatenate((o_test,X_test),axis=1)
	#Declare new perceptron
	p1=Logistic_Regression(35,0.1)
	iterations,error_rate1=p1.fit(X_train,y_train,0)
	print("Training convergence time is "+str(iterations)+" iterations with error as "+str(error_rate1[-1][1]))
	print("Testing error is "+str(p1.find_error(X_test,y_test)))
	plt.plot(error_rate1[:,0],error_rate1[:,1],label="Non normalized "+str(iterations)+" iterations")
	# plt.legend()
	#L1 norm
	p2=Logistic_Regression(35,10)
	X_norm1_train=Utilities.norm(X_train,1)
	X_norm1_test=Utilities.norm(X_test,1)

	print (X_norm1_train.shape)
	print (X_norm1_train)


	iterations,error_rate2=p2.fit(X_norm1_train,y_train,0.0)
	print("Training convergence time is "+str(iterations)+" iterations with error as "+str(error_rate2[-1][1]))
	print("Testing error is "+str(p2.find_error(X_norm1_test,y_test)))
	plt.plot(error_rate2[:,0],error_rate2[:,1],label="1D Norm "+str(iterations)+" iterations")

	# # #L2 norm
	p3=Logistic_Regression(35,.1)
	X_norm2_train=Utilities.norm(X_train,2)
	X_norm2_test=Utilities.norm(X_test,2)
	iterations,error_rate3=p3.fit(X_norm2_train,y_train,0.0)
	print("Training convergence time is "+str(iterations)+" iterations with error as "+str(error_rate3[-1][1]))
	print("Testing error is "+str(p3.find_error(X_norm2_test,y_test)))
	plt.plot(error_rate3[:,0],error_rate3[:,1],label="2D Norm "+str(iterations)+" iterations")
	plt.legend()
	plt.show()
if __name__=="__main__":
	__main__()