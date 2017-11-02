#!/usr/bin/env python3
import os
import numpy as np
import Utilities
import math
from matplotlib import pyplot as plt
class LW_Logistic_Regression(object):
	"""docstring for Logistic_Regression"""
	def __init__(self,w,lamda):
		self.theta=np.zeros(w)
		self.lamda=lamda
	def getweight(x,X,tau):
		return (np.exp(-(np.power(x-X,2)/(2*tau**2))))
	def loss_fcn(X,y):
		loss=0
		w=np.ones(len(y))
		for i in range(len(y)):
			z=self.calc(X[i])
			loss+=w[i]*(y[i]*np.log(z)+(1-y[i])*np.log(1-z))
		loss-=self.lamda*np.dot(self.theta.T,self.theta)
	def calc(self,X):
		p=np.dot(X,self.theta)
		z=Utilities.sigmoid(p)
		return z
	def cost(self,X,y):
		z=self.calc(X)
		z[z>=0.5]=1
		z[z<0.5]=0
		err=y-z
		return err
	def getHessian(self,X,W,err):
		right_part=np.dot(X.T,err)
		wx=np.dot(W,X)
		# print(W,wx)
		xtwx=np.dot(X.T,wx)
		# print(xtwx)
		inv_xtwx=np.linalg.inv(xtwx)
		h=np.dot(inv_xtwx,right_part)
		return h
		# print(self.theta)
	def fit(self,X,y):
		i=0
		error_rate=[]
		total_error=1
		while(total_error>0):
			z=self.calc(X)
			W=np.diag(z*(1-z))
			err=self.cost(X,y)
			# print(W,err)
			H=self.getHessian(X,W,err)
			self.theta-=H
			i+=1
			total_error=(np.sum(np.absolute(err))/len(y))
			print(total_error)
			error_rate+=[[i,total_error]]
		# print(self.alpha,self.theta)
		return i,np.array(error_rate)
	def predict(self,X):
		z=self.calc(X)
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
	X_train=np.delete(X_train,[1],axis=1)
	X_test=np.delete(X_test,[1],axis=1)
	# print(X_train)
	#make adjust for bias
	o_train=np.ones((len(y_train),1))
	X_train=np.concatenate((o_train,X_train),axis=1)
	o_test=np.ones((len(y_test),1))
	X_test=np.concatenate((o_test,X_test),axis=1)
	#Declare new perceptron
	p1=LW_Logistic_Regression(34,1)
	iterations,error_rate1=p1.fit(X_train,y_train)
	print("Training convergence time is "+str(iterations)+" iterations with error as "+str(error_rate1[-1][1]))
	print("Testing error is "+str(p1.find_error(X_test,y_test)))
	plt.plot(error_rate1[:,0],error_rate1[:,1],label="Non normalized")
	# plt.legend()
	#L1 norm
	p2=LW_Logistic_Regression(34,1)
	X_norm1_train=Utilities.norm(X_train,1)
	X_norm1_test=Utilities.norm(X_test,1)
	iterations,error_rate2=p2.fit(X_norm1_train,y_train)
	print("Training convergence time is "+str(iterations)+" iterations with error as "+str(error_rate2[-1][1]))
	print("Testing error is "+str(p2.find_error(X_norm1_test,y_test)))
	plt.plot(error_rate2[:,0],error_rate2[:,1],label="1D Norm")

	# # #L2 norm
	p3=LW_Logistic_Regression(34,1)
	X_norm2_train=Utilities.norm(X_train,2)
	X_norm2_test=Utilities.norm(X_test,2)
	iterations,error_rate3=p3.fit(X_norm2_train,y_train)
	print("Training convergence time is "+str(iterations)+" iterations with error as "+str(error_rate3[-1][1]))
	print("Testing error is "+str(p3.find_error(X_norm2_test,y_test)))
	plt.plot(error_rate3[:,0],error_rate3[:,1],label="2D Norm")
	plt.legend()
	plt.show()
if __name__=="__main__":
	__main__()