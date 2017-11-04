#!/usr/bin/env python3
import os
import numpy as np
import Utilities
import math
from matplotlib import pyplot as plt
class LW_Logistic_Regression(object):
	"""docstring for Logistic_Regression"""
	def __init__(self,w,lamda,tau,alpha):
		self.theta=np.zeros(w)
		self.lamda=lamda
		self.tau=tau
		self.alpha = alpha
	def getweight(self,x,X):
		return (np.exp(-(np.linalg.norm(x-X,2,axis=1)**2/(2*self.tau**2))))
	def calc(self,X):
		p=np.dot(X,self.theta)
		z=Utilities.sigmoid(p)
		return z
	def cost(self,X,y,mode="normal"):
		z=self.calc(X)
		# print(z)
		if mode=="discrete":
			if z>=0.5:
				z=1
			else:
				z=0
		err=y-z
		return err
	def getHessian(self,X,y,x):
		z=self.calc(X)
		W=self.getweight(x,X)*np.diag(z*(1-z))
		wx=np.dot(W,X)
		# print(W,wx)
		h=-np.dot(X.T,wx)-2*self.lamda
		# print(xtwx)
		return h
		# print(self.theta)
	def getGradient(self,X,y,x,err):
		weight=self.getweight(x,X)
		# print(weight)
		gradient=np.dot(X.T,weight*err)-2*self.lamda*self.theta
		return gradient

	def fit(self,X,y,x_test,y_test):
		i=0
		error_rate=[]
		total_error=1
		op=[]
		for x,y_t in zip(x_test,y_test):
			i=0
			self.theta=np.zeros(self.theta.shape[0])
			while(i<200):
				err=self.cost(X,y)
				# print(W,err)
				H=self.getHessian(X,y,x)
				g=self.getGradient(X,y,x,err)
				# print (x)
				# print (i)
				# print (H)
				# print (g)
				h_inv=np.linalg.pinv(H)
				self.theta-=self.alpha*np.dot(h_inv,g)
				i+=1
			z=self.calc(x)
			if z>=0.5:
				op+=[1]
			else:
				op+=[0]
		op=np.array(op).reshape((len(y_test)))
		total_error=(np.sum(np.absolute(y_test-op))/len(y_test))


		return total_error
		# total_error=(np.sum(np.absolute(err))/len(y_test))
		# print(total_error)
		# error_rate+=[[i,total_error]]
		# print(self.alpha,self.theta)
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
	tau=[0.01,0.05,0.1,0.5,1,1]
	plot_vector=[]
	for t in tau:
		p1=LW_Logistic_Regression(34,0.001,t,t/100)
		err1=p1.fit(X_train,y_train,X_test,y_test)
		# print(err1)
		# iterations,error_rate1=p1.fit(X_train,y_train)
		# print("Training convergence time is "+str(iterations)+" iterations with error as "+str(error_rate1[-1][1]))
		# print("Testing error is "+str(p1.find_error(X_test,y_test)))
		# plt.plot(error_rate1[:,0],error_rate1[:,1],label="Non normalized")
		# # plt.legend()
		#L1 norm
		p2=LW_Logistic_Regression(34,0.001,t,0.01)
		X_norm1_train=Utilities.norm(X_train,1)
		X_norm1_test=Utilities.norm(X_test,1)
		err2=p2.fit(X_norm1_train,y_train,X_norm1_test,y_test)
		# print(err2)
		# iterations,error_rate2=p2.fit(X_norm1_train,y_train)
		# print("Training convergence time is "+str(iterations)+" iterations with error as "+str(error_rate2[-1][1]))
		# print("Testing error is "+str(p2.find_error(X_norm1_test,y_test)))
		# plt.plot(error_rate2[:,0],error_rate2[:,1],label="1D Norm")

		# # #L2 norm
		p3=LW_Logistic_Regression(34,0.001,t,t/100)
		X_norm2_train=Utilities.norm(X_train,2)
		X_norm2_test=Utilities.norm(X_test,2)
		err3=p3.fit(X_norm2_train,y_train,X_norm2_test,y_test)
		plot_vector+=[t,err1,err2,err3]
	# iterations,error_rate3=p3.fit(X_norm2_train,y_train)
	# print("Training convergence time is "+str(iterations)+" iterations with error as "+str(error_rate3[-1][1]))
	# print("Testing error is "+str(p3.find_error(X_norm2_test,y_test)))
	# plt.plot(error_rate3[:,0],error_rate3[:,1],label="2D Norm")
	# plt.legend()
	# plt.show()
	plot_vector=np.array(plot_vector)
	plt.plot(plot_vector[:,0],plot_vector[:,1],label="Non normalaized case")
	plt.plot(plot_vector[:,0],plot_vector[:,1],label="L1 Norm")
	plt.plot(plot_vector[:,0],plot_vector[:,1],label="L2 Norm")
	plt.legend()
	plt.show()
if __name__=="__main__":
	__main__()