#!/usr/bin/env python3
import os
import numpy as np
import sklearn.preprocessing as prep
def file_to_Array(filename,path,mode=0):
	file=open(path+filename,"r")
	lines =file.readlines()
	labels=[]
	features=[]
	for line in lines:
		line=line[:-1]
		feature=line.split("\t")
		label=int(feature[0])
		feature=feature[1:]
		features+=[feature]
		labels+=[label]
	labels=np.array(labels).astype(float)
	if mode!=0:
		labels[labels==-1]=0
	return labels,np.array(features).astype(float)
# def norm(X,order):
# 	X_norm=np.divide(X,np.linalg.norm(X,ord=order,axis=0))
# 	X_norm=np.nan_to_num(X_norm)
	return X_norm
def norm(X,order):
	order='l'+str(order)
	X_norm=prep.normalize(X,norm=order, axis=0)
	# print(X_norm)
	return X_norm
def sigmoid(X):
	return (1/(1+np.exp(-X)))
# labels,features=file_to_Array("bclass-train",desktop_path+"/ML_Project/")
# print (features,labels)