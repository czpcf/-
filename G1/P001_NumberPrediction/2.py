import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from scipy.io import loadmat
from sklearn.preprocessing import OneHotEncoder
from scipy.optimize import minimize
from sklearn.metrics import classification_report###评价报告
"""
输入数据为20*20的灰度图像，标签为1~10
只有一层hidden layer
x:样本
y:标签
m:样本数
th:参数
"""
size_input=20*20
size_hidden=30
size_labels=10
alpha=1###就是lambda

#######################################sigmoid函数
def sigmoid(x):
	return 1/(1+np.exp(-x))
#######################################fp
def forprop(x,th1,th2):
	m=x.shape[0]
	a1=np.insert(x,0,values=np.ones(m),axis=1)###添加偏置项
	z2=a1*th1.T
	a2=np.insert(sigmoid(z2),0,values=np.ones(m),axis=1)###添加偏置项
	z3=a2*th2.T
	h=sigmoid(z3)
	return a1,z2,a2,z3,h###现在的h的列数等于size_labels
#######################################预测
def predict(x,th1,th2):
	a1,z2,a2,z3,h=forprop(x,th1,th2)
	print(np.matrix(np.argmax(h,axis=1)))
#######################################获得参数
def get(path):
	file=open(path,"r")
	fmin=file.read().split()
	for i in range(len(fmin)):
		fmin[i]=float(fmin[i])
	th1=np.matrix(np.reshape(fmin[:size_hidden*(size_input+1)],(size_hidden,(size_input+1))))
	th2=np.matrix(np.reshape(fmin[size_hidden*(size_input+1):],(size_labels,(size_hidden+1))))
	return th1,th2
#######################################主函数
def main():
#	th11,th12=get("save.txt")
#	th21,th22=get("save1.txt")
#	th31,th32=get("save2.txt")
#	th41,th42=get("save3.txt")
#	th51,th52=get("save4.txt")
	th61,th62=get("save++.txt")
	while(True):
		G=input().split(",")
		x=np.matrix(np.zeros((1,size_input)))
		for i in range(size_input):
			x[0,i]=int(G[i])
#		a11,z12,a12,z13,h1=forprop(x,th11,th12)
#		a21,z22,a22,z23,h2=forprop(x,th21,th22)
#		a31,z32,a32,z33,h3=forprop(x,th31,th32)
#		a41,z42,a42,z43,h4=forprop(x,th41,th42)
#		a51,z52,a52,z53,h5=forprop(x,th51,th52)
		a61,z62,a62,z63,h6=forprop(x,th61,th62)
		for i in range(size_labels):
			print("  ",i,end=" ")
		print("")
#		for i in range(size_labels):
#			print(("%.2f"%h1[0,i],2)[0],end=" ")
#		print("")
#		for i in range(size_labels):
#			print(("%.2f"%h2[0,i],2)[0],end=" ")
#		print("")
#		for i in range(size_labels):
#			print(("%.2f"%h3[0,i],2)[0],end=" ")
#		print("")
#		for i in range(size_labels):
#			print(("%.2f"%h4[0,i],2)[0],end=" ")
#		print("")
#		for i in range(size_labels):
#			print(("%.2f"%h5[0,i],2)[0],end=" ")
#		print("")
		for i in range(size_labels):
			print(("%.2f"%h6[0,i],2)[0],end=" ")
		print("")
#		y=np.matrix(np.argmax(h1,axis=1))
#		print("Most likely : ",y[0,0])
#		y=np.matrix(np.argmax(h2,axis=1))
#		print("              ",y[0,0])
#		y=np.matrix(np.argmax(h3,axis=1))
#		print("              ",y[0,0])
#		y=np.matrix(np.argmax(h4,axis=1))
#		print("              ",y[0,0])
#		y=np.matrix(np.argmax(h5,axis=1))
#		print("              ",y[0,0])
		
		y=np.matrix(np.argmax(h6,axis=1))
		print("              ",y[0,0])

if(__name__=="__main__"):
	main()
