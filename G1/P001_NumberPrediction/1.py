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
alpha=3###就是lambda

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
#######################################cost函数
def cost(th1,th2,size_input,size_hidden,size_labels,x,y,alpha):
	m=x.shape[0]
	x=np.matrix(x)
	y=np.matrix(y)
	a1,z2,a2,z3,h=forprop(x,th1,th2)
	A=np.multiply((-y),np.log(h))###数乘
	B=np.multiply((1-y),np.log(1-h))
	reg=(np.sum(np.power(th1[:,1:],2))+np.sum(np.power(th2[:,1:],2)))*alpha/(2*m)
	return np.sum(A-B)/m+reg
#######################################sigmoid导数
def sigmoidGradient(x):
	return np.multiply(sigmoid(x),1-sigmoid(x))
#######################################随机初始化
def random():
	return (np.random.random(size=size_hidden*(size_input+1)+size_labels*(size_hidden+1))-0.5)*0.24
#######################################bp

TOT=0
def backprop(th,size_input,size_hidden,size_labels,x,y,alpha):
	global TOT
	m=x.shape[0]
	print(TOT)
	TOT=TOT+1
	x=np.matrix(x)
	y=np.matrix(y)
	th1=np.matrix(np.reshape(th[:size_hidden*(size_input+1)],(size_hidden,(size_input+1))))###请注意
	th2=np.matrix(np.reshape(th[size_hidden*(size_input+1):],(size_labels,(size_hidden+1))))
	a1,z2,a2,z3,h=forprop(x,th1,th2)
	A=np.multiply((-y),np.log(h))
	B=np.multiply((1-y),np.log(1-h))
	reg=(np.sum(np.power(th1[:,1:],2))+np.sum(np.power(th2[:,1:],2)))*alpha/(2*m)
	J=np.sum(A-B)/m+reg
	
	del1=np.zeros(th1.shape)
	del2=np.zeros(th2.shape)
	
	for t in range(m):
		a1t=a1[t,:]###(1,401)
		z2t=z2[t,:]###(1,25)
		a2t=a2[t,:]###(1,26)
		ht=h[t,:]  ###(1,10)
		yt=y[t,:]  ###(1,10)
		
		d3t=ht-yt
		z2t=np.insert(z2t,0,values=np.ones(1))###(1,26)，老传统
		d2t=np.multiply((th2.T*d3t.T).T,sigmoidGradient(z2t))
		
		del1=del1+(d2t[:,1:]).T*a1t
		del2=del2+d3t.T*a2t
	del1=del1/m
	del2=del2/m
	
	del1[:,1:]=del1[:,1:]+(th1[:,1:]*alpha)/m
	del2[:,1:]=del2[:,1:]+(th2[:,1:]*alpha)/m
	
	tmp=np.concatenate((np.ravel(del1),np.ravel(del2)))
	
	return J,tmp
#######################################获得数据
maxlen=6
def getStr(x):
	y=maxlen-len(str(x))
	return "0"*y+str(x)

def dataNoise(x,c):
	for i in range(c):
		x[np.random.randint(0,size_input)]=0
	return x

def getdata():
	file=open("dig/NUM.txt")
	m=int(file.read())
	x=np.zeros((m*4,size_input))
	y=np.zeros((m*4,size_labels))
	tot=0
	for i in range(m):
		path="dig/"+getStr(i)+".txt"
		file=open(path)
		Q=file.read()
		A=Q.split(",")
		for j in range(size_input):
			x[tot,j]=int(A[j])
		y[tot,int(A[size_input])]=1
		tot+=1
		
		X=x[tot-1]
		Y=y[tot-1]
		x[tot]=dataNoise(X,2)
		y[tot]=Y
		tot+=1
		
		X=x[tot-1]
		Y=y[tot-1]
		x[tot]=dataNoise(X,2)
		y[tot]=Y
		tot+=1
		
		X=x[tot-1]
		Y=y[tot-1]
		x[tot]=dataNoise(X,2)
		y[tot]=Y
		tot+=1
	m=m*4
	return x,y
#######################################预测
def predict(x,th1,th2):
	a1,z2,a2,z3,h=forprop(x,th1,th2)
	print(np.matrix(np.argmax(h,axis=1)))
#######################################主函数
def main():
	x,y=getdata()
	th=random()
	fmin=minimize(fun=backprop,x0=th,args=(size_input,size_hidden,size_labels,x,y,alpha),method='TNC',jac=True,options={'maxiter':300})
	G=fmin.x
	file=open("save++.txt","w")
	for i in range(G.shape[0]):
		file.write(str(G[i])+"\n")
	file.close()
	x = np.matrix(x)
	th1 = np.matrix(np.reshape(fmin.x[:size_hidden* (size_input + 1)], (size_hidden, (size_input + 1))))
	th2 = np.matrix(np.reshape(fmin.x[size_hidden * (size_input + 1):], (size_labels, (size_hidden + 1))))
	a1, z2, a2, z3, h = forprop(x, th1, th2 )
	y_pred = np.matrix(np.argmax(h, axis=1))
	m=x.shape[0]
	Y=np.zeros(m)
	for i in range(m):
		for j in range(size_labels):
			if(y[i,j]):
				Y[i]=j
	print(classification_report(Y,y_pred))

if(__name__=="__main__"):
	main()
