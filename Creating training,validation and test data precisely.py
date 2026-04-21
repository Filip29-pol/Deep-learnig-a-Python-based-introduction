import numpy as np
from sklearn.datasets import make_classification
a,b=make_classification(n_samples=10000,weights=[0.9,0.1])
idx=np.where(b==0)[0]
x0=a[idx,:]
y0=b[idx]
print("x0: ",x0)
print("y0: ",y0)
idx=np.where(b==1)[0]
x1=a[idx,:]
y1=b[idx]
print("x1: ",x1)
print("y1: ",y1)

idx=np.argsort(np.random.random(y0.shape))
y0=y0[idx]
x0=x0[idx]
print("y0: ",y0)
print("x0: ",x0)
idx=np.argsort(np.random.random(y1.shape))
y1=y1[idx]
x1=x1[idx]
print("y1: ",y1)
print("x1: ",x1)

ntrn0=int(0.9*x0.shape[0])
ntrn1=int(0.9*x1.shape[0])
print("ntrn0: ",ntrn0)
print("ntrn1: ",ntrn1)
xtrn=np.zeros((int(ntrn0+ntrn1),20))
ytrn=np.zeros(int(ntrn0+ntrn1))
print("xtrn: ",xtrn)
print("ytrn: ",ytrn)
xtrn[:ntrn0]=x0[:ntrn0]
xtrn[ntrn0:]=x1[:ntrn1]
ytrn[:ntrn0]=y0[:ntrn0]
ytrn[ntrn0:]=y1[:ntrn1]
print("xtrn: ",xtrn)
print("ytrn: ",ytrn)
n0=int(x0.shape[0]-ntrn0)
n1=int(x1.shape[0]-ntrn1)
xval=np.zeros((int(n0/2+n1/2),20))
yval=np.zeros(int(n0/2+n1/2))
xval[:(n0//2)]=x0[ntrn0:(ntrn0+n0//2)]
xval[(n0//2):]=x1[ntrn1:(ntrn1+n1//2)]
yval[:(n0//2)]=y0[ntrn0+n0//2]
yval[(n0//2):]=y1[ntrn1+n1//2]
print("xval: ",xval)
print("yval: ",yval)
xtst=np.concatenate((x0[(ntrn0+n0//2):],x1[(ntrn1+n1//2):]))
ytst=np.concatenate((y0[(ntrn0+n0//2):],y1[(ntrn1+n1//2):]))
print("xtst: ",xtst)
print("ytst: ",ytst)