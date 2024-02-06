# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 15:45:15 2024

@author: icon
"""
#supervised
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt

wbcd=pd.read_csv("C:/CSV files/wbcd.csv.xls")
# there are 569 rows and 32 columns
wbcd.describe()
#in output column there is only 8 for Benien and M for maliganat
wbcd["diagnosis"]=np.where(wbcd["diagnosis"]=="B", "Beniegn",wbcd["diagnosis"])

#in wbcd there is column named diagnosis, where there is "B replace with Benign
# similarly M with malignant
wbcd["diagnosis"]=np.where(wbcd["diagnosis"]=="M","Malignant",wbcd["diagnosis"])


wbcd=wbcd.iloc[:,1:32]

#normalization

def norm_func(i):
    x=(i-i.min())/(i.max()-i.min())
    return x
wbcd_n=norm_func(wbcd.iloc[:,1:32])


#let us now apply x as input and y as output

X=np.array(wbcd_n.iloc[:,:])
y=np.array(wbcd["diagnosis"])

# now split the data into training and testing

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)

#here you pass x,y instead dataframe
#there could be chance of unbalacing data
#so stratified sampling is used to split the sample data

from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=21)
knn.fit(X_train,y_train)
pred=knn.predict(X_test)
pred

#now let us evaluate the model
from sklearn.metrics import accuracy_score
print(accuracy_score(pred,y_test))
pd.crosstab(pred,y_test)

#let us check applicability of model
# i.e miss classification actual pateint in maligent
#i.e cancer patient but predicted is beniegn is 1
#actual patient is beniegn and predicted as malignant is 5
#hence the model is not acceptable


#let us try to select correct value of k

acc=[]
# running knn algorithm from k=3 to 50 in step of 2
for i in range(3,50,2):
    #declare the model
    neigh=KNeighborsClassifier(n_neighbors=i)
    neigh.fit(X_train,y_train)
    train_acc=np.mean(neigh.predict(X_train)==y_train)
    test_acc=np.mean(neigh.predict(X_test)==y_test)
    acc.append([train_acc,test_acc])

# if you will see acc , it has got accuracy ,i[0].train_acc
#i[i]=test_acc
# to plot graph of train_acc and test_acc
plt.plot(np.arange(3,50,2),[i[0] for i in acc],"ro-")
plt.plot(np.arange(3,50,2),[i[1] for i in acc],"bo-")


##there are 3,5,7 and 9 possible value with high accuray

knn=KNeighborsClassifier(n_neighbors=7)
knn.fit(X_train,y_train)
pred=knn.predict(X_test)

accuracy_score(pred,y_test)
pd.crosstab(pred,y_test)

#










