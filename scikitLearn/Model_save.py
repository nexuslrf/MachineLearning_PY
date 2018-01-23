# -*- coding: utf-8 -*-
import numpy as np
from sklearn import datasets # sklearn 中的数据库
from sklearn.model_selection import train_test_split # 将数据随机分成两部分
from sklearn.neighbors import KNeighborsClassifier # K-临近算法

iris = datasets.load_iris() # sklearn 自有数据 iris花
iris_X = iris.data
iris_Y = iris.target
print iris_X[0] 

X_train,X_test,Y_train,Y_test = train_test_split(iris_X,iris_Y,test_size=0.3)

knn = KNeighborsClassifier()
knn.fit(X_train,Y_train)

print(knn.predict(X_test))

print

print(Y_test)

import pickle
with open("save/knn.pickle",'wb') as f:
	pickle.dump(knn,f)

with open("save/knn.pickle",'rb') as f:
	clf1=pickle.load(f)
	print(clf1.predict(iris_X[0:1]))

from sklearn.externals import joblib
joblib.dump(knn,"save/knn.pkl")

clf2=joblib.load("save/knn.pkl")
print(clf2.predict([iris_X[0]]))