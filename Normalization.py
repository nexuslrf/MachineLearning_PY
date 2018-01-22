# -*- coding: utf-8 -*-
#normalization: 由于资料的偏差与跨度会影响机器学习的成效，因此正规化(标准化)数据可以提升机器学习的成效
# example below：
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.datasets.samples_generator import make_classification
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import numpy as np

a = np.array([[10, 2.7, 3.6],
	[-100, 5, -2],
	[120, 20, 40]],dtype=np.float64)

print a
print preprocessing.scale(a)

X,y = make_classification(n_samples=300, n_features=2, n_redundant=0,n_informative=2,
	random_state=13,n_clusters_per_class=1,scale=100)
plt.scatter(X[:,0],X[:,1],c=y)
plt.show()

X_normalized = preprocessing.scale(X)    # normalization step
X_train, X_test, y_train, y_test = train_test_split(X_normalized, y, test_size=.3)
clf_1 = SVC()
clf_1.fit(X_train, y_train)
print(clf_1.score(X_test, y_test)) #>>0.877777777778

#without normalization
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3)
clf = SVC()
clf.fit(X_train, y_train)
print(clf.score(X_test, y_test)) #much awful result >>0.488888888889