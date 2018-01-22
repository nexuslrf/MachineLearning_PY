# -*- coding: utf-8 -*-
# more detail: http://scikit-learn.org/stable/modules/classes.html#module-sklearn.datasets
from sklearn import datasets
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

loaded_data = datasets.load_boston()
data_X = loaded_data.data
data_y = loaded_data.target

model = LinearRegression()
model.fit(data_X,data_y)

print model.coef_ # linear parameters
print model.intercept_ # interception
print model.get_params() # model's init func's init parameters
print model.score(data_X,data_y) # R^2 coefficient of determination in LinearRegression: Show how well the learning is by percentage

print model.predict(data_X[:4,:])
print data_y[:4]

X,y = datasets.make_regression(n_samples=100,n_features=1,n_targets=1,noise=20)

plt.scatter(X,y)
plt.show()