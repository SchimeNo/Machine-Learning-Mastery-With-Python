#In this lesson we are going to take a look at seven regression algorithms that you can spot-check
#on your dataset. Starting with four linear machine learning algorithms:
import scipy, numpy, matplotlib, pandas, numpy, sklearn

from pandas import read_csv
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
filename = 'housing.csv'
names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO','B', 'LSTAT', 'MEDV']
dataframe = read_csv(filename, delim_whitespace=True, names=names)
array = dataframe.values
X = array[:,0:13]
Y = array[:,13]
kfold = KFold(n_splits=10, random_state=7)

#####LINEAR ALGORITHMS

# Linear Regression.
from sklearn.linear_model import LinearRegression
model = LinearRegression()
scoring = 'neg_mean_squared_error'
results = cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
print(results.mean())

# Ridge Regression.
#an extension of linear regression where the loss function is modi
ed to minimize 
#the complexity of the model measured as the sum squared value of the coecient values (L2-norm). 
from sklearn.linear_model import Ridge
model = Ridge()
scoring = 'neg_mean_squared_error'
results = cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
print(results.mean())

# LASSO Linear Regression.
from sklearn.linear_model import Lasso
model = Lasso()
scoring = 'neg_mean_squared_error'
results = cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
print(results.mean())

# Elastic Net Regression.
from sklearn.linear_model import ElasticNet
model = ElasticNet()
scoring = 'neg_mean_squared_error'
results = cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
print(results.mean())
####NON-LINEAR ALGORITHM

# k-Nearest Neighbors.

from sklearn.neighbors import KNeighborsRegressor
model = KNeighborsRegressor()
scoring = 'neg_mean_squared_error'
results = cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
print(results.mean())


# Classi
cation and Regression Trees.
from sklearn.tree import DecisionTreeRegressor
model = DecisionTreeRegressor()
scoring = 'neg_mean_squared_error'
results = cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
print(results.mean())

# Support Vector Machines.
from sklearn.svm import SVR
model = SVR()
scoring = 'neg_mean_squared_error'
results = cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
print(results.mean())
