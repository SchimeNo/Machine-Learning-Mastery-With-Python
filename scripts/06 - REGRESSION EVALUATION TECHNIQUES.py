import scipy, numpy, matplotlib, pandas, numpy, sklearn


####MAE (Mean Absolute Error)
#the sum of the absolute dierences between predictions and actual values. 
#It gives an idea of how wrong the predictions were.
#but no idea of the direction (e.g. over or under predicting).
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
model = LinearRegression()
scoring = 'neg_mean_absolute_error'
results = cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
print(("MAE: %.3f (%.3f)") % (results.mean(), results.std()))


###MSE MEAN SQUARED ERROR
# like MAE it provides a gross idea of the magnitude of error.
#This metric too is inverted so that the results are increasing.
scoring = 'neg_mean_squared_error'
results = cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
print(("MSE: %.3f (%.3f)") % (results.mean(), results.std()))

#R-SQUARED
#is a value between 0 and 1 for no-
t and perfect 
t respectively.
scoring = 'r2'
results = cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
print(("R^2: %.3f (%.3f)") % (results.mean(), results.std()))
#predictions have a poor 
t to the actual values with a value closer to zero and less than 0.5.
