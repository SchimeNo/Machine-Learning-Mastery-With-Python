import scipy, numpy, matplotlib, pandas, numpy, sklearn

names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = pandas.read_csv('pima-indians-diabetes.csv', names=names)
array= dataframe.values


X=array[:,0:8]
Y=array[:,8]


#TRAIN & TEST
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

test_size = 0.33
seed = 7

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)


#LOGISITIC REGRESSION
model = LogisticRegression()
model.fit(X_train, Y_train)
result = model.score(X_test, Y_test)
result #Accuracy

###k folds CROSS VALIDATION###
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression

num_folds = 10
kfold=KFold(n_splits=num_folds, random_state=seed)
model = LogisticRegression()
results = cross_val_score(model, X, Y, cv=kfold) #applying cross validation
results.mean() #mean of the accuracy
results.std() #deviation

###Leave One Out Cross Validation###
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression

num_folds = 10
loocv = LeaveOneOut()
model = LogisticRegression()
results = cross_val_score(model, X, Y, cv=loocv)

results.mean() #mean of the accuracy
results.std() #deviation


###Repeated Random Test-Train Splits###
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
n_splits = 10
test_size = 0.33
seed = 7
kfold = ShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=seed)
model = LogisticRegression(solver='lbfgs')
results = cross_val_score(model, X, Y, cv=kfold)
results.mean() #mean of the accuracy
results.std() #deviation
