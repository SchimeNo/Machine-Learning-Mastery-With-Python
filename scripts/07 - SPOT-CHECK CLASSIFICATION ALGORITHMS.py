#Spot-checking is a way of discovering which algorithms perform well on your problem

import scipy, numpy, matplotlib, pandas, numpy, sklearn
from pandas import read_csv
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = pandas.read_csv('pima-indians-diabetes.csv', names=names)
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]

#####LINEAR ALGORITHMS

####Logistic Regression Classification
#Logistic regression assumes a Gaussian distribution for the numeric input variables and can
#model binary classi
cation problems.
from sklearn.linear_model import LogisticRegression
num_folds = 10
kfold = KFold(n_splits=10, random_state=7)
model = LogisticRegression()
results = cross_val_score(model, X, Y, cv=kfold)
print(results.mean())

# Linear Discriminant Analysis.
#statistical technique for binary and multiclass. It too assumes a Gaussian distribution
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
model=LinearDiscriminantAnalysis()
results= cross_val_score(model, X, Y, cv=kfold)
print(results.mean())


####NON-LINEAR ALGORITHMS
# k-Nearest Neighbors.
from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier()
results = cross_val_score(model, X, Y, cv=kfold)
print(results.mean())


# Naive Bayes.
#probability of each class and the conditional probability of each class.
from sklearn.naive_bayes import GaussianNB
model = GaussianNB()
results = cross_val_score(model, X, Y, cv=kfold)
print(results.mean())

# Classi
cation and Regression Trees.
#construct a binary tree from the training data. Split points are chosen by 
#evaluating each attribute and each value of each attribute in the training data 
#in order to minimize a cost function
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier()
results = cross_val_score(model, X, Y, cv=kfold)
print(results.mean())

# Support Vector Machines.
#seek a line that best separates two classes. Those data instances that are closest 
#to the line that best separates the classes are called support vectors and  infuence where the line is placed.
from sklearn.svm import SVC
model = SVC()
results = cross_val_score(model, X, Y, cv=kfold)
print(results.mean())
