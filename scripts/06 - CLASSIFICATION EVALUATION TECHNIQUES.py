import scipy, numpy, matplotlib, pandas, numpy, sklearn

names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = pandas.read_csv('pima-indians-diabetes.csv', names=names)
array= dataframe.values
X=array[:,0:8]
Y=array[:,8]

#### CV CLASSIFICATION ACCURACY#####
#number of correct predictions made as a ratio of all predictions made#
#only suitable when there are an equal number of observations in each class
# and that all predictions and prediction errors are equally important#

from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
kfold = KFold(n_splits=10, random_state=7)
model = LogisticRegression()
scoring = 'accuracy'
results = cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
print(("Accuracy: %.3f \nSTD: %.3f") % (results.mean(), results.std()))


####LOGARTITHMIC LOSS####
#evaluating the predictions of probabilities of membership to a class. 
#The scalar probability between 0 and 1 can be seen as a measure
#of confidence for a prediction by an algorithm.
#rewarded or punished proportionally to the condence of the prediction.#

model = LogisticRegression()
scoring = 'neg_log_loss'
results = cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
print(("Logloss: % .3f (%.3f)") % (results.mean(), results.std()))
#Smaller logloss is better, with 0 representing a perfect logloss.



####AREA UNDER ROC Curve (AUC)####
#Represents a model's ability to discriminate between positive and negative
#An area of 1.0 = model that made all predictions perfectly. 
#An area of 0.5 represents a model that is as good as random.
#Sensitivity:  true positive rate (recall). instances from the positive (
rst) class that actually predicted correctly
#Speci
city: is also called the true negative rate.

scoring = 'roc_auc'
results = cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
print(("AUC: %.3f (%.3f)") % (results.mean(), results.std()))



#CONFUSION MATRIX
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

test_size = 0.33
seed=7
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)
model.fit(X_train, Y_train)
predicted = model.predict(X_test)
matrix = confusion_matrix(Y_test, predicted)
print(matrix)


###CLASSIFICATION REPORT 
#This function displays the precision, recall, F1-score and support for each class.

from sklearn.metrics import classification_report
report = classification_report(Y_test, predicted)
print(report)
