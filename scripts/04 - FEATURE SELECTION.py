import scipy, numpy, matplotlib, pandas, numpy, sklearn

names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = pandas.read_csv('pima-indians-diabetes.csv', names=names)
array= dataframe.values


X=array[:,0:8]
Y=array[:,8]

#### Univariate Statistical Tests (Chi-squared for classification)####
# chi-squared (chi2) is a statistical test for non-negative features 
# to select 4 of the best features 
# feature extraction
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
test=SelectKBest(score_func=chi2, k=4)
fit=test.fit(X,Y)
# summarize scores
scores=fit.scores_ #choose attributes with the highest scores (plas, test, mass and age.)
features=fit.transform(X)
features[0:5,:]


###RFE (Recursive Feature Elimination)###
#recursively removing attributes and building a model on those attributes that remain.
#uses the model accuracy to see the attributes contribute the most to predict.
#this example uses RFE with the logistic regression algorithm
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
# feature extraction
model=LogisticRegression()
rfe=RFE(model, 3) #select top 3 features
fit=rfe.fit(X,Y)
#info
fit.n_features_ #Number of features
valuesRFE=fit.support_ #TRUE values  (preg, mass and pedi).

###PCA Principal Component Analysis###
#uses linear algebra to transform the dataset into a compressed form
#we use PCA and select 3 principal components.
from sklearn.decomposition import PCA
pca = PCA(n_components=3) #extracting 3 principal components
fit=pca.fit(X)
#info
fit.explained_variance_ratio_ #Explained Variance
fit.components_
#the transformed dataset (3 principal components) 
#bare little resemblance to the source data.


###Feature Importance###
#Random Forest and Extra Trees can be used to estimate the importance of features.
from sklearn.ensemble import ExtraTreesClassifier
# feature extraction
model = ExtraTreesClassifier()
model.fit(X, Y)
model.feature_importances_

#CHECK RESULTS

results=pandas.concat([pandas.DataFrame(names[0:8]), pandas.DataFrame(model.feature_importances_) ], axis=1, ignore_index=True)
results=results.sort_values(by=[1], ascending=False)


