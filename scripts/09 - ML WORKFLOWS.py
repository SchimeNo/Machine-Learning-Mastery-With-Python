#Automating Machine Learning Workflows
#Pipelines help to clearly de
ne and automate these workflows.

# Create a pipeline that standardizes the data then creates a model
from pandas import read_csv
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
# load data
filename = 'pima-indians-diabetes.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = read_csv(filename, names=names)
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]

# create pipeline The pipeline is defi
ned with two steps:
#1. Standardize the data.
#2. Learn a Linear Discriminant Analysis model.
estimators=[]
estimators.append(('standarize', StandardScaler()))
estimators.append(('1da', LinearDiscriminantAnalysis()))
model=Pipeline(estimators)
# evaluate pipeline
kfold = KFold(n_splits=10, random_state=7)
results = cross_val_score(model, X, Y, cv=kfold)
print(results.mean())


#Feature Extraction
#must be restricted to the data in your training dataset.
#all the feature extraction and the feature union occurs
#within each fold of the cross validation procedure

#1. Feature Extraction with Principal Component Analysis (3 features).
#2. Feature Extraction with Statistical Selection (6 features).
#3. Feature Union.
#4. Learn a Logistic Regression Model.

from sklearn.pipeline import FeatureUnion
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest
# create feature union
features = []
features.append(('pca', PCA(n_components=3)))
features.append(('select_best', SelectKBest(k=6)))
feature_union = FeatureUnion(features)
# create pipeline
estimators = []
estimators.append(('feature_union', feature_union))
estimators.append(('logistic', LogisticRegression()))
model = Pipeline(estimators)
# evaluate pipeline
kfold = KFold(n_splits=10, random_state=7)
results = cross_val_score(model, X, Y, cv=kfold)
print(results.mean())