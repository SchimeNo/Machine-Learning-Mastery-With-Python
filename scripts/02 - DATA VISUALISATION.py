
#libraries
import scipy
import numpy
import matplotlib
import pandas
import numpy
import sklearn # scikit-learn
import matplotlib.pyplot as plt
from pandas import read_csv
filename = 'pima-indians-diabetes.csv'

                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         # Load CSV using Pandas
#IF URL filename = 'https://goo.gl/vhm1eU'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data = read_csv(filename, names=names)


#PANDAS PROFILE
#report = pandas_profiling.ProfileReport(data)
#report.to_file("report.html")

#Summary
data.shape #size
data.head(20) #head
data.dtypes #attribute classes
data.describe() #data summary

data.groupby('class').size() #distribution (classification only)

#SKEWNESS
data.skew()

####VISUALISATION####
from matplotlib import pyplot

#HISTOGRAMS
data.hist()
pyplot.show()

#DENSITY
data.plot(kind='density', subplots=True, layout=(3,3), sharex=False)
pyplot.show()

#BOXPLOT
data.plot(kind='box', subplots=True, layout=(3,3), sharex=False, sharey=False)
pyplot.show()

#CORRELATIONS
from pandas import set_option
set_option('display.width', 100)
set_option('precision', 3)
correlations = data.corr(method='pearson')
correlations

#CORRELATIONS matrix
fig = pyplot.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(correlations, vmin=-1, vmax=1)
fig.colorbar(cax)
ticks = numpy.arange(0,9,1)
ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.set_xticklabels(names)
ax.set_yticklabels(names)
pyplot.show()

#SCATTER MATRIX
from pandas.plotting import scatter_matrix
scatter_matrix(data)
pyplot.show()
