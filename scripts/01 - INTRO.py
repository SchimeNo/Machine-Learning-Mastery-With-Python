#Machine Learning Mastery With Python

#libraries
import scipy
import numpy
import matplotlib
import pandas
import numpy
import sklearn # scikit-learn
import matplotlib.pyplot as plt

# Strings
data = 'hello world'
data[0]
data[0:4]
len(data)

#Numbers
value = 123.1
value

# Boolean
a = True
b = False
print(a, b)

# Multiple Assignment
a, b, c = 1, 2, 3
(a, b, c)

####STATEMENTS####

#IF STATEMENT
value = 99
if value == 99:
    print ('That is fast')
elif value > 200:
    print ('u too fast dog')
else:
    print ('u good')

# For-Loop
for i in range(10):
    print(i)
    
# While-Loop
i = 0
while i < 10:
    print (i)
    i += 1
    
#Lists 
mylist = [1, 2, 3]
mylist[0]
mylist.append(4)#add a number
len(mylist)
for value in mylist:
    print (value)

#Dictionary
mydict = {'a': 1, 'b': 2, 'c': 3}
mydict['a']
mydict['a'] = 11
mydict['a']
mydict.keys()
mydict.values()
for key in mydict.keys():
    print (mydict[key])
    
# functions
def mysum(x, y):
    return x + y

result = mysum(1, 3)
result

# define an array
mylist = [1, 2, 3]
myarray = numpy.array(mylist)
myarray
myarray.shape

# access values
mylist = [[1, 2, 3], [3, 4, 5]]
myarray = numpy.array(mylist)
myarray[0]
myarray[-1] #Last Row
myarray[0, 2] #row and column
myarray[:, 2]#whole column

# arithmetic
myarray1 = numpy.array([2, 2, 2])
myarray2 = numpy.array([3, 3, 3])
(myarray1 + myarray2) #sum arrays
(myarray1 * myarray2) #multiplying arrays

####PLOTS#####

# LINE plot
myarray = numpy.array([1, 2, 3])
plt.plot(myarray)
plt.xlabel('some x axis')
plt.ylabel('some y axis')
plt.show()

#SCATTER PLOT
x = numpy.array([1, 2, 3])
y = numpy.array([2, 4, 6])
plt.scatter(x,y)
plt.xlabel('some x axis')
plt.ylabel('some y axis')
plt.show()


####PANDA####

#SERIES
myarray = numpy.array([1, 2, 3])
rownames = ['a', 'b', 'c']
myseries = pandas.Series(myarray, index=rownames)
myseries['a']

#DATAFRAME
myarray = numpy.array([[1, 2, 3], [4, 5, 6]])
rownames = ['a', 'b']
colnames = ['one', 'two', 'three']
mydataframe = pandas.DataFrame(myarray, index=rownames, columns=colnames)
mydataframe['one']
mydataframe.one
