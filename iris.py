import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

data=pd.read_csv('C:/Users/ershi/Downloads/Iris.csv')

data=data.drop('Id',axis='columns')
# print(data.describe())
# print(data.shape)
# print(data.info())

count=data['Species'].value_counts()

print(count)
print(data.dtypes)

le = LabelEncoder()
data['Species'] = le.fit_transform(data['Species'])
print('Unique values in species:', data['Species'].unique())

print(data.columns)
meanSepal=data['SepalLengthCm'].mean()
print(meanSepal)
n=data.isnull().sum()
print(n)

plt.scatter(data['SepalLengthCm'],data['SepalWidthCm'],color='m')

bars=count.plot(kind='bar',color='green',width=0.3)
# plt.show()
data=pd.DataFrame(data)
print(data)
# gtData=data['SepalLengthCm']>5
# filtered_data = data[gtData]

# Display the results
# print(filtered_data)
plt.hist(data['SepalLengthCm'],edgecolor='m')
plt.title('Sepal Length Cm')


plt.hist(data['SepalWidthCm'])
plt.title('Sepal Width Cm')


plt.hist(data['PetalLengthCm'])
plt.title('Petal Length Cm')



plt.hist(data['PetalWidthCm'])
plt.title('Petal Width Cm')


# X_train,Y_train,X_test,Y_test=train_test_split()
train, test=train_test_split(data,test_size=0.3)
print(train.shape)
print(test.shape)

X=data.drop('Species',axis=1)
Y=data['Species']
X_train=X
Y_train=Y
print(X)
print(Y)

X_test=X
Y_test=Y

print(X_train)
print(Y_train)

model=LinearRegression()
model1=svm.SVC


train_data=model.fit(X_train,Y_train)
Y_pred=model.predict(X_test)
acc=metrics.r2_score(Y_test,Y_pred)
print('The accuracy of linearR is:', acc)




