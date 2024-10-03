import pandas as pd
import  matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import datetime
import pickle
import joblib
import streamlit
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn import metrics
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
from sklearn.preprocessing import LabelEncoder
pd.set_option('display.max_columns', None)
pd.set_option('future.no_silent_downcasting', True)
# dataset
data=pd.read_csv('C:/Users/ershi/Downloads/car data.csv')
print(data)
# print(data.head())
# print(data.info())
# print(data.describe())
# missing value from each col
print(data.isnull().sum())
# check tx type seller type , fuel type distribution  of categorical data
date_time = datetime.datetime.now()



data['Age']=date_time.year-data['Year']
print(data)
# print(data['Fuel_Type'].value_counts())
# print(data['Seller_Type'].value_counts())
# print(data['Transmission'].value_counts())
data=data.drop('Year',axis=1)
print(data)



# assign numerical value to types, encoding  data as 0 , 1 and 2
# fuel type col


data['Fuel_Type'] = data['Fuel_Type'].replace({'Petrol': 0, 'Diesel': 1, 'CNG': 2})

data=data.replace({'Seller_Type':{'Dealer':0,'Individual':1}})
data=data.replace({'Transmission':{'Manual':0,'Automatic':1}})
data = data.infer_objects(copy=False)
# o=data['Owner'].unique()
# print(o)

# print(data)

# splitting the data and Target
# axis =1 for dropping cols
X=data.drop(['Car_Name','Selling_Price'],axis=1)
Y=data['Selling_Price']

# print(X)
# print(Y)

# splitting the data into train and test data

X_train,X_test,Y_train,Y_test=train_test_split(X,Y, test_size=0.1,random_state=2)

# Model Training
# load LinearRegression model
model=LinearRegression()
model=model.fit(X_train, Y_train)
value = 'eon'
try:
    float_value = float(value)
except ValueError:
    print(f"Cannot convert {value} to float.")
# Model evaluation
training_data_prediction=model.predict(X_train)


error_score=metrics.r2_score(Y_train,training_data_prediction)
print('R2 score:', error_score)

# Visualize actual and predicted prices
# categories = np.random.choice(['A', 'B'], size=100)
# print("Unique categories:", np.unique(categories))
# categories = [str(category) for category in categories]
# # Define a color map
# color_map = {'A': 'blue', 'B': 'red'}
# colors = [color_map[category] for category in categories]
# plt.scatter(Y_train,training_data_prediction ,c=colors)

# plt.scatter(Y_train,training_data_prediction)
#
# plt.xlabel('Actual prices')
# plt.ylabel('Predicted Prices')
# plt.title('Car Price Actual vs Predict')


test_data_prediction=model.predict(X_test)
print(test_data_prediction)
error_score=metrics.r2_score(Y_test,test_data_prediction)
print('R2 score:', error_score)

mae = mean_absolute_error(Y_test, test_data_prediction)  # y_test is the actual prices
mse = mean_squared_error(Y_test, test_data_prediction)
r2 = r2_score(Y_test, test_data_prediction)

print(f'MAE: {mae}, MSE: {mse}, R²: {r2}')

# Visualize actual and predicted prices
# categories = np.random.choice(['A', 'B'], size=100)
# categories = [str(category) for category in categories]
# # Define a color map
# color_map = {'A': 'blue', 'B': 'red'}
# colors = [color_map[category] for category in categories]
# plt.scatter(Y_test,test_data_prediction,color='green')
plt.scatter(Y_test,test_data_prediction)

plt.xlabel('Actual prices Test')
plt.ylabel('Predicted Prices')
plt.title('Car Price Actual vs Predict')
plt.show()






# Prepare new data for prediction
new_data = pd.DataFrame({
    'Car_Name':'ritz',
    'Present_Price':6.59,
    'Kms_Driven':10000,
    'Fuel_Type':0,
    'Seller_Type':0,
    'Transmission':1,
    'Owner':0,
    'Age':5
},index=[0])

# # Ensure to match the expected features in your model
# new_prediction = model.predict(new_data)
#
# print(f'Predicted Present Price: {new_prediction[0]}')

 # Drop Car_Name for prediction
features_for_prediction = new_data.drop(['Car_Name'], axis=1)

# Ensure to match the expected features in your model
new_prediction = model.predict(features_for_prediction)
print(f'Predicted Present Price for "{new_data["Car_Name"][0]}": {new_prediction[0]}')

plt.figure(figsize=(10,6))
sns.histplot(data['Selling_Price'],bins=20,kde=True)
plt.title('Distribution of Selling Price')
plt.xlabel('Selling Price')
plt.ylabel('Frequency')
plt.show()

# plt.figure(figsize=(10,6))
# sns.boxplot(x='Present_Price',y='Selling_Price')
# plt.title('Fuel type vs Selling Price')
# plt.xlabel('Selling Price')
# plt.ylabel('Frequency')
# plt.show()

plt.figure(figsize=(10,6))
data.groupby('Transmission')['Selling_Price'].mean().plot(kind='bar',color='magenta')
plt.title('Average selling price of cars based on Transmission Type')
plt.show()


import pickle as pk
app=pk.dump(model,open('model.pkl','wb'))
