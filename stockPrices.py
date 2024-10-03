import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
import tensorflow
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import keras

from tensorflow.keras.layers import Dense, Dropout, LSTM

from tensorflow.keras.models import Sequential


from tensorflow.keras.models import Sequential

# last 10 years data
start ='2014-01-01'
end='2024-01-01'
stock='GOOG'
data=yf.download(stock,start,end)
# print(data)
data=data.reset_index()
print(data)

# moving avergaes
ma_100days=data.Close.rolling(100).mean()
# print(ma_100days)

plt.figure(figsize=(10,6),facecolor='lightyellow')
plt.plot(ma_100days,'r' ,label='ma 100days')
plt.plot(data.Date,data.Close,'g',label='Close Price')
plt.title('Moving Average_100 days')
plt.xlabel('x-axis')
plt.ylabel('y-axis')
plt.legend(loc='upper right',facecolor='orange', edgecolor='green', framealpha=0.8)

# plt.show()

ma_200days=data.Close.rolling(200).mean()
# print(ma_100days)

plt.figure(figsize=(10,6),facecolor='lightyellow')
plt.plot(data.Date,ma_100days,'r' ,label='ma 100days')
plt.plot(data.Date,data.Close,'g',label='Close Price')
plt.plot(data.Date,ma_200days,'b',label='ma 200days')
plt.title('Moving Average_ 100 vs 200 days')
plt.xlabel('x-axis')
plt.ylabel('y-axis')
plt.legend(loc='upper right',facecolor='orange', edgecolor='green', framealpha=0.8)

# plt.show()
# plt.plot(data.Date, data.Close ,'r')
# plt.show()
data=pd.DataFrame(data)
data=data.drop('Adj Close',axis=1)
# y=data['Adj Close']
# print(x)
# print(y)
# X_train, Y_train, X_test,Y_test=train_test_split(x,y,test_size=0.2)

train_data=pd.DataFrame(data['Close'][0:int(len(data)*0.7)])

test_data=pd.DataFrame(data['Close'][int(len(data)*0.7):int(len(data))])
print(train_data.shape)
print(test_data.shape)


# scalling training dataset between 0 and 1
scaler=MinMaxScaler(feature_range=(0,1))
data_train_scale=scaler.fit_transform(train_data)

x_train=[]
y_train=[]
for i in range(100,data_train_scale.shape[0]):
    x_train.append(data_train_scale[i-100:i])
    y_train.append(data_train_scale[i,0])
x_train,y_train=np.array(x_train),np.array(y_train)

print(x_train.shape)
print(y_train.shape)

# Reshape x_train to 3D
x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
model=Sequential()
s0=model.add(LSTM(units=50,activation='relu',return_sequences=True ,input_shape=(x_train.shape[1],1)))
d0=model.add(Dropout(0.2))

s1=model.add(LSTM(units=60,activation='relu',return_sequences=True))
d1=model.add(Dropout(0.3))

s2=model.add(LSTM(units=80,activation='relu',return_sequences=True))
d2=model.add(Dropout(0.4))

s3=model.add(LSTM(units=120,activation='relu',return_sequences=False))
d3=model.add(Dropout(0.5))

d4=model.add(Dense(1))

summary=model.summary()
compiled=model.compile(optimizer='adam',loss='mean_squared_error')

training=model.fit(x_train,y_train,epochs=5)


saving=model.save('keras_model.h5')
print(test_data.head())
print(train_data.tail())

past_100days=train_data.tail(100)
test_data=pd.concat([past_100days, test_data],ignore_index=True)
print(test_data.head())

test_data_scale=scaler.fit_transform(test_data)

print(test_data_scale)
print(test_data_scale.shape)
x_test=[]
y_test=[]
for i in range(100,test_data_scale.shape[0]):
    x_test.append(test_data_scale[i-100:i])
    y_test.append(test_data_scale[i,0])
x_test,y_test=np.array(x_test),np.array(y_test)

print('x_test:',x_test.shape)
print('y_test:',y_test.shape)
# Reshape x_test to 3D
x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))


print("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
y_pred=model.predict(x_test)
# print('y_pred:',y_pred.shape)

# Check the shape of y_pred
print("Shape of y_pred before reshaping:", y_pred.shape)

# If y_pred is still 3D, we need to reshape it
if y_pred.ndim == 3:
    y_pred_reshaped = y_pred[:, -1, 0]  # Get the last timestep for predictions
else:
    y_pred_reshaped = y_pred.reshape(-1)

print("Shape of y_pred after reshaping:", y_pred_reshaped.shape)  # Should be (samples,)



# scale_factor=scaler.scale_
# scale_factors=1/scale_factor
# y_pred=y_pred*scale_factors
# y_test=y_test*scale_factors

y_pred_reshaped = y_pred.reshape(-1)

# Rescaling predictions
y_pred_rescaled = scaler.inverse_transform(y_pred_reshaped.reshape(-1, 1))
y_test_rescaled = scaler.inverse_transform(y_test.reshape(-1, 1))
print('y_predict after reshape:', y_pred.shape)
plt.figure(figsize=(10,6),facecolor='lightyellow')
plt.plot(y_test,'r' ,label='Original Price')
plt.plot(y_pred_reshaped,'g', label='Predicted Price')

plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()

import pickle as pk
# app=pk.dump(model,open('modelStock.pkl','wb'))

saving=model.save('keras_model.h5')


