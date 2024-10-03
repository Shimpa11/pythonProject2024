# import  pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import  seaborn as sns
# import  pickle as pk
# import streamlit as st
# import pandas_datareader as web
# import yfinance as yf
# from sklearn.preprocessing import MinMaxScaler
# from tensorflow.keras.models import load_model
#
# from sklearn import metrics
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LinearRegression
# import keras
#
# from tensorflow.keras.layers import Dense, Dropout, LSTM
#
# from tensorflow.keras.models import Sequential
#
#
# from tensorflow.keras.models import Sequential
#
#
#
#
# st.set_page_config(
#     page_title="Stock Price Prediction",
#     page_icon="🚗",
#     layout="centered",  # or "wide" for full-width
#     initial_sidebar_state="expanded",)
# # last 10 years data
# start ='2014-01-01'
# end='2024-01-01'
# stock='GOOG'
# data=yf.download(stock,start,end)
#
#
#
#
#
#
# st.title('Stock Trend Prediction')
# user_input=st.text_input('Enter Stock Name', stock)
# if user_input:
#     try:
#     # Use yfinance for data retrieval
#          data = yf.download(user_input, start, end)
#          if data.empty:
#             st.error("No data found for this stock symbol.")
#             st.stop()
#         st.write(data.head())  # Display the first few rows of the DataFrame
#
#         st.stop()
#         st.write(f"Fetching data for: {user_input}")
#     # df=web.DataReader(user_input,'yahoo',start,end)
#     # try:
#     #     df = web.DataReader(user_input, 'yahoo', start, end)
#     # except Exception as e:
#     #     st.error(f"Error retrieving data: {e}")
#     #     st.stop()
#
#
#     # Describe data
#     st.subheader('Data from 2014-2024')
#     st.write(df.describe())
#
#     # visualization
#     st.subheader('Closing Price vs Time chart')
#     fig=plt.figure(figsize=(12,6))
#     plt.plot(df.Close)
#     st.pyplot(fig)
#
#
#     st.subheader('Closing Price vs Time chart woth MA100')
#     ma100=df.Close.rolling(100).mean()
#     ma200=df.Close.rolling(200).mean()
#     fig=plt.figure(figsize=(12,6))
#     plt.plot(ma100)
#     plt.plot(ma200)
#     plt.plot(df.Close)
#     st.pyplot(fig)
#
#
#     train_data=pd.DataFrame(data['Close'][0:int(len(data)*0.7)])
#
#     test_data=pd.DataFrame(data['Close'][int(len(data)*0.7):int(len(data))])
#     print(train_data.shape)
#     print(test_data.shape)
#
#
#
#     # scalling training dataset between 0 and 1
#     scaler=MinMaxScaler(feature_range=(0,1))
#     data_train_scale=scaler.fit_transform(train_data)
#
#     x_train=[]
#     y_train=[]
#     for i in range(100,data_train_scale.shape[0]):
#         x_train.append(data_train_scale[i-100:i])
#         y_train.append(data_train_scale[i,0])
#     x_train,y_train=np.array(x_train),np.array(y_train)
#
#     print(x_train)
#     print(y_train)
#
#     # model=pk.load('modelStock.pkl')
#     model=load_model('keras_model.h5')
#
#     past_100days=train_data.tail(100)
#     test_data=pd.concat([past_100days, test_data],ignore_index=True)
#     print(test_data.head())
#
#     test_data_scale=scaler.fit_transform(test_data)
#
#     print(test_data_scale)
#     print(test_data_scale.shape)
#     x_test=[]
#     y_test=[]
#     for i in range(100,test_data_scale.shape[0]):
#         x_test.append(test_data_scale[i-100:i])
#         y_test.append(test_data_scale[i,0])
#     x_test,y_test=np.array(x_test),np.array(y_test)
#
#     print('x_test:',x_test.shape)
#     print('y_test:',y_test.shape)
#     # Reshape x_test to 3D
#     x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))
#
#
#     print("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
#     y_pred=model.predict(x_test)
#     # print('y_pred:',y_pred.shape)
#
#     # Check the shape of y_pred
#     print("Shape of y_pred before reshaping:", y_pred.shape)
#
#     # If y_pred is still 3D, we need to reshape it
#     if y_pred.ndim == 3:
#         y_pred_reshaped = y_pred[:, -1, 0]  # Get the last timestep for predictions
#     else:
#         y_pred_reshaped = y_pred.reshape(-1)
#
#     print("Shape of y_pred after reshaping:", y_pred_reshaped.shape)  # Should be (samples,)
#
#
#
#     # scale_factor=scaler.scale_
#     # scale_factors=1/scale_factor
#     # y_pred=y_pred*scale_factors
#     # y_test=y_test*scale_factors
#
#     y_pred_reshaped = y_pred.reshape(-1)
#
#     # Rescaling predictions
#     y_pred_rescaled = scaler.inverse_transform(y_pred_reshaped.reshape(-1, 1))
#     y_test_rescaled = scaler.inverse_transform(y_test.reshape(-1, 1))
#
#
#     st.subheader('Predictions vs Original')
#     fig2=plt.figure(figsize=(10,6),facecolor='lightyellow')
#     plt.plot(y_test,'r' ,label='Original Price')
#     plt.plot(y_pred_reshaped,'g', label='Predicted Price')
#
#     plt.xlabel('Time')
#     plt.ylabel('Price')
#     plt.legend()
#
#     st.pyplot(fig2)
# except Exception as e:
#         st.error(f"Error retrieving data: {e}")


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model

st.set_page_config(
    page_title="Stock Price Prediction",
    page_icon="📈",
    layout="centered",
    initial_sidebar_state="expanded",
)

# Set time period
start = '2014-01-01'
end = '2024-01-01'
default_stock = 'GOOG'

st.title('Stock Trend Prediction')
user_input = st.text_input('Enter Stock Name', default_stock)

if user_input:
    try:
        # Use yfinance for data retrieval
        data = yf.download(user_input, start, end)

        if data.empty:
            st.error("No data found for this stock symbol.")
            st.stop()

        st.write(data.head())  # Display the first few rows of the DataFrame

        # Describe data
        st.subheader('Data from 2014-2024')
        st.write(data.describe())

        # Visualization
        st.subheader('Closing Price vs Time chart')
        fig = plt.figure(figsize=(12, 6))
        plt.plot(data['Close'])
        st.pyplot(fig)

        st.subheader('Closing Price vs Time chart with MA100 and MA200')
        ma100 = data['Close'].rolling(100).mean()
        ma200 = data['Close'].rolling(200).mean()
        fig = plt.figure(figsize=(12, 6))
        plt.plot(ma100, label='MA100')
        plt.plot(ma200, label='MA200')
        plt.plot(data['Close'], label='Close Price')
        plt.legend()
        st.pyplot(fig)

        # Prepare training and testing datasets
        train_data = data['Close'][0:int(len(data) * 0.7)]
        test_data = data['Close'][int(len(data) * 0.7):]

        # Scale the training dataset
        scaler = MinMaxScaler(feature_range=(0, 1))
        data_train_scale = scaler.fit_transform(train_data.values.reshape(-1, 1))

        # Prepare training data
        x_train, y_train = [], []
        for i in range(100, len(data_train_scale)):
            x_train.append(data_train_scale[i - 100:i])
            y_train.append(data_train_scale[i, 0])
        x_train, y_train = np.array(x_train), np.array(y_train)

        # Load the model
        model = load_model('keras_model.h5')

        # Prepare test data
        past_100days = train_data.tail(100)
        test_data = pd.concat([past_100days, test_data], ignore_index=True)
        test_data_scale = scaler.transform(test_data.values.reshape(-1, 1))

        x_test, y_test = [], []
        for i in range(100, len(test_data_scale)):
            x_test.append(test_data_scale[i - 100:i])
            y_test.append(test_data_scale[i, 0])
        x_test, y_test = np.array(x_test), np.array(y_test)

        # Reshape x_test for LSTM
        x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))

        # Predictions
        y_pred = model.predict(x_test)

        # Rescaling predictions
        y_pred_rescaled = scaler.inverse_transform(y_pred)
        y_test_rescaled = scaler.inverse_transform(y_test.reshape(-1, 1))

        # Plot predictions vs original
        st.subheader('Predictions vs Original')
        fig2 = plt.figure(figsize=(10, 6), facecolor='lightyellow')
        plt.plot(y_test_rescaled, 'r', label='Original Price')
        plt.plot(y_pred_rescaled, 'g', label='Predicted Price')
        plt.xlabel('Time')
        plt.ylabel('Price')
        plt.legend()
        st.pyplot(fig2)

    except Exception as e:
        st.error(f"Error retrieving data: {e}")
