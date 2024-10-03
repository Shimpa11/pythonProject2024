import  pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import  seaborn as sns
import  pickle as pk
import streamlit as st

# Set page config (including theme)
st.set_page_config(
    page_title="Car Price Prediction",
    page_icon="🚗",
    layout="centered",  # or "wide" for full-width
    initial_sidebar_state="expanded",
    # You can set the primary and background colors here
    # theme={
    #     'primaryColor': '#F39C12',
    #     'backgroundColor': '#2C3E50',
    #     'secondaryBackgroundColor': '#34495E',
    #     'textColor': '#ECF0F1'
    # }
)


model=pk.load(open('model.pkl','rb'))

st.header('Car Price Prediction Model')
data=pd.read_csv('data/car data.csv')

def brand_name(Car_Name):
    Car_Name=Car_Name.split(' ')[0]
    return  Car_Name.strip()
data['Car_Name']=data['Car_Name'].apply(brand_name)
brand=st.selectbox('Select Car Brand',data['Car_Name'].unique())
present_price=st.slider('Present Price', 0,2000000)



km=st.slider('No of Km Driven', 0,1000000)
fuel_type_map = {'Petrol': 0, 'Diesel': 1, 'CNG': 2}
fuel = st.selectbox('Fuel Type', list(fuel_type_map.keys()))

seller_type_map = {'Dealer': 0, 'Individual': 1}
seller=st.selectbox('Seller Type',list(seller_type_map.keys()))

transmission_type_map = {'Manual': 0, 'Automatic': 1}
transmission=st.selectbox('Transmission Type',list(transmission_type_map.keys()))

owner=st.selectbox('Owner',data['Owner'].unique())
age=st.slider('Car Age', 0,20)

if st.button('Predict Price'):
    new_data = pd.DataFrame(
       [ [present_price,km,
          fuel_type_map[fuel],  # Map fuel to numerical value
          seller_type_map[seller],  # Map seller to numerical value
          transmission_type_map[transmission],  # Map transmission to numerical value
          owner,age]],
        columns=['Present_Price','Kms_Driven','Fuel_Type','Seller_Type','Transmission','Owner','Age'])
    st.write(new_data)
    try:
        new_prediction = model.predict(new_data)
        # Display predicted price
        st.markdown('**Predicted Car Price:** $' + str(round(new_prediction[0], 2)))
    except ValueError as e:
        st.error(f"Error during prediction: {e}")

st.subheader("Cars by  Transmission Type")
fig, ax = plt.subplots(figsize=(10, 6))

plt.figure(figsize=(10, 6))
sns.countplot(x='Transmission', data=data, ax=ax,color='orange')
ax.set_title('Count of Cars by Transmission Type')
ax.set_xlabel('Transmission Type')
ax.set_ylabel('Count')
st.pyplot(fig)

# Histogram of Selling Prices
st.subheader("Distribution of Selling Prices")
fig, ax = plt.subplots(figsize=(12, 8))
sns.histplot(data['Selling_Price'], bins=20, kde=True, ax=ax,color='orange')
ax.set_title('Distribution of Selling Prices')
ax.set_xlabel('Selling Price')
ax.set_ylabel('Frequency')
st.pyplot(fig)  # Render the plot

# Create a scatter plot of Present Price vs. Selling Price
st.subheader("Present Price vs. Selling Price")
fig, ax = plt.subplots(figsize=(10, 6))
sns.scatterplot(x='Present_Price', y='Selling_Price', data=data, ax=ax,color='red')
ax.set_title('Present Price vs. Selling Price')
ax.set_xlabel('Present Price')
ax.set_ylabel('Selling Price')
st.pyplot(fig)
