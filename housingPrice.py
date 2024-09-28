from  sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import  pandas as pd
import matplotlib.pyplot as plt
#load dataset
df=pd.read_csv('C:/Users/ershi/Downloads/Housing.csv')

# print(df)
print(df.columns)
print(df.head())

#define features nd targets
X = df[['area', 'bedrooms', 'bathrooms']] # Add other features as needed
y = df['price']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)
# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")

# # Residual plot
# residuals = y_test - y_pred
#
# plt.figure(figsize=(10, 6))
# plt.bar(y_pred, residuals, color='blue', s=10, alpha=0.5)
# plt.hlines(y=0, xmin=min(y_pred), xmax=max(y_pred), colors='red')
# plt.xlabel('Predicted Values')
# plt.ylabel('Residuals')
# plt.title('Residuals vs Predicted Values')
# plt.show()


plt.figure(figsize=(12, 6))

# Plot actual vs. predicted values
plt.plot(range(len(y_test)), y_test, label='Actual Values', color='blue', marker='o')
plt.plot(range(len(y_pred)), y_pred, label='Predicted Values', color='red', linestyle='--', marker='x')

plt.xlabel('Index')
plt.ylabel('House Price')
plt.title('Actual vs. Predicted House Prices')
plt.legend()