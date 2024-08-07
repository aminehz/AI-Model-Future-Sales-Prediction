

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
import plotly.graph_objects as go
import plotly.express as px

data=pd.read_csv('advertising.csv')
print(data.head())

#check if a null values exist
print(data.isnull().sum())

figure = px.scatter(data_frame= data, x="Sales", y="TV", trendline="ols")
figure.show()

figure = px.scatter(data_frame= data, x="Sales", y="Newspaper",size="Newspaper" , trendline="ols")
figure.show()

figure = px.scatter(data_frame= data, x="Sales", y="Radio", size="Radio", trendline="ols")
figure.show()

correlation = data.corr()
print(correlation["Sales"].sort_values(ascending=False))

#Split the data
x= np.array(data.drop(["Sales"],axis= 1))
y = np.array(data["Sales"])

xtrain, xtest, ytrain, ytest = train_test_split(x,y, test_size=0.2, random_state=42)

#Model
model = LinearRegression()
model.fit(xtrain, ytrain)
print(model.score(xtest, ytest))

#Our features are TV , Radio , Newspaper
features= np.array([[230.1,37.8, 69.2]])
print(model.predict(features))

# Calculate metrics
y_pred= model.predict(xtest)
mse = mean_squared_error(ytest, y_pred)
rmse = mse**0.5
mae = mean_absolute_error(ytest, y_pred)
r2 = r2_score(ytest, y_pred)

# Plot Predicted vs Actual values
plt.figure(figsize=(12, 6))
plt.scatter(ytest, y_pred, color='blue', edgecolor='k', alpha=0.7)
plt.plot([ytest.min(), ytest.max()], [ytest.min(), ytest.max()], 'r--', lw=2)
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Predicted vs Actual')
plt.show()

# Plot Residuals
residuals = ytest - y_pred
plt.figure(figsize=(12, 6))
plt.scatter(y_pred, residuals, color='blue', edgecolor='k', alpha=0.7)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Predicted')
plt.ylabel('Residuals')
plt.title('Residuals Plot')
plt.show()

# Plot Residuals Histogram
plt.figure(figsize=(12, 6))
sns.histplot(residuals, kde=True)
plt.xlabel('Residuals')
plt.title('Residuals Histogram')
plt.show()

# Print Metrics
print(f"Mean Absolute Error: {mae}")
print(f"Mean Squared Error: {mse}")
print(f"Root Mean Squared Error: {rmse}")
print(f"R-squared: {r2}")

