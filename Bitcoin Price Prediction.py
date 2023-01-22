import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Read the data
data = pd.read_csv('BTC-USD.csv')

# Split the data into features and labels
X = data[['Open', 'High', 'Low', 'Volume']]
y = data['Close']

# Fit the model
model = LinearRegression()
model.fit(X, y)

# Make predictions
predictions = model.predict(X)

# Plot the results
plt.plot(y, color='blue', label='Actual Price')
plt.plot(predictions, color='red', label='Predicted Price')
plt.title('Bitcoin Close Price Prediction')
plt.xlabel('Time')
plt.ylabel('Close Price')
plt.legend()
plt.show()

# Save the results in a csv file
results = pd.DataFrame({'Actual Price': y, 'Predicted Price': predictions})
results.to_csv('bitcoin_predictions.csv', index=False)