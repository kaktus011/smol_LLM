import pandas as pd
import joblib

# Load the data from the CSV file
data = pd.read_csv('data/financial_data.csv')

# Create DataFrame
df = pd.DataFrame(data)

# Combine Month and Year for x-axis labels
df['Month_Year'] = df['Month'] + ' ' + df['Year'].astype(str)

# Calculate Profit
df['Profit'] = df['Income'] - df['Expenses']

# Save the data to a CSV file
df.to_csv('data/financial_data.csv', index=False)

# Load the trained model
model = joblib.load('financial_model.pkl')

# Prepare the data for prediction
df['Month'] = pd.to_datetime(df['Month_Year'], format='%b %Y').dt.month
df['Year'] = pd.to_datetime(df['Month_Year'], format='%b %Y').dt.year
X = df[['Month', 'Year', 'Marketing_Spend', 'Number_of_Employees']]

# Make predictions
df['Predicted_Profit'] = model.predict(X)

# Display the prediction
print(df[['Month_Year', 'Income', 'Expenses', 'Marketing_Spend', 'Number_of_Employees', 'Profit', 'Predicted_Profit']])