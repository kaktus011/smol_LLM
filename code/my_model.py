import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import joblib
import os

# Load the data
df = pd.read_csv('data/financial_training_data.csv')

# Feature Engineering
df['Month_Year'] = pd.to_datetime(df['Month_Year'], format='%b %Y')  # Convert 'Month_Year' to datetime
df['Month'] = df['Month_Year'].dt.month  # Extract month from 'Month_Year'
df['Year'] = df['Month_Year'].dt.year  # Extract year from 'Month_Year'

# Define features and target
X = df[['Month', 'Year', 'Marketing_Spend', 'Number_of_Employees']]  # Features: month, year, marketing spend, number of employees
y = df['Profit']  # Target: profit

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  # 80% training, 20% testing

# Define the model
model = RandomForestRegressor(random_state=42)

# Define the parameter grid
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Perform Grid Search
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, n_jobs=-1, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)

# Get the best model
best_model = grid_search.best_estimator_

# Evaluate the model
y_pred = best_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# Save the model
model_filename = 'financial_model.pkl'
try:
    joblib.dump(best_model, model_filename)  # Save the trained model to a file
    if os.path.exists(model_filename):
        print(f"Model saved successfully as '{model_filename}'")
    else:
        print(f"Failed to save the model to '{model_filename}'")
except Exception as e:
    print(f"An error occurred while saving the model: {e}")