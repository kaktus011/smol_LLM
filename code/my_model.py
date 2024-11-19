import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os

def load_data(filepath):
    return pd.read_csv(filepath)

def preprocess_data(df):
    df['Month_Year'] = pd.to_datetime(df['Month_Year'], format='%b %Y')
    df['Month'] = df['Month_Year'].dt.month
    df['Year'] = df['Month_Year'].dt.year
    return df

def define_features_and_target(df):
    X = df[['Income', 'Expenses']]
    y = df['Profit']
    return X, y

def train_model(X_train, y_train, param_grid):
    model = RandomForestRegressor(random_state=42)
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, n_jobs=-1, scoring='neg_mean_squared_error')
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f'Mean Squared Error: {mse}')
    print(f'R^2 Score: {r2}')
    return y_pred

def save_model(model, filename):
    try:
        joblib.dump(model, filename)
        if os.path.exists(filename):
            print(f"Model saved successfully as '{filename}'")
        else:
            print(f"Failed to save the model to '{filename}'")
    except Exception as e:
        print(f"An error occurred while saving the model: {e}")

def save_predictions(y_test, y_pred, filename):
    try:
        predictions_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
        predictions_df.to_csv(filename, index=False)
        if os.path.exists(filename):
            print(f"Predictions saved successfully as '{filename}'")
        else:
            print(f"Failed to save the predictions to '{filename}'")
    except Exception as e:
        print(f"An error occurred while saving the predictions: {e}")

if __name__ == "__main__":
    df = load_data('data/financial_training_data.csv')
    df = preprocess_data(df)
    X, y = define_features_and_target(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    best_model = train_model(X_train, y_train, param_grid)
    y_pred = evaluate_model(best_model, X_test, y_test)
    save_model(best_model, 'financial_model.pkl')
    save_predictions(y_test, y_pred, 'train_predictions.csv')