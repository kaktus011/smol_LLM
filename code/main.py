import pandas as pd
import joblib

def load_data(filepath):
    return pd.read_csv(filepath)

def preprocess_data(df):
    df['Month_Year'] = df['Month'] + ' ' + df['Year'].astype(str)
    df['Actual_Profit'] = df['Income'] - df['Expenses']
    df.to_csv('data/financial_data.csv', index=False)
    df['Month'] = pd.to_datetime(df['Month_Year'], format='%b %Y').dt.month
    df['Year'] = pd.to_datetime(df['Month_Year'], format='%b %Y').dt.year
    return df

def load_model(filepath):
    return joblib.load(filepath)

def make_predictions(model, df):
    X = df[['Month', 'Year', 'Marketing_Spend', 'Number_of_Employees']]
    df['Predicted_Profit'] = model.predict(X)
    return df

def display_predictions(df):
    print(df[['Month_Year', 'Income', 'Expenses', 'Marketing_Spend', 'Number_of_Employees', 'Actual_Profit', 'Predicted_Profit']])

if __name__ == "__main__":
    df = load_data('data/financial_data.csv')
    df = preprocess_data(df)
    model = load_model('financial_model.pkl')
    df = make_predictions(model, df)
    display_predictions(df)