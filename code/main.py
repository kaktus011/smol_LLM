import pandas as pd
import matplotlib.pyplot as plt

# Define the data
data = {
    'Month': ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
    'Income': [5000, 4500, 4800, 4700, 5100, 5300, 4900, 4600, 5400, 5500, 4700, 4800],
    'Expenses': [3000, 3200, 3100, 3300, 2900, 3000, 3200, 3300, 2900, 2950, 3100, 3400]
}

# Create DataFrame
df = pd.DataFrame(data)

# Calculate Profit
df['Profit'] = df['Income'] - df['Expenses']

# Print DataFrame
print(df)

# Plot Income, Expenses, and Profit
plt.figure(figsize=(10, 6))
plt.plot(df['Month'], df['Income'], label='Income')
plt.plot(df['Month'], df['Expenses'], label='Expenses')
plt.plot(df['Month'], df['Profit'], label='Profit')

# Add titles and labels
plt.title('Monthly Financial Overview')
plt.xlabel('Month')
plt.ylabel('Amount ($)')
plt.legend()

# Show plot
plt.show()

# Generate insights
insights = []
for i in range(1, len(df)):
    insight = f"Input: {df['Month'][i-1]} - Income ${df['Income'][i-1]}, Expenses ${df['Expenses'][i-1]}; {df['Month'][i]} - Income ${df['Income'][i]}, Expenses ${df['Expenses'][i]}. "
    if df['Expenses'][i] > df['Expenses'][i-1]:
        insight += "Expenses rose slightly, "
    else:
        insight += "Expenses decreased, "
    
    if df['Income'][i] > df['Income'][i-1]:
        insight += "income increased."
    else:
        insight += "income decreased."
    
    insights.append(insight)

# Print insights
for insight in insights:
    print(insight)