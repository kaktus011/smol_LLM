import pandas as pd
from myplotlib import MyPyPlot

# Define the data
data = {
    'Month': ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec',
              'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
    'Year': [2022] * 12 + [2023] * 12,
    'Income': [5000, 4500, 4800, 4700, 5100, 5300, 4900, 4600, 5400, 5500, 4700, 4800,
               5200, 5600, 5800, 6000, 6200, 6400, 6600, 6800, 7000, 7200, 7400, 7600],
    'Expenses': [3000, 3200, 3100, 3300, 2900, 3000, 3200, 3300, 2900, 2950, 3100, 3400,
                 3500, 3600, 3700, 3800, 3900, 4000, 4100, 4200, 4300, 4400, 4500, 4600]
}

# Create DataFrame
df = pd.DataFrame(data)

# Combine Month and Year for x-axis labels
df['Month_Year'] = df['Month'] + ' ' + df['Year'].astype(str)

# Calculate Profit
df['Profit'] = df['Income'] - df['Expenses']

# Plot Income, Expenses, and Profit using MyPyPlot
plot = MyPyPlot()
plot.figure(figsize=(1200, 800))  # Increased size in pixels for tkinter canvas
plot.plot(df['Month_Year'], df['Income'], label='Income', color='blue')
plot.plot(df['Month_Year'], df['Expenses'], label='Expenses', color='red')
plot.plot(df['Month_Year'], df['Profit'], label='Profit', color='green')

# Add titles and labels
plot.title('Monthly Financial Overview')
plot.root.mainloop()