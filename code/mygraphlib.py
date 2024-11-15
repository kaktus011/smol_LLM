import pandas as pd
import tkinter as tk
import math

def plot_pie_chart(csv_file):
    # Read the data from the CSV file
    df = pd.read_csv(csv_file)

    # Calculate total Income, Expenses, and Profit
    total_income = df['Income'].sum()
    total_expenses = df['Expenses'].sum()
    total_profit = df['Profit'].sum()

    # Data for the pie chart
    pie_data = {
        'Income': total_income,
        'Expenses': total_expenses,
        'Profit': total_profit
    }

    colors = ['gold', 'lightcoral', 'lightskyblue']

    def draw_pie_chart(canvas, data, colors):
        total = sum(data.values())
        start_angle = 0

        for i, (key, value) in enumerate(data.items()):
            extent = (value / total) * 360
            canvas.create_arc((50, 50, 350, 350), start=start_angle, extent=extent, fill=colors[i])
            start_angle += extent

            # Draw labels
            mid_angle = start_angle - extent / 2
            x = 200 + 150 * math.cos(math.radians(mid_angle))
            y = 200 - 150 * math.sin(math.radians(mid_angle))
            canvas.create_text(x, y, text=key, fill="black")

    # Create the main window
    root = tk.Tk()
    root.title("Financial Pie Chart")

    # Create a canvas widget
    canvas = tk.Canvas(root, width=400, height=400)
    canvas.pack()

    # Draw the pie chart
    draw_pie_chart(canvas, pie_data, colors)

    # Run the application
    root.mainloop()