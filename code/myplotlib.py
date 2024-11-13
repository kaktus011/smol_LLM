import tkinter as tk

class MyPyPlot:
    def __init__(self):
        self.figures = []
        self.figsize = (1200, 800)
        self.left_margin = 50
        self.right_margin = 150
        self.bottom_margin = 100
        self.root = tk.Tk()
        self.canvas = tk.Canvas(self.root, width=self.figsize[0] + self.left_margin + self.right_margin, height=self.figsize[1] + self.bottom_margin)
        self.canvas.pack()

    def figure(self, figsize=(1200, 800)):
        self.figsize = figsize
        self.canvas.config(width=self.figsize[0] + self.left_margin + self.right_margin, height=self.figsize[1] + self.bottom_margin)
        self.canvas.delete("all")
        self.draw_grid()

    def draw_grid(self):
        width, height = self.figsize
        for i in range(self.left_margin, width + self.left_margin, width // 10):
            self.canvas.create_line([(i, 0), (i, height)], tag='grid_line', fill='gray', dash=(2, 2))
        for i in range(0, height, height // 10):
            self.canvas.create_line([(self.left_margin, i), (width + self.left_margin, i)], tag='grid_line', fill='gray', dash=(2, 2))

    def plot(self, x, y, label=None, color="blue"):
        width, height = self.figsize
        if isinstance(x[0], str):
            x_labels = list(x)  # Convert to list
            x = list(range(len(x)))  # Convert string x values to numerical indices
        else:
            x_labels = None
        max_x, max_y = max(x), max(y)
        min_x, min_y = min(x), min(y)
        scaled_x = [(xi - min_x) / (max_x - min_x) * width + self.left_margin for xi in x]
        scaled_y = [height - (yi - min_y) / (max_y - min_y) * height for yi in y]
        self.canvas.create_line(list(zip(scaled_x, scaled_y)), fill=color, width=2)
        if label:
            self.figures.append((scaled_x, scaled_y, label, color))
        if x_labels:
            self.add_x_labels(x_labels, scaled_x)

    def add_x_labels(self, labels, positions):
        for label, pos in zip(labels, positions):
            self.canvas.create_text(pos, self.figsize[1] + 40, text=label, font=("Arial", 8), angle=90)

    def title(self, title):
        self.canvas.create_text((self.figsize[0] + self.left_margin + self.right_margin) // 2, 20, text=title, font=("Arial", 16))

    def legend(self):
        legend_x = self.figsize[0] + self.left_margin + 20
        legend_y = 50
        for _, _, label, color in self.figures:
            self.canvas.create_rectangle(legend_x, legend_y, legend_x + 20, legend_y + 10, fill=color)
            self.canvas.create_text(legend_x + 30, legend_y + 5, text=label, anchor='w')
            legend_y += 20