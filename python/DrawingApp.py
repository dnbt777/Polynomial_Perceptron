import tkinter as tk
from PIL import Image, ImageDraw
import numpy as np

class DrawingApp:
    def __init__(self, root, model):
        self.root = root
        self.model = model
        self.canvas = tk.Canvas(root, width=280, height=280, bg='white')
        self.canvas.pack()
        self.canvas.bind("<B1-Motion>", self.paint)
        self.button_clear = tk.Button(root, text="Clear", command=self.clear)
        self.button_clear.pack()
        self.button_predict = tk.Button(root, text="Predict", command=self.predict)
        self.button_predict.pack()
        self.image = Image.new("L", (280, 280), 255)  # Use a larger image for drawing
        self.draw = ImageDraw.Draw(self.image)

    def paint(self, event):
        x1, y1 = (event.x - 1), (event.y - 1)
        x2, y2 = (event.x + 1), (event.y + 1)
        self.canvas.create_oval(x1, y1, x2, y2, fill="black", width=10)
        self.draw.line([x1, y1, x2, y2], fill=0, width=10)

    def clear(self):
        self.canvas.delete("all")
        self.image = Image.new("L", (280, 280), 255)  # Use a larger image for drawing
        self.draw = ImageDraw.Draw(self.image)  # Reinitialize the ImageDraw object

    def predict(self):
        print(self.image)
        img = self.image.resize((28, 28))  # Resize the larger image to 28x28
        img = np.array(img).reshape(1, 784)
        img = 255 - img  # Invert colors
        img = img / 255.0  # Normalize
        yhat = self.model.forward(img.flatten())
        print("Predicted probabilities:", yhat)
        prediction = np.argmax(yhat)
        print("Prediction: ", prediction)

# Example usage:
# root = tk.Tk()
# model = YourModelClass()  # Replace with your model class
# app = DrawingApp(root, model)
# root.mainloop()