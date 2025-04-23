import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageDraw, ImageChops
import numpy as np
from neural_network import NN

class SimplePaintApp:
    def __init__(self, root, model):
        self.root = root
        self.root.title("''НЕЙРО-СЕТЬ''")
        self.root.geometry("290x310")
        self.model = model
        
        self.drawing = False
        self.last_x = 0
        self.last_y = 0
        self.color = "black"
        self.brush_size = 10
        self.pred = 'Нарисуйте число'
        
        self.canvas = tk.Canvas(self.root, bg="white", width=280, height=280)
        self.canvas.pack()
        
        self.create_toolbar()
        
        self.canvas.bind("<Button-1>", self.start_drawing)
        self.canvas.bind("<B1-Motion>", self.draw)
        self.canvas.bind("<ButtonRelease-1>", self.stop_drawing)
        
        self.image = Image.new("L", (280, 280), "white")
        print(self.image)
        self.draw_image = ImageDraw.Draw(self.image)
    
    def create_toolbar(self):
        toolbar = ttk.Frame(self.root)
        toolbar.pack(fill=tk.X, padx=2, pady=2)
        self.prediction_label = ttk.Label(toolbar, text=str(self.pred))
        self.prediction_label.pack(side=tk.LEFT, padx=2)
        
        ttk.Button(
            toolbar, 
            text="Очистить", 
            command=self.clear_canvas
        ).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(
            toolbar, 
            text="Распознать", 
            command=self.get_pred,
        ).pack(side=tk.LEFT, padx=5)
    
    def start_drawing(self, event):
        self.drawing = True
        self.last_x = event.x
        self.last_y = event.y
    
    def draw(self, event):
        if self.drawing:
            self.canvas.create_line(
                self.last_x, self.last_y, event.x, event.y,
                width=self.brush_size, fill=self.color, capstyle=tk.ROUND, smooth=True
            )
            
            self.draw_image.line(
                [(self.last_x, self.last_y), (event.x, event.y)],
                fill=self.color,
                width=self.brush_size
            )
            
            self.last_x = event.x
            self.last_y = event.y
    
    def stop_drawing(self, event):
        self.drawing = False
    
    def clear_canvas(self):
        self.canvas.delete("all")
        self.image = Image.new("L", (self.canvas.winfo_width(), self.canvas.winfo_height()), "white")
        self.draw_image = ImageDraw.Draw(self.image)
    
    def get_pred(self):
        width = self.canvas.winfo_width()
        height = self.canvas.winfo_height()
        data = ImageChops.invert(self.image.crop((0, 0, width, height)).resize((28, 28)))
        self.pred = np.argmax(self.model.forward(np.asarray(data, dtype=np.uint8).flatten().T))
        # a = model.z3
        # Image.fromarray(np.uint8(a * 255)).show()
        self.prediction_label.config(text=str(self.pred))
        print(model.softmax(model.z3))


if __name__ == "__main__":
    root = tk.Tk()
    model = NN(784, 256, 64, 10)
    model.load()
    app = SimplePaintApp(root, model)
    root.mainloop()