import os
import pathlib
import numpy as np

import tkinter as tk
from tkinter import messagebox

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import torchvision as tv
from torchvision.transforms import v2
from torchvision import datasets

from models import DenseNet
from trainer import Trainer

DATA_DIR = pathlib.Path("data")
DATA_DIR.mkdir(exist_ok=True)
MODELS_DIR = pathlib.Path("models")
MODELS_DIR.mkdir(exist_ok=True)
MODEL_PATH = MODELS_DIR / "trained_model.pt"

# Load the model
model = DenseNet()

# Check if a trained model exists, load if present, otherwise train
if os.path.exists(MODEL_PATH):
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()
    print("Loaded pre-trained model.")
    # load the trainer after loading the model, for finetuning
    trainer = Trainer(model)
else:
    # Load MNIST data
    transform = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])
    train_dataset = datasets.MNIST(
        root=DATA_DIR, train=True, transform=transform, download=True
    )
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    # Train the model
    trainer = Trainer(model)
    trainer.train(train_loader, epochs=5)
    torch.save(model.state_dict(), MODEL_PATH)
    print("Model trained and saved.")

# Define Tkinter GUI
class DigitRecognizerApp:
    def __init__(self, root, model):
        self.root = root
        self.model = model

        # --------------------------------------------------------------------------------
        # Set up main frame
        self.main_frame = tk.Frame(root)
        self.main_frame.pack(pady=10, padx=10)

        # --------------------------------------------------------------------------------
        # Canvas for drawing
        self.canvas = tk.Canvas(self.main_frame, width=280, height=280, bg="white")
        self.scaled_canvas = np.zeros((28, 28))
        self.canvas.grid(row=0, column=0, rowspan=20, pady=10, padx=10)
        self.canvas.bind("<B1-Motion>", self.draw)

        # --------------------------------------------------------------------------------
        # Set up Matplotlib figure for probabilities
        self.probabilities_fig, self.ax = plt.subplots(figsize=(1.4, 1.4))
        self.ax.set_xlabel("Digits")
        self.ax.set_ylabel("Probability")
        self.ax.set_xticks(range(10))
        self.ax.tick_params(axis="both", which="major", labelsize=2.5)
        self.bar_container = self.ax.bar(range(10), [0.0] * 10)
        self.ax.set_ybound(lower=0, upper=1.1)

        # Embed the plot into Tkinter window
        self.plot_canvas = FigureCanvasTkAgg(
            self.probabilities_fig, master=self.main_frame
        )
        self.plot_canvas.get_tk_widget().grid(
            row=0, column=1, rowspan=20, padx=10, pady=10
        )
        self.plot_canvas.draw()

        # --------------------------------------------------------------------------------
        # Buttons
        row, col = 0, 40
        rowspan = 8

        # Prediction button
        self.predict_button = tk.Button(
            self.main_frame, text="Predict", command=self.predict
        )
        self.predict_button.grid(
            row=row, column=col, rowspan=rowspan, pady=10, sticky="nesw"
        )

        # Clear button
        self.clear_button = tk.Button(
            self.main_frame, text="Clear", command=self.clear_canvas
        )
        self.clear_button.grid(
            row=row,
            column=col + 1,
            rowspan=rowspan,
            columnspan=1,
            pady=10,
            sticky="nesw",
        )

        row += rowspan
        rowspan = 4

        # Finetune button
        self.finetune_button = tk.Button(
            self.main_frame, text="Finetune", command=self.finetune
        )
        self.finetune_button.grid(
            row=row, column=col, rowspan=rowspan, pady=5, sticky="nesw"
        )

        # Entry for user label
        tk.Label(self.main_frame, text="Finetuning label:").grid(
            row=row, column=col + 1, pady=5, columnspan=1, sticky="e"
        )
        self.user_label_entry = tk.Entry(self.main_frame, width=7)
        self.user_label_entry.grid(
            row=row + 1, column=col + 1, columnspan=1, pady=5, sticky="e"
        )

        row += rowspan + 2

        # Load MNIST image button
        self.load_image_button = tk.Button(
            self.main_frame,
            text="Load Random\nMNIST Image",
            command=self.load_random_image,
        )
        self.load_image_button.grid(
            row=row, column=col, rowspan=rowspan, columnspan=1, pady=5, sticky="nesw"
        )

        # Option menu for class selection
        tk.Label(self.main_frame, text="From class:").grid(
            row=row, column=col + 1, pady=5, sticky="e"
        )
        self.selected_class = tk.StringVar(self.main_frame)
        self.selected_class = tk.StringVar(self.main_frame)
        self.class_option_menu = tk.OptionMenu(
            self.main_frame, self.selected_class, *["any"] + list(range(10))
        )
        self.selected_class.set("any")  # Default to 'any' class
        self.class_option_menu.config(bg="white", fg="black")
        self.class_option_menu.grid(row=row + 1, column=col + 1, pady=5, sticky="e")

    def draw(self, event):
        x, y = event.x, event.y
        r = 1  # Size of scaled brush for the 280x280 canvas
        # Draw on the large canvas
        self.canvas.create_oval(x - r, y - r, x + r, y + r, fill="black", outline="")
        # Update the 28x28 scaled version
        scaled_x, scaled_y = x // 10, y // 10
        if 0 <= scaled_x < 28 and 0 <= scaled_y < 28:
            # Distribute the brush effect to surrounding pixels to simulate a smoother boundary
            for dx in range(-1, 1):
                for dy in range(-1, 1):
                    nx, ny = scaled_x + dx, scaled_y + dy
                    if 0 <= nx < 28 and 0 <= ny < 28:
                        distance = max(abs(dx), abs(dy))
                        self.scaled_canvas[ny, nx] += 1.5 / (distance + 1)
            self.scaled_canvas = np.clip(
                self.scaled_canvas, 0, 1
            )  # Ensure values stay between 0 and 1

        # Apply a Gaussian blur to simulate anti-aliasing
        from scipy.ndimage import gaussian_filter

        blurred_canvas = gaussian_filter(self.scaled_canvas, sigma=0.5)

        # Redraw the canvas to show the scaled-up, pixelated version
        self.canvas.delete("all")
        for i in range(28):
            for j in range(28):
                intensity = blurred_canvas[i, j]
                if intensity > 0:
                    color = int((1.0 - intensity) * 255)
                    x0, y0 = j * 10, i * 10
                    x1, y1 = x0 + 10, y0 + 10
                    self.canvas.create_rectangle(
                        x0,
                        y0,
                        x1,
                        y1,
                        fill=f"#{color:02x}{color:02x}{color:02x}",
                        outline="",
                    )

    def clear_canvas(self):
        self.canvas.delete("all")
        self.scaled_canvas = np.zeros((28, 28))
        # Remove prediction probabilities on the plot
        for rect in self.bar_container:
            rect.set_height(0)
        self.plot_canvas.draw()

    def predict(self):
        canvas_data = self.get_canvas_image()
        with torch.no_grad():
            output = self.model(canvas_data)
            probabilities = F.softmax(output, dim=1).cpu().numpy().flatten()
            predicted_label = np.argmax(probabilities)
        # print([(i, p.item()) for i, p in enumerate(probabilities)])

        # Update prediction probabilities on the plot
        for rect, prob in zip(self.bar_container, probabilities):
            rect.set_height(prob)
        self.ax.set_title(f"Predicted Digit: {predicted_label}", fontsize=6)
        self.plot_canvas.draw()

    def finetune(self):
        user_label = self.user_label_entry.get()
        if not user_label.isdigit() or not (0 <= int(user_label) <= 9):
            messagebox.showerror("Error", "Please enter a valid digit (0-9).")
            return
        label = torch.tensor([int(user_label)], dtype=torch.long)
        canvas_data = self.get_canvas_image()
        trainer.finetune(canvas_data, label)
        self.predict()
        # torch.save(self.model.state_dict(), MODEL_PATH)  # Save the updated model

    def get_canvas_image(self):
        self.canvas_path = DATA_DIR / "canvas.eps"
        self.canvas.postscript(file=self.canvas_path)
        from PIL import Image

        with Image.open(self.canvas_path) as img:
            img = img.resize((28, 28)).convert("L")
            img_data = np.array(img)
            img_data = 1.0 - img_data / 255.0  # Invert colors
            img_data = (
                torch.tensor(img_data, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            )
        return img_data

    def load_random_image(self):
        import random

        # Choose a random image from the train or test dataset
        transform = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])
        dataset = datasets.MNIST(
            root=DATA_DIR, train=False, transform=transform, download=True
        )

        selected_class = self.selected_class.get()
        if selected_class == "any":
            # If no specific class is selected, choose any random image
            idx = random.randint(0, len(dataset) - 1)
        else:
            # If a specific class is selected, filter the dataset by that class
            class_label = int(selected_class)
            indices = [
                i for i, (_, label) in enumerate(dataset) if label == class_label
            ]
            idx = random.choice(indices)

        image, _ = dataset[idx]
        image = image.squeeze().numpy() * 255.0

        # Clear the canvas and draw the image
        self.clear_canvas()
        for i in range(28):
            for j in range(28):
                color = int(255 - image[i, j])
                if color < 255:  # Only draw non-white pixels
                    x0, y0 = j * 10, i * 10
                    x1, y1 = x0 + 10, y0 + 10
                    self.canvas.create_rectangle(
                        x0,
                        y0,
                        x1,
                        y1,
                        fill=f"#{color:02x}{color:02x}{color:02x}",
                        outline="",
                    )

        # Predict the digit after loading the random image
        self.predict()


if __name__ == "__main__":
    root = tk.Tk()
    root.title("Interactive Finetuner")
    app = DigitRecognizerApp(root, model)
    # quit commands
    root.bind("<Control-c>", lambda e: root.quit())
    root.bind("<Command-c>", lambda e: root.quit())
    root.bind("<Command-c>", lambda e: root.quit())
    root.bind("<Command-q>", lambda e: root.quit())
    root.mainloop()
