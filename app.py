import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import torch
import torchvision.models as models
import torchvision.transforms as transforms
import torch.nn as nn  

# Define class names (labels) as per dataset
class_names = ['cane', 'cavallo', 'elefante', 'farfalla', 'gallina', 'gatto', 'mucca', 'pecora', 'ragno', 'scoiattolo']

# Define the translation dictionary
translate = {"cane": "dog", "cavallo": "horse", "elefante": "elephant", "farfalla": "butterfly", "gallina": "chicken", 
             "gatto": "cat", "mucca": "cow", "pecora": "sheep", "scoiattolo": "squirrel", "dog": "cane", 
             "cavallo": "horse", "elephant" : "elefante", "butterfly": "farfalla", "chicken": "gallina", 
             "cat": "gatto", "cow": "mucca", "spider": "ragno", "squirrel": "scoiattolo"}

# Load the model
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, len(class_names))  # Adjust the final layer

# Load the saved weights
model.load_state_dict(torch.load('animal_classifier.pth', map_location=torch.device('cpu')))
model.eval()  # Set the model to evaluation mode

# Define the transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def predict_image(model, image_path, class_names, translate):
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0)  # Add batch dimension
    outputs = model(image)
    _, predicted = torch.max(outputs, 1)
    predicted_class = class_names[predicted.item()]
    translated_class = translate.get(predicted_class, predicted_class)
    return translated_class

def load_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        img = Image.open(file_path)
        img.thumbnail((300, 300))
        img = ImageTk.PhotoImage(img)
        panel.config(image=img)
        panel.image = img
        predicted_class = predict_image(model, file_path, class_names, translate)
        result_label.config(text=f'Predicted class: {predicted_class}')

# Create the GUI
root = tk.Tk()
root.title("Animal Classifier")

panel = tk.Label(root)
panel.pack(side="top", fill="both", expand="yes")

load_button = tk.Button(root, text="Load Image", command=load_image)
load_button.pack(side="bottom", fill="both", expand="yes", padx=10, pady=10)

result_label = tk.Label(root, text="", font=("Helvetica", 16))
result_label.pack(side="bottom", fill="both", expand="yes", padx=10, pady=10)

root.mainloop()
