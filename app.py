import torch
import torch.nn as nn
from torchvision import models
import gradio as gr
from typing import Tuple, Dict
from timeit import default_timer as timer

class DogEmotionResNet(nn.Module):
    def __init__(self, num_classes, weights=None):
        super().__init__()
        
        self.resnet = models.resnet50(weights=weights)
        for param in self.resnet.parameters():
            param.requires_grad = False
        
        in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(in_features, num_classes)
    
    def forward(self, x):
        return self.resnet(x)

def load_model(weights_path: str, num_classes: int) -> DogEmotionResNet:
    resnet_weights = models.ResNet50_Weights.DEFAULT
    model = DogEmotionResNet(num_classes=num_classes, weights=resnet_weights)
    model.load_state_dict(torch.load(weights_path))
    return model

def load_class_names(file_path: str) -> list:
    with open(file_path, "r") as f:
        class_names = [emotion.strip() for emotion in f.readlines()]
    return class_names

def predict(img) -> Tuple[Dict, float]:
    """Transforms and performs a prediction on img and returns prediction and time taken."""
    start_time = timer()
    
    img = resnet_transform(img).unsqueeze(0)
    
    model.eval()
    with torch.inference_mode():
        pred_probs = torch.softmax(model(img), dim=1)
    
    pred_labels_and_probs = {class_names[i]: float(pred_probs[0][i]) for i in range(len(class_names))}
    
    pred_time = round(timer() - start_time, 5)
    return pred_labels_and_probs, pred_time

# Load model and class names
weights_path = 'dog_emotion_model.pth'
class_names_file = "classes.txt"

model = load_model(weights_path, num_classes=4)
class_names = load_class_names(class_names_file)
resnet_weights = models.ResNet50_Weights.DEFAULT
resnet_transform = resnet_weights.transforms()

# Gradio Interface
title = "Dog Emotion Classifier 🐶🎭"
description = "This app classifies the emotion of a dog in an image into one of four categories: happy, sad, angry, or relaxed."
article = ""

demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs=[
        gr.Label(num_top_classes=5, label="Predictions"),
        gr.Number(label="Prediction time (s)"),
    ],
    title=title,
    description=description,
    article=article,
)

# Launch the app!
demo.launch()
