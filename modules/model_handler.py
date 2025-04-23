import os
import torch
from resources.cnn_model import TeakGradeCNN
from torchvision import  models
import torch.nn as nn

is_loading_saved_model = True

def load_model(model_name):

    if model_name=="ResNet18":
        model = models.resnet18(pretrained=True)

        # Modify the final layer to hold 4 classes (Quality Grades)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 4)

        if is_loading_saved_model:
            # Check if the model file exists before loading
            if os.path.exists("resources/best_model_wts_optimized_resnet18.pth"):
                model.load_state_dict(torch.load("resources/best_model_wts_optimized_resnet18.pth"))
            else:
                print("Could not locate the saved model")
        return model

    else:
        model = TeakGradeCNN(num_classes=4)

        if is_loading_saved_model:
            # Check if the model file exists before loading
            if os.path.exists("resources/best_model_wts_baseline.pth"):
                model.load_state_dict(torch.load("resources/best_model_wts_baseline.pth"))
            else:
                print("Could not locate the saved model")
        return model
    


    