import torch
import torch.nn as nn
from torchvision import models, datasets
import pandas as pd
import numpy as np
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

grade_dict = {0:"D" , 1:"C", 2:"B" ,3:"A"}

# Highlight the row with the highest probability
def highlight_max_row(row, max_val):
    return ['background-color: yellow' if row['Probability Percentage (%)'] == max_val else '' for _ in row]

def get_grade_results(model,image):
    model.eval()
    input_image = image.to(device)
    with torch.no_grad():
        outputs = model(input_image) 
        _, predicted = torch.max(outputs, 1)

    grade = predicted.item()
    grade_label = grade_dict[grade]

    # finding output probabilities
    probabilities = F.softmax(outputs, dim=1)
    probabilities_np = probabilities.detach().cpu().numpy().squeeze()

    return grade, grade_label, probabilities_np

def create_output_prob_df(probabilities_np):
    # Create class labels' list
    class_labels = list(grade_dict.values())

    df = pd.DataFrame({
    'Grade': class_labels,
    'Probability Value': probabilities_np,
    'Probability Percentage (%)': (probabilities_np * 100).round(2)  # Convert to percentage
    })

    df_sorted_by_index = df.sort_index(ascending=False)

    max_val = df_sorted_by_index['Probability Percentage (%)'].max()

    # Apply the styling function to the entire DataFrame
    styled_df = df_sorted_by_index.style.apply(highlight_max_row, axis=1, max_val=max_val)

    return styled_df

def create_output_prob_json(probabilities_np):

    # Create class labels' list
    class_labels = list(grade_dict.values())[::-1]

    # Reverse the probabilities to match label order
    probabilities_np = probabilities_np[::-1].astype(float)

    # Round to 1 decimal place
    grade_probability = {
        key: round(value, 1)
        for key, value in zip(class_labels, probabilities_np)
    }

    return grade_probability

def get_accumulated_prob(probabilities):
    # Convert to numpy array for easier manipulation
    # Take the mean across patches (axis=0)
    final_probabilities = np.mean(probabilities, axis=0)

    return final_probabilities