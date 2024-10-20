import streamlit as st
import pandas as pd
import numpy as np
import torch.nn.functional as F
import os
import torch
import torch.nn as nn
from torchvision import transforms, models
from cnn_model import KneeOACNN
from PIL import Image
import pickle
import cv2
import base64
from openai import OpenAI
from dotenv import load_dotenv

is_loading_saved_model = True

mean_std = "mean_std.pkl"

number_of_severity_levels = 5
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
severity_dict = {0:"Healthy" , 1:"Doubtful", 2:"Minimal" ,3:"Moderate",4:"Severe"}

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.forward_output = None
        self.model.eval()
        
        # Register hooks for gradients
        self.target_layer.register_forward_hook(self.forward_hook)
        self.target_layer.register_backward_hook(self.backward_hook)

    def forward_hook(self, module, input, output):
        self.forward_output = output  # Save the output of the layer

    def backward_hook(self, module, grad_in, grad_out):
        self.gradients = grad_out[0]  # Save the gradients

    def generate_heatmap(self, input_image, target_class=None):
        # Forward pass
        output = self.model(input_image)
        if target_class is None:
            target_class = torch.argmax(output)  # Use predicted class if not provided

        # Zero all gradients and do backward pass
        self.model.zero_grad()
        output[:, target_class].backward(retain_graph=True)

        # Generate Grad-CAM heatmap
        weights = torch.mean(self.gradients, dim=(2, 3))  # Global average pooling
        grad_cam_map = torch.sum(weights[:, :, None, None] * self.forward_output, dim=1).squeeze()

        # Apply ReLU and normalize heatmap
        grad_cam_map = F.relu(grad_cam_map)
        grad_cam_map = grad_cam_map.cpu().data.numpy()
        heatmap = (grad_cam_map - grad_cam_map.min()) / (grad_cam_map.max() - grad_cam_map.min())  # Normalize

        return heatmap

def overlay_heatmap(heatmap, img_path, alpha=0.5, colormap=cv2.COLORMAP_JET):
    # Load the image
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Resize heatmap to match the input image size
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))

    # Convert heatmap to a color map
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, colormap)

    # Overlay heatmap on the image
    overlayed_img = cv2.addWeighted(img, alpha, heatmap, 1 - alpha, 0)

    return overlayed_img

def explian_image(model_name,model,preprocessed_image ,img_path):
    if model_name == "ResNet18":
        grad_cam = GradCAM(model, target_layer=model.layer4)
    else:
        grad_cam = GradCAM(model, target_layer=model.conv4)  # Target last conv layer
    
    heatmap = grad_cam.generate_heatmap(preprocessed_image)
    # Overlay heatmap on original image
    overlayed_image = overlay_heatmap(heatmap, img_path)

    return overlayed_image


def load_model(model_name):

    if model_name=="ResNet18":
        model = models.resnet18(pretrained=True)

        # Modify the final layer to hold 5 classes (severity levels)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 5)

        if is_loading_saved_model:
            # Check if the model file exists before loading
            if os.path.exists("best_model_wts_optimized_resnet18.pth"):
                model.load_state_dict(torch.load("best_model_wts_optimized_resnet18.pth"))
            else:
                print("Could not locate the saved model")
        return model

    else:
        model = KneeOACNN(num_classes=5)

        if is_loading_saved_model:
            # Check if the model file exists before loading
            if os.path.exists("best_model_wts_optimized_2.pth"):
                model.load_state_dict(torch.load("best_model_wts_optimized_2.pth"))
            else:
                print("Could not locate the saved model")
        return model

def load_image(uploaded_file):
    uploaded_image = Image.open(uploaded_file).convert('RGB')
    return uploaded_image


def preprocessing(image):

    with open(mean_std, 'rb') as f:
        data = pickle.load(f)

    mean = data['mean']
    std = data['std']

    pred_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),        
    transforms.Normalize(mean=mean, std=std)
    ])

    transformed_image = pred_transform(image)
    transformed_image = transformed_image.unsqueeze(0)

    return transformed_image

def convert_to_displayable_image(transformed_image):
    transformed_image = transformed_image.squeeze(0)  # Shape: (C, H, W)
    transformed_image = transformed_image.permute(1, 2, 0).numpy()  # Convert to (H, W, C) format

    # Normalize the values to the [0, 1] range for display
    # Since the values might be in the [-1, 1] range (common for normalized tensors), 
    # scale them to [0, 1] for proper display
    transformed_image = (transformed_image - transformed_image.min()) / (transformed_image.max() - transformed_image.min())

    # Convert to 8-bit format for display
    transformed_image = (transformed_image * 255).astype(np.uint8)
    return transformed_image

def identify_severity(model,image):
    model.eval()
    input_image = image.to(device)
    with torch.no_grad():
        outputs = model(input_image) 
        _, predicted = torch.max(outputs, 1)

    severity_level = predicted.item()
    severity_level_label = severity_dict[severity_level]

    # finding output probabilities
    probabilities = F.softmax(outputs, dim=1)
    probabilities_np = probabilities.detach().cpu().numpy().squeeze()

    return severity_level, severity_level_label, probabilities_np

# Function to encode images in order to upload to gpt API
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')
    
# Function which call the API for image captioning
def gpt_api(client,system_prompt,img_url, title):
    encoded_image = encode_image(img_url)
    response = client.chat.completions.create(
    model="gpt-4-turbo",
    messages=[
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user",
            "content": [
                {"type": "image_url",
                "image_url": {
                "url": f"data:image/jpeg;base64,{encoded_image}"
                }
                },
            ],
        },
        {
            "role": "user",
            "content": title
        }
    ],
        max_tokens=300,
        top_p=0.1
    )

    return response.choices[0].message.content


def get_image_analysis(image_path):

    load_dotenv()

    client = OpenAI()

    system_prompt = '''
    You will be provided with an X-ray image of knee.
    Provide the signs of osteoarthritis.
    provide the severity level at last in short.
    '''
    title = "X-Ray Image of Knee"

    return gpt_api(client,system_prompt,image_path,title)

def format_page():
    st.sidebar.markdown(
    """
    <style>
    .sidebar-footer {
        position: relative;
        bottom: 0;
        width: 100%;
        background-color: #ffffff;
        padding: 10px;
        font-size: 12px;
        color: #666;
        border-radius: 5px;
        box-shadow: 0 0 5px rgba(0,0,0,0.1);
        margin-top: 20px;  /* Add margin to separate from other sidebar content */
    }
    </style>
    <div class="sidebar-footer">
        <p>Developed by D. Shan Siraj</p>
    </div>
    """,
    unsafe_allow_html=True
    )

def create_output_prob_df(probabilities_np):
    # Create class labels' list
    class_labels = list(severity_dict.values())

    df = pd.DataFrame({
    'Severity Level': class_labels,
    'Probability Value': probabilities_np,
    'Probability Percentage (%)': (probabilities_np * 100).round(2)  # Convert to percentage
    })

    return df

# Highlight the row with the highest probability
def highlight_max(s):
    is_max = s == s.max()
    return ['background-color: yellow' if v else '' for v in is_max]

def highlight_max_row(row, max_val):
    return ['background-color: yellow' if row['Probability Percentage (%)'] == max_val else '' for _ in row]


def main():

    # Set the page layout to wide
    st.set_page_config(layout="wide")
    st.title('Knee Osteoarthritis X-Ray Analysis')
    st.sidebar.title("Knee Osteoarthritis")
    st.sidebar.header('User Input')

    uploaded_file = st.sidebar.file_uploader(label="Upload X-Ray Image:",type=["png","jpg","jpeg"])

    model_names = ['Custom CNN', 'ResNet18']
    selected_model = st.sidebar.selectbox('Select CNN Model', model_names)

    model = load_model(selected_model)

    # Button to trigger the activity
    if st.sidebar.button('Start Analysis from CNN'):

        if uploaded_file:

            original_image = load_image(uploaded_file)

            img_path = f"temp_image.jpg"  # Temporary path to save image
            original_image.save(img_path)

            preprocessed_image = preprocessing(original_image)

            transformed_image = convert_to_displayable_image(preprocessed_image)

            # Create interpreted image
            overlayed_image = explian_image(selected_model,model,preprocessed_image ,img_path)

            severity_level, severity_level_text, probabilities_np = identify_severity(model,preprocessed_image)

            # Create a DataFrame for severity levels and probabilities
            df = create_output_prob_df(probabilities_np)
            # styled_df = df.style.apply(highlight_max, subset=['Probability Percentage (%)'])

            max_val = df['Probability Percentage (%)'].max()

            # Apply the styling function to the entire DataFrame
            styled_df = df.style.apply(highlight_max_row, axis=1, max_val=max_val)

            # styled_df = df.style.apply(highlight_max_row, axis=1)

            st.markdown(f"<h3><strong>Identified Severity Level : {severity_level} - {severity_level_text}</strong></h3>", unsafe_allow_html=True)

            # Display components
            col1, col2,col3 = st.columns([1,1,1])

            with col1:
                st.write("Original X-Ray Image")
                st.image(original_image, use_column_width=False)

            with col2:
                st.write("Transformed X-Ray Image")
                st.image(transformed_image, use_column_width=False)

            with col3:
                st.write("Interpreted X-Ray Image")
                st.image(overlayed_image, caption="Grad-CAM Interpretation", use_column_width=False)
           
            col1, col2= st.columns([1,1])

            with col1:
                # Display the probability values for each severity levels
                st.write("Output probabilities")
                st.write(styled_df)

            with col2:
                # Display output probabilities in a barchart
                st.write("Output probability chart")
                st.bar_chart(probabilities_np)

        else:
            st.error("Image is not loaded")

    # Button to trigger the activity
    if st.sidebar.button('Start Analysis from genAI'):

        if uploaded_file:

            original_image = load_image(uploaded_file)

            img_path = f"temp_image.jpg"  # Temporary path to save image
            original_image.save(img_path)

            preprocessed_image = preprocessing(original_image)

            severity_level, severity_level_text, probabilities_np = identify_severity(model,preprocessed_image)

            st.markdown(f"<h3><strong>Identified Severity Level : {severity_level} - {severity_level_text}</strong></h3>", unsafe_allow_html=True)

            # Display components
            col1, col2 = st.columns([1,3])

            with col1:
                st.write("Original X-Ray Image")
                st.image(original_image, use_column_width=False)

            with col2:
                st.write("Gen AI Textual Analysis")
                image_analysis = get_image_analysis(img_path)
                st.write(image_analysis)

        else:
            st.error("Image is not loaded")

    format_page()

if __name__ == "__main__":
    main()

