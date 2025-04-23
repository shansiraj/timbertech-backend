import cv2
import torch
import torch.nn.functional as F
import numpy as np

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