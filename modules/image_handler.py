from PIL import Image
import numpy as np

def load_image(uploaded_file):
    uploaded_image = Image.open(uploaded_file).convert('RGB')
    return uploaded_image

def convert_to_displayable_image(transformed_image):
    new_transformed_image = transformed_image.squeeze(0)  # Shape: (C, H, W)
    new_transformed_image = new_transformed_image.permute(1, 2, 0).numpy()  # Convert to (H, W, C) format

    # Normalize the values to the [0, 1] range for display
    # Since the values might be in the [-1, 1] range (common for normalized tensors), 
    # scale them to [0, 1] for proper display
    new_transformed_image = ((new_transformed_image - new_transformed_image.min()) / 
                (new_transformed_image.max() - new_transformed_image.min()))

    # Convert to 8-bit format for display
    new_transformed_image = (new_transformed_image * 255).astype(np.uint8)
    return new_transformed_image


def resize_image_maintain_aspect_ratio(image: Image.Image, target_size: int):
    """
    Resize an image while maintaining its aspect ratio.
    The target size is the longer side of the image, and the shorter side is scaled accordingly.
    """
    # Get the original dimensions
    width, height = image.size
    
    # Calculate aspect ratio
    aspect_ratio = width / height
    
    # Resize based on the target size (longer side)
    if width > height:
        # Width is the longer side
        new_width = target_size
        new_height = int(target_size / aspect_ratio)
    else:
        # Height is the longer side
        new_height = target_size
        new_width = int(target_size * aspect_ratio)
    
    # Resize the image
    return image.resize((new_width, new_height), Image.ANTIALIAS)

def divide_image_into_patches(image: Image.Image, patch_size: int):
    """
    Divide the input image into smaller patches of size patch_size x patch_size.
    This function splits the image into non-overlapping patches.
    """
    # Convert the image to a numpy array for easier manipulation
    image_np = np.array(image)
    
    patches = []
    img_width, img_height = image.size
    
    # Loop over the image and extract patches of size patch_size x patch_size
    for i in range(0, img_width, patch_size):
        for j in range(0, img_height, patch_size):
            # Get the region of interest
            patch = image_np[j:j + patch_size, i:i + patch_size]
            
            # Convert the patch back to a Pillow image and append to the list of patches
            patches.append(Image.fromarray(patch))

    return patches

    