import pickle
from torchvision import transforms

mean_std = "resources/mean_std.pkl"

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



