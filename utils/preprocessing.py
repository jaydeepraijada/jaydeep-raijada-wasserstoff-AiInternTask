
import torchvision.transforms as T


def transform_image(image):
    transform = T.Compose([
        T.Resize((256, 256)),  # Resize to match model input
        T.ToTensor(),          # Convert to torch tensor
        T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize
    ])
    return transform(image).unsqueeze(0) # Add batch dimension to get [1,C,H,W] #C is channels, RGB has 3, greyscale has 1