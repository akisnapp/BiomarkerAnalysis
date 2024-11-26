import os
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import torchvision.utils as vutils
import cv2
import torchio as tio
from PIL import Image, ImageEnhance

# Constants for default configuration
DATA_PATH = '/storage/ice1/shared/d-pace_community/makerspace-datasets/MEDICAL/OLIVES/OLIVES/'
LABEL_DATA_PATH = os.path.join(DATA_PATH, 'Biomarker_Clinical_Data_Images_Updated.csv')
BIOMARKER_COLUMNS = [
    'Atrophy / thinning of retinal layers', 'Disruption of EZ', 'DRIL', 'IR hemorrhages',
    'IR HRF', 'Partially attached vitreous face', 'Fully attached vitreous face',
    'Preretinal tissue/hemorrhage', 'DRT/ME', 'Fluid (IRF)', 'Fluid (SRF)',
    'Disruption of RPE', 'PED (serous)', 'SHRM'
]

def preprocess_biomarkers(data_frame, biomarker_cols):
    for col in biomarker_cols:
        if col in data_frame:
            data_frame[col] = data_frame[col].fillna(data_frame[col].mean())  # Impute missing values
            data_frame[col] = (data_frame[col] - data_frame[col].mean()) / data_frame[col].std()  # Normalize
    return data_frame

# Custom noise addition transformations
class AddSaltPepperNoise:
    def __init__(self, amount=0.01):
        self.amount = amount

    def __call__(self, img):
        return Image.fromarray(self._add_salt_pepper(np.array(img), self.amount).astype(np.uint8))

    @staticmethod
    def _add_salt_pepper(image, amount):
        s_vs_p = 0.5
        noisy_img = np.copy(image)

        # Add salt
        num_salt = int(np.ceil(amount * image.size * s_vs_p))
        salt_coords = [np.random.randint(0, dim - 1, num_salt) for dim in image.shape[:2]]
        noisy_img[salt_coords[0], salt_coords[1]] = 255

        # Add pepper
        num_pepper = int(np.ceil(amount * image.size * (1. - s_vs_p)))
        pepper_coords = [np.random.randint(0, dim - 1, num_pepper) for dim in image.shape[:2]]
        noisy_img[pepper_coords[0], pepper_coords[1]] = 0

        return noisy_img


class AddGaussianNoise:
    def __init__(self, mean=0, variance=0.01):
        self.mean = mean
        self.variance = variance

    def __call__(self, img):
        return Image.fromarray((self._add_gaussian_noise(np.array(img), self.mean, self.variance) * 255).astype(np.uint8))

    @staticmethod
    def _add_gaussian_noise(image, mean=0, variance=0.01):
        noise = np.random.normal(mean, np.sqrt(variance), image.shape)
        noisy_img = image + noise
        return np.clip(noisy_img, 0, 1)


# Define more transformations
class AddRandomBrightnessContrast:
    def __call__(self, img):
        img = np.array(img).astype(np.float32) / 255.0
        alpha = 1.0 + (np.random.rand() - 0.5) * 0.4  # Brightness range: 0.8 to 1.2
        beta = (np.random.rand() - 0.5) * 0.2  # Contrast range: -0.1 to 0.1
        img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta * 255)
        return Image.fromarray(img)


class AddRandomRotation:
    def __init__(self, max_angle=30):
        self.max_angle = max_angle

    def __call__(self, img):
        angle = np.random.uniform(-self.max_angle, self.max_angle)
        return img.rotate(angle)

class ElasticDeformation: #TODO: This is not currently used, look into if this actually offers improvement
    def __init__(self, num_control_points=4, max_displacement=5):
        self.transform = tio.ElasticDeformation(
            num_control_points=num_control_points, max_displacement=max_displacement
        )

    def __call__(self, img):
        return self.transform(img)
    
class RandomSharpness: #TODO: Not currently used, this might help with picking up on patterns, look into using it
    def __init__(self, factor_range=(0.5, 2.0)):
        self.factor_range = factor_range

    def __call__(self, img):
        factor = np.random.uniform(*self.factor_range)
        enhancer = ImageEnhance.Sharpness(img)
        return enhancer.enhance(factor)
    
class RandomErasing: #TODO: Not currently used, I think this might help with having AI pick up on broader patterns/trends if that becomes an issue later
    def __init__(self, p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3)):
        self.transform = transforms.RandomErasing(p=p, scale=scale, ratio=ratio)

    def __call__(self, img):
        return self.transform(img)

# Preprocessing for clinical and biomarker data
def preprocess_biomarkers(data_frame, biomarker_cols):
    for col in biomarker_cols:
        if col in data_frame:
            data_frame[col] = data_frame[col].fillna(data_frame[col].mean())  # Impute missing values
            data_frame[col] = (data_frame[col] - data_frame[col].mean()) / data_frame[col].std()  # Normalize
    return data_frame


class MultiTransformOLIVESDataset(Dataset):
    def __init__(self, csv_path, image_root, biomarker_cols, transforms_list, num_transforms=1, image_mode="RGB"):
        self.data = pd.read_csv(csv_path)
        self.image_root = image_root
        self.biomarker_cols = biomarker_cols
        self.transforms_list = transforms_list
        self.num_transforms = num_transforms
        self.image_mode = image_mode  # "RGB" for 3 channels, "L" for grayscale
        self.data = preprocess_biomarkers(self.data, biomarker_cols)

    def __len__(self):
        return len(self.data) * self.num_transforms

    def __getitem__(self, idx):
        data_idx = idx // self.num_transforms
        transform_idx = idx % self.num_transforms

        image_path = os.path.join(self.image_root, self.data.iloc[data_idx]['Path (Trial/Arm/Folder/Visit/Eye/Image Name)'].strip())
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        image = Image.open(image_path).convert(self.image_mode)
        transform = self.transforms_list[transform_idx]
        if transform:
            image = transform(image)

        biomarker_data = self.data.iloc[data_idx][self.biomarker_cols].values.astype(np.float32)
        label = self.data.iloc[data_idx]['BCVA']
        biomarker_tensor = torch.tensor(biomarker_data, dtype=torch.float32)
        label_tensor = torch.tensor(label, dtype=torch.float32)

        return image, biomarker_tensor, label_tensor


def get_data_loaders(batch_size=16, num_workers=1, pin_memory=True):
    """
    Returns train and validation DataLoaders with default settings.

    Args:
        batch_size (int): Batch size for the DataLoaders. Default is 16.
        num_workers (int): Number of workers for data loading. Default is 1.
        pin_memory (bool): Whether to use pinned memory. Default is True.

    Returns:
        tuple: (train_loader, val_loader)
    """

    # Define multiple transformation pipelines
    transform_0 = transforms.Compose([ #TODO: Not used yet
        transforms.Resize((224, 224)),
        ElasticDeformation(num_control_points=5, max_displacement=5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    transform_sharpness = transforms.Compose([ #TODO: Not used yet
        transforms.Resize((224, 224)),
        RandomSharpness(factor_range=(0.5, 2.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    transform_random_erase = transforms.Compose([ #TODO: Not used yet
        transforms.Resize((224, 224)),
        RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    transform_1 = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    transform_2 = transforms.Compose([
        transforms.Resize((224, 224)),
        AddSaltPepperNoise(amount=0.01),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    transform_3 = transforms.Compose([
        transforms.Resize((224, 224)),
        AddGaussianNoise(mean=0, variance=0.01),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    transform_4 = transforms.Compose([
        transforms.Resize((224, 224)),
        AddRandomBrightnessContrast(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    transform_5 = transforms.Compose([
        transforms.Resize((224, 224)),
        AddRandomRotation(max_angle=30),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    transforms_list = [transform_1, transform_2, transform_3, transform_4, transform_5]
    dataset = MultiTransformOLIVESDataset(
        csv_path=LABEL_DATA_PATH,
        image_root=DATA_PATH,
        biomarker_cols=BIOMARKER_COLUMNS,
        transforms_list=transforms_list,
        num_transforms=len(transforms_list)
    )

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)

    return train_loader, val_loader
