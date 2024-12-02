import os
import pandas as pd
import numpy as np
from PIL import Image, ImageEnhance
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from scipy.ndimage import gaussian_filter, map_coordinates

# Constants
# !pip install zenodo-get
DATA_PATH ='/storage/ice1/shared/d-pace_community/makerspace-datasets/MEDICAL/OLIVES/OLIVES'
LABEL_DATA_PATH = 'OLIVES_Dataset_Labels/full_labels/Biomarker_Clinical_Data_Images.csv'

# Define Biomarker Columns (Binary Labels)
BIOMARKER_COLUMNS_BINARY = [
    'Atrophy / thinning of retinal layers', 'Disruption of EZ', 'DRIL', 'IR hemorrhages',
    'IR HRF', 'Partially attached vitreous face', 'Fully attached vitreous face',
    'Preretinal tissue/hemorrhage', 'DRT/ME', 'Fluid (IRF)', 'Fluid (SRF)',
    'Disruption of RPE', 'PED (serous)', 'SHRM'
]

# Define Continuous Labels (e.g., Clinical Measurements)
CONTINUOUS_COLUMNS = [
    'BCVA',  # Best Corrected Visual Acuity
    'CST'    # Central Subfield Thickness
]
# Preprocessing Function
def preprocess_data(data_frame, binary_cols=BIOMARKER_COLUMNS_BINARY, continuous_cols=CONTINUOUS_COLUMNS):
    # Preprocess binary columns
    for col in binary_cols:
        if col in data_frame:
            data_frame[col] = data_frame[col].fillna(0)
            data_frame[col] = (data_frame[col] > 0).astype(int)  # Ensure binary labels

    # Preprocess continuous columns
    for col in continuous_cols:
        if col in data_frame:
            data_frame[col] = data_frame[col].fillna(data_frame[col].mean())  # Impute missing values
            data_frame[col] = (data_frame[col] - data_frame[col].mean()) / data_frame[col].std()  # Normalize

    return data_frame

# Custom Transformations
class AddSaltPepperNoise:
    def __init__(self, amount=0.01):
        self.amount = amount

    def __call__(self, img):
        img_np = np.array(img)
        s_vs_p = 0.5
        noisy_img = np.copy(img_np)

        # Add salt
        num_salt = int(np.ceil(self.amount * img_np.size * s_vs_p))
        coords = [np.random.randint(0, i - 1, num_salt) for i in img_np.shape[:2]]
        noisy_img[coords[0], coords[1]] = 255

        # Add pepper
        num_pepper = int(np.ceil(self.amount * img_np.size * (1. - s_vs_p)))
        coords = [np.random.randint(0, i - 1, num_pepper) for i in img_np.shape[:2]]
        noisy_img[coords[0], coords[1]] = 0

        return Image.fromarray(noisy_img.astype(np.uint8))

class AddGaussianNoise:
    def __init__(self, mean=0, variance=0.01):
        self.mean = mean
        self.variance = variance

    def __call__(self, img):
        img_np = np.array(img).astype(np.float32) / 255.0
        noise = np.random.normal(self.mean, np.sqrt(self.variance), img_np.shape)
        noisy_img = img_np + noise
        noisy_img = np.clip(noisy_img, 0, 1) * 255
        return Image.fromarray(noisy_img.astype(np.uint8).astype(np.uint8))

class AddRandomBrightnessContrast:
    def __call__(self, img):
        img = img.convert("RGB")
        enhancer_brightness = ImageEnhance.Brightness(img)
        enhancer_contrast = ImageEnhance.Contrast(img)
        brightness_factor = 1.0 + (np.random.rand() - 0.5) * 0.4  # 0.8 to 1.2
        contrast_factor = 1.0 + (np.random.rand() - 0.5) * 0.2    # 0.9 to 1.1
        img = enhancer_brightness.enhance(brightness_factor)
        img = enhancer_contrast.enhance(contrast_factor)
        return img

class AddRandomRotation:
    def __init__(self, max_angle=30):
        self.max_angle = max_angle

    def __call__(self, img):
        angle = np.random.uniform(-self.max_angle, self.max_angle)
        return img.rotate(angle)

class RandomSharpness:
    def __init__(self, factor_range=(0.5, 2.0)):
        self.factor_range = factor_range

    def __call__(self, img):
        factor = np.random.uniform(*self.factor_range)
        enhancer = ImageEnhance.Sharpness(img)
        return enhancer.enhance(factor)

class RandomErasing:
    def __init__(self, p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3)):
        self.transform = transforms.RandomErasing(p=p, scale=scale, ratio=ratio)

    def __call__(self, img):
        return self.transform(img)

class ElasticDeformation:
    def __init__(self, alpha=1.0, sigma=10.0, p=0.5):
        self.alpha = alpha
        self.sigma = sigma
        self.p = p

    def __call__(self, img):
        if np.random.rand() > self.p:
            return img

        img_np = np.array(img)
        shape = img_np.shape

        random_state = np.random.RandomState(None)
        dx = gaussian_filter((random_state.rand(*shape[:2]) * 2 - 1), self.sigma, mode="constant", cval=0) * self.alpha
        dy = gaussian_filter((random_state.rand(*shape[:2]) * 2 - 1), self.sigma, mode="constant", cval=0) * self.alpha

        x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
        distored_x = np.clip(x + dx, 0, shape[1] - 1)
        distored_y = np.clip(y + dy, 0, shape[0] - 1)

        distorted_img = np.zeros_like(img_np)
        for i in range(shape[2]):  # For RGB channels
            distorted_img[..., i] = map_coordinates(img_np[..., i], [distored_y, distored_x], order=1, mode='reflect')

        return Image.fromarray(distorted_img)

# Dataset Class
class MultiLabelOLIVESDataset(Dataset):
    def __init__(self, csv_path, image_root, binary_cols, continuous_cols, transform=None, image_mode="RGB"):
        self.data = pd.read_csv(csv_path)
        self.image_root = image_root
        self.binary_cols = binary_cols
        self.continuous_cols = continuous_cols
        self.transform = transform
        self.image_mode = image_mode
        self.data = preprocess_data(self.data, binary_cols, continuous_cols)
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        image_path = self.image_root + self.data.iloc[idx]['Path (Trial/Arm/Folder/Visit/Eye/Image Name)']
        image_path = image_path.replace(" ", "_")
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        image = Image.open(image_path).convert(self.image_mode)
        if self.transform:
            image = self.transform(image)

        # Get binary labels
        binary_labels = row[self.binary_cols].values.astype(np.float32)
        binary_labels = torch.tensor(binary_labels, dtype=torch.float32)

        # Get continuous labels
        continuous_labels = row[self.continuous_cols].values.astype(np.float32)
        continuous_labels = torch.tensor(continuous_labels, dtype=torch.float32)

        return image, binary_labels, continuous_labels

# Prepare DataLoader
def prepare_dataset(csv_path, image_root, batch_size, binary_cols, continuous_cols, image_mode="RGB", num_workers=1, pin_memory=True):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        # Add your custom transformations here
        AddSaltPepperNoise(amount=0.01),
        AddGaussianNoise(mean=0, variance=0.01),
        AddRandomBrightnessContrast(),
        AddRandomRotation(max_angle=30),
        RandomSharpness(factor_range=(0.5, 2.0)),
        ElasticDeformation(alpha=1.0, sigma=10.0, p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    dataset = MultiLabelOLIVESDataset(csv_path, image_root, binary_cols, continuous_cols, transform, image_mode)
    total_size = len(dataset)
    train_size = int(0.8 * total_size)
    val_size = int(0.1 * total_size)
    test_size = total_size - train_size - val_size

    # Split dataset
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)

    return train_loader, val_loader, test_loader
