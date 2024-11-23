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


# Preprocessing for clinical and biomarker data
def preprocess_biomarkers(data_frame, biomarker_cols):
    for col in biomarker_cols:
        if col in data_frame:
            data_frame[col] = data_frame[col].fillna(data_frame[col].mean())  # Impute missing values
            data_frame[col] = (data_frame[col] - data_frame[col].mean()) / data_frame[col].std()  # Normalize
    return data_frame


# Dataset class supporting multiple transformations per image
class MultiTransformOLIVESDataset(Dataset):
    def __init__(self, csv_path, image_root, biomarker_cols, transforms_list, num_transforms=5):
        self.data = pd.read_csv(csv_path)
        self.image_root = image_root
        self.biomarker_cols = biomarker_cols
        self.transforms_list = transforms_list
        self.num_transforms = num_transforms

        # Preprocess biomarkers
        self.data = preprocess_biomarkers(self.data, biomarker_cols)

    def __len__(self):
        return len(self.data) * self.num_transforms

    def __getitem__(self, idx):
        # Map to original data index
        data_idx = idx // self.num_transforms
        transform_idx = idx % self.num_transforms

        # Get image path
        image_path = self.image_root + self.data.iloc[data_idx]['Path (Trial/Arm/Folder/Visit/Eye/Image Name)'].strip()
        

        if not os.path.exists(image_path):
            print(f"Image not found: {image_path}")
            return None, None, None

        # Load and preprocess image
        image = Image.open(image_path).convert("RGB")
        transform = self.transforms_list[transform_idx]
        if transform:
            image = transform(image)

        # Extract clinical labels and biomarkers
        biomarker_data = self.data.iloc[data_idx][self.biomarker_cols].values.astype(np.float32)
        label = self.data.iloc[data_idx]['BCVA']  # Example: BCVA as the main label

        # Convert to tensors
        biomarker_tensor = torch.tensor(biomarker_data, dtype=torch.float32)
        label_tensor = torch.tensor(label, dtype=torch.float32)
        return image, biomarker_tensor, label_tensor


# Visualization
def visualize_batch(dataloader):
    images, biomarkers, labels = next(iter(dataloader))
    grid = vutils.make_grid(images, normalize=True)
    plt.figure(figsize=(12, 12))
    plt.imshow(grid.permute(1, 2, 0))
    plt.title(f"Sample Batch - Labels: {labels.tolist()}")
    plt.axis("off")
    plt.show()


def main():
    # Paths
    DATA_PATH = '/storage/ice1/shared/d-pace_community/makerspace-datasets/MEDICAL/OLIVES/OLIVES'
    CURRENT_DIR = os.getcwd()
    LABEL_DATA_PATH = os.path.join(CURRENT_DIR, '..', 'OLIVES_Dataset_Labels', 'full_labels', 'Biomarker_Clinical_Data_Images.csv')

    # Define biomarker columns
    biomarker_columns = [
        'Atrophy / thinning of retinal layers', 'Disruption of EZ', 'DRIL', 'IR hemorrhages',
        'IR HRF', 'Partially attached vitreous face', 'Fully attached vitreous face',
        'Preretinal tissue/hemorrhage', 'DRT/ME', 'Fluid (IRF)', 'Fluid (SRF)',
        'Disruption of RPE', 'PED (serous)', 'SHRM'
    ]

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

    # Dataset and DataLoader
    dataset = MultiTransformOLIVESDataset(
        csv_path=LABEL_DATA_PATH,
        image_root=DATA_PATH,
        biomarker_cols=biomarker_columns,
        transforms_list=transforms_list,
        num_transforms=len(transforms_list)
    )
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=2, pin_memory=True)


if __name__ == "__main__":
    main()
