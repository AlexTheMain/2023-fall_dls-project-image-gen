# Импорт необходимых модулей и библиотек
import glob
import os
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
import random

def denorm(tensors):
    # Денормализация тензоров изображений
    return tensors * 0.5 + 0.5

class ImageDataset(Dataset):
    def __init__(self, path, transform=None, unaligned=False, mode="train"):
        # Инициализация датасета
        self.transform = transforms.Compose(transform)
        self.unaligned = unaligned

        # Получение путей к изображениям
        self.files_A = sorted(glob.glob(os.path.join(path, f"{mode}A") + "/*.*"))
        self.files_B = sorted(glob.glob(os.path.join(path, f"{mode}B") + "/*.*"))

    def __getitem__(self, index):
        # Получение изображений из датасета
        if self.unaligned:
            image_B = Image.open(self.files_B[random.randint(0, len(self.files_B) - 1)])
            image_A = Image.open(self.files_A[random.randint(0, len(self.files_A) - 1)])
        else:
            image_B = Image.open(self.files_B[index % len(self.files_B)])
            image_A = Image.open(self.files_A[index % len(self.files_A)])

        # Применение трансформации к изображениям
        item_A = self.transform(image_A)
        item_B = self.transform(image_B)
        return {"A": item_A, "B": item_B}

    def __len__(self):
        # Возвращает размер датасета
        return max(len(self.files_A), len(self.files_B))
