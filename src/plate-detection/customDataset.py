import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self, image_folder, annotation_folder, image_size=416):
        self.image_folder = image_folder
        self.annotation_folder = annotation_folder
        self.image_size = image_size
        self.data = self.preprocess_data()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_path, annotations = self.data[idx]
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (self.image_size, self.image_size))
        image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1) / 255.0
        annotations = torch.tensor(annotations, dtype=torch.float32)
        return image, annotations

    def load_annotations(self, annotation_file):
        with open(annotation_file, "r") as f:
            lines = f.readlines()
        annotations = []
        for line in lines:
            parts = line.strip().split()
            annotations.append([float(parts[i]) for i in range(1, len(parts))])
        return np.array(annotations)

    def preprocess_data(self):
        data = []
        for image_name in os.listdir(self.image_folder):
            if image_name.endswith(".jpg"):
                image_path = os.path.join(self.image_folder, image_name)
                annotation_name = os.path.splitext(image_name)[0] + ".txt"
                annotation_path = os.path.join(self.annotation_folder, annotation_name)
                if os.path.exists(annotation_path):
                    annotations = self.load_annotations(annotation_path)
                    data.append((image_path, annotations))
        return data