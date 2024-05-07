import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self, image_folders, image_size=416):
        self.image_folders = image_folders
        self.annotation_folders = image_folders
        self.image_size = image_size
        self.data = self.preprocess_data()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_path, annotations = self.data[idx]
        # Charger l'image en niveau de gris
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        # Récupérer les dimensions originales de l'image
        original_image_size = image.shape[:2]
        # Redimensionner l'image
        image = cv2.resize(image, (self.image_size, self.image_size))
        # Convertir l'image en tenseur et normaliser
        image = torch.tensor(image, dtype=torch.float32).unsqueeze(0) / 255.0
        # Convertir les annotations au format YOLO en coordonnées de coins de rectangle
        annotations = torch.tensor(self.convert_yolo_to_corners(annotations, original_image_size), dtype=torch.float32)
        return image, annotations

    def load_annotations(self, annotation_file):
        annotations = []
        try:
            with open(annotation_file, "r") as f:
                # Plusieurs lignes dans le fichier d'annotations
                for line in f.readlines() :
                    annotations.append([float(v) for v in line.strip().split()[1:]])
        except Exception:
            print(f"[WARNING] loading annotation {annotation_file} failed !")
            pass
        return np.array(annotations)

    def preprocess_data(self):
        data = []
        for image_folder, annotation_folder in zip(self.image_folders, self.annotation_folders):
            for annotation_name in [
                f for f in os.listdir(annotation_folder) if f.endswith(".txt")
            ]:
                image_name = os.path.splitext(annotation_name)[0] + ".jpg"
                image_path = os.path.join(image_folder, image_name)
                annotation_path = os.path.join(annotation_folder, annotation_name)
                if os.path.exists(image_path) and os.path.exists(annotation_path):
                    annotations = self.load_annotations(annotation_path)
                    if len(annotations) > 0:
                        data.append((image_path, annotations))
        return data

    def convert_yolo_to_corners(self, annotations, original_image_size):
        r = []
        for a in annotations:
            # Taille de l'image originale
            original_width, original_height = original_image_size
            # Taille de l'image redimensionnée
            resized_width, resized_height = self.image_size, self.image_size
            # Ajuster les coordonnées YOLO en fonction de la taille réelle de l'image
            x_center, y_center, width, height = a
            x_min = (x_center - width / 2) * original_width
            y_min = (y_center - height / 2) * original_height
            x_max = (x_center + width / 2) * original_width
            y_max = (y_center + height / 2) * original_height
            # Redimensionner les coordonnées en fonction de la taille de l'image redimensionnée
            x_min_resized = x_min * resized_width / original_width
            y_min_resized = y_min * resized_height / original_height
            x_max_resized = x_max * resized_width / original_width
            y_max_resized = y_max * resized_height / original_height
            # Retourner les coordonnées de la bounding box redimensionnée
            r.append([x_min_resized, y_min_resized, x_max_resized, y_max_resized])
        return r