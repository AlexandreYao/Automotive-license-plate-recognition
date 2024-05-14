import os
import cv2
import glob
import torch
import numpy as np
from torch.utils.data import Dataset



class CustomDataset(Dataset):
    def __init__(self, image_folders, max_objects=2, image_size=416):
        """
        Initialise un CustomDataset.

        Args :
            image_folders (list): Liste des chemins vers les dossiers contenant les images et les annotations.
            max_objects (int, optional): Nombre maximum d'objets par image. Par défaut, 2.
            image_size (int, optional): Taille des images. Par défaut, 416.
        """
        self.image_folders = image_folders
        self.image_size = image_size
        self.max_objects = max_objects
        self.data = self.prepare_data()

    def __len__(self):
        """
        Renvoie la taille du jeu de données.
        """
        return len(self.data)

    def __getitem__(self, idx):
        """
        Renvoie un échantillon du jeu de données.
        Args:
            idx (int): Indice de l'échantillon à récupérer.
        Returns:
            dict: Un dictionnaire contenant l'image à utiliser en pytorch, l'image original, ses annotations et le chemin vers l'image.
        """
        # Récupère les chemins de l'image et des annotations pour l'index donné
        image_path, annotations_path = self.data[idx]
        # Charge l'image à partir du chemin
        image = cv2.imread(image_path)
        # Récupère les dimensions de l'image
        height, width, _ = image.shape
        # Calcule la différence de dimension entre la hauteur et la largeur de l'image
        dim_diff = np.abs(height - width)
        # Calcule les valeurs de padding en fonction de la différence de dimension
        pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
        padding_value = 128
        padding = (
            ((pad1, pad2), (0, 0), (0, 0))
            if height <= width
            else ((0, 0), (pad1, pad2), (0, 0))
        )
        # Applique le padding à l'image
        padded_image = np.pad(image, padding, "constant", constant_values=padding_value)
        # Récupère les nouvelles dimensions de l'image après le padding
        padded_height, padded_width, _ = padded_image.shape
        # Redimensionne l'image à la taille spécifiée
        padded_image = cv2.resize(padded_image, (self.image_size, self.image_size))
        # Transpose les dimensions de l'image et convertit en tenseur PyTorch
        input_img = padded_image[:, :, ::-1].transpose((2, 0, 1)).copy()
        input_img = torch.from_numpy(input_img).float().div(255.0)
        # Charge les annotations de l'image
        filled_labels = self.load_annotations(
            annotation_file=annotations_path,
            height=height,
            width=width,
            padding=padding,
            padded_height=padded_height,
            padded_width=padded_width,
        )
        # Crée un dictionnaire contenant l'échantillon d'image et ses informations associées
        sample = {
            "input_img": input_img,
            "original_img": padded_image,
            "label": filled_labels,
            "path": image_path,
        }
        return sample

    def load_annotations(
        self, annotation_file, height, width, padding, padded_height, padded_width
    ):
        """
        Charge les annotations à partir d'un fichier.
        Args:
            annotation_file (str): Chemin vers le fichier d'annotations.
            height (int): Hauteur de l'image d'origine.
            width (int): Largeur de l'image d'origine.
            padding (tuple): Valeurs de padding utilisées pour l'image.
            padded_height (int): Hauteur de l'image après padding.
            padded_width (int): Largeur de l'image après padding.
        Returns:
            torch.Tensor: Annotations remplies.
        """
        # Charge les annotations depuis le fichier et les reshape pour obtenir un tableau 2D
        labels = np.loadtxt(annotation_file).reshape(-1, 5)
        # Calcule les coordonnées des coins des boîtes englobantes
        x1 = width * (labels[:, 1] - labels[:, 3] / 2)
        y1 = height * (labels[:, 2] - labels[:, 4] / 2)
        x2 = width * (labels[:, 1] + labels[:, 3] / 2)
        y2 = height * (labels[:, 2] + labels[:, 4] / 2)
        # Ajoute le padding aux coordonnées x et y des coins
        x1 += padding[1][0]
        y1 += padding[0][0]
        x2 += padding[1][0]
        y2 += padding[0][0]
        # Normalise les coordonnées des centres et les dimensions par rapport aux dimensions de l'image après padding
        labels[:, 1] = ((x1 + x2) / 2) / padded_width
        labels[:, 2] = ((y1 + y2) / 2) / padded_height
        labels[:, 3] *= width / padded_width
        labels[:, 4] *= height / padded_height
        # Initialise un tableau de zéros pour les annotations et les remplit avec les annotations chargées
        filled_labels = np.zeros((self.max_objects, 5))
        filled_labels[: len(labels)] = labels[: self.max_objects]
        # Convertit le tableau numpy en tensor PyTorch et le retourne
        filled_labels = torch.from_numpy(filled_labels)
        return filled_labels

    def prepare_data(self):
        """
        Prépare les données en associant chaque image à ses annotations.
        Returns:
            list: Liste des paires (chemin de l'image, chemin de l'annotation) pour chaque image valide.
        """
        # Initialise une liste pour stocker les paires image-annotation
        data = []
        # Parcourt chaque dossier contenant des images
        for image_folder in self.image_folders:
            annotation_folder = image_folder
            # Parcourt chaque fichier d'annotation dans le dossier
            for annotations_path in glob.glob(os.path.join(annotation_folder, "*.txt")):
                # Construit le chemin de l'image correspondant au fichier d'annotation
                image_name = (
                    os.path.splitext(os.path.basename(annotations_path))[0] + ".jpg"
                )
                image_path = os.path.join(image_folder, image_name)
                # Vérifie si l'image correspondante existe
                if os.path.exists(image_path):
                    # Ajoute la paire image-annotation à la liste des données
                    data.append((image_path, annotations_path))
        # Renvoie la liste complète des paires image-annotation
        return data