import os
import cv2
import torch
import unicodedata
import pandas as pd
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self, image_folder, data_path, image_size=416):
        self.image_folder = image_folder
        self.data_path = data_path
        self.image_size = image_size
        self.data = self._preprocess_data()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_path, label = self.data[idx]
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (self.image_size, self.image_size))
        image = torch.tensor(image, dtype=torch.float32) / 255.0
        image = image.permute(2, 0, 1)
        return image, torch.tensor(label, dtype=torch.float32)

    def _preprocess_data(self):
        data = []
        df = self._prepare_data()
        for _, row in df.iterrows():
            image_path = os.path.join(self.image_folder, row["car_img_name"])
            if os.path.exists(image_path):
                labels = row[1:].values.tolist()
                data.append((image_path, labels))
        return data

    def _prepare_data(self):
        df = pd.read_csv(self.data_path)
        df["car_model"] = (
            df["car_model"]
            .fillna("")
            .str.lower()
            .str.replace("-", " ")
            .apply(
                lambda x: unicodedata.normalize("NFKD", x)
                .encode("ASCII", "ignore")
                .decode("utf-8")
            )
        )
        df["car_model"] = df["car_model"].apply(lambda cm: self._parse_car_model(cm))
        models = ["citroen", "peugeot", "renault"]
        df = df[["car_img_name", "car_model"]]
        df = df[df["car_model"].isin(models)]
        encoded_columns = pd.get_dummies(df["car_model"]).astype(int)
        df = pd.concat([df, encoded_columns], axis="columns").drop(columns=["car_model"])
        return df

    def _parse_car_model(self, car_model):
        models_known = {
            "abarth": "abarth",
            "alfa romeo": "alfa romeo",
            "alpine": "alpine",
            "aston martin": "aston martin",
            "audi": "audi",
            "autobianchi": "autobianchi",
            "bentley": "bentley",
            "bmw": "bmw",
            "chevrolet": "chevrolet",
            "chrysler": "chrysler",
            "citron": "citroen",
            "cupra": "cupra",
            "dacia": "dacia",
            "dodge": "dodge",
            "ferrari": "ferrari",
            "fiat": "fiat",
            "ford": "ford",
            "honda": "honda",
            "hyundai": "hyundai",
            "ineos": "ineos",
            "irizar": "irizar",
            "isuzu": "isuzu",
            "iveco": "iveco",
            "jaguar": "jaguar",
            "jeep": "jeep",
            "kia": "kia",
            "lamborghin": "lamborghin",
            "land rover": "range rover",
            "lexus": "lexus",
            "maserat": "maserat",
            "mazda": "mazda",
            "mclaren": "mclaren",
            "mercedes": "mercedes",
            "mercury": "mercury",
            "microcar": "microcar",
            "mini": "mini",
            "mitsubishi": "mitsubishi",
            "neoplan": "neoplan",
            "nissan": "nissan",
            "opel": "opel",
            "peugeot": "peugeot",
            "piaggio": "piaggio",
            "porsche": "porsche",
            "range rover": "range rover",
            "renault": "renault",
            "rolls royce": "rolls royce",
            "saab": "saab",
            "scania": "scania",
            "seat": "seat",
            "setra": "setra",
            "simca": "simca",
            "skoda": "skoda",
            "subaru": "subaru",
            "suzuki": "suzuki",
            "tesla": "tesla",
            "toyota": "toyota",
            "triumph": "triumph",
            "volkswagen": "volkswagen",
            "volvo": "volvo",
        }
        for k, v in models_known.items():
            if k in car_model:
                return v
        return 0
