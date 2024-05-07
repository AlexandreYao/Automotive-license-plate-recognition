import pandas as pd
import requests
import os
import time
from random import randint

country = "pl"
df = pd.read_csv(rf"..\data\data\data_{country}.csv")
save_dir = rf"..\data\images\cars\{country}"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"
}
def download_image(url, filename):
    try:
        time.sleep(randint(1, 10))
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            with open(filename, "wb") as f:
                f.write(response.content)
                print(f"Image téléchargée : {filename}")
        else:
            print(f"Échec du téléchargement de l'image : {filename}")
    except Exception as e:
        print(f"Erreur lors du téléchargement de l'image : {filename}")
        print(e)


for index, row in df.iterrows():
    car_img_url = row["car_img_url"]
    car_img_name = row["car_img_name"]
    save_path = os.path.join(save_dir, car_img_name)
    if os.path.exists(save_path):
        print(f"L'image {car_img_name} a déjà été téléchargée.")
    else:
        download_image(car_img_url, save_path)