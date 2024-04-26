import os
import pandas as pd
import urllib.request
from bs4 import BeautifulSoup
import errno
import random


class AutomotiveLicenceScraper:
    def __init__(self):
        self.base_url = "https://platesmania.com/fr/gallery"
        self.max_pages_to_scrap = 10
        self.num_pages_scraped = 0
        self.directory_img_plate = os.path.join("../data", "images", "plates")
        self.directory_img_global = os.path.join("../data", "images", "cars")
        self.csv_filepath = os.path.join("../data", "data", "data.csv")
        self.create_directory(os.path.join("../data", "data"))
        self.create_directory(self.directory_img_plate)
        self.create_directory(self.directory_img_global)
        if not os.path.isfile(self.csv_filepath):
            columns = [
                "date",
                "voiture_modele",
                "img_global_name",
                "img_plaque_name",
                "plate_number",
            ]
            self.global_data = pd.DataFrame(columns=columns)
            self.global_data.to_csv(self.csv_filepath, encoding="utf-8", index=False)
        else:
            # Charger les données depuis le fichier CSV
            self.global_data = pd.read_csv(self.csv_filepath)

    def create_directory(self, directory):
        if not os.path.exists(directory):
            try:
                os.makedirs(directory)
            except OSError as e:
                if e.errno != errno.EEXIST:
                    raise

    def parse(self):
        paginations = []
        url = self.base_url
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"
        }
        while self.num_pages_scraped < self.max_pages_to_scrap:
            pagination = random.randint(1, 100)
            while pagination in paginations:
                pagination = random.randint(1, 100)
            url = f"{self.base_url}-{self.num_pages_scraped}"
            req = urllib.request.Request(url, headers=headers)
            try:
                response = urllib.request.urlopen(req)
            except urllib.error.HTTPError as e:
                print("Erreur HTTP")
                print(e)
            soup = BeautifulSoup(response, "html.parser")
            img_containers = soup.find_all("div", class_="panel panel-grey")
            for img_container in img_containers:
                panel_body = img_container.find("div", class_="panel-body")
                modele = panel_body.find("h4").find("a").get_text(strip=True)
                date = panel_body.find("small", class_="pull-right").get_text(
                    strip=True
                )
                images = img_container.find_all("div", class_="row")
                sub_container_img_global = images[1]
                sub_container_img_plate = images[2]
                url_img_global = sub_container_img_global.find("a").find("img")["src"]
                url_img_plate = sub_container_img_plate.find("a").find("img")["src"]
                plate_number = sub_container_img_plate.find("a").find("img")["alt"]
                img_global_name = os.path.basename(url_img_global)
                img_plate_name = os.path.basename(url_img_plate)
                dest_folder_img_plate = os.path.join(
                    self.directory_img_plate, img_plate_name
                )
                dest_folder_img_global = os.path.join(
                    self.directory_img_global, img_global_name
                )
                row = {
                    "date": date,
                    "voiture_modele": modele,
                    "img_global_name": img_global_name,
                    "img_plaque_name": img_plate_name,
                    "plate_number": plate_number,
                }
                is_unique = self.global_data[
                    (self.global_data["date"] == date)
                    & (self.global_data["voiture_modele"] == modele)
                    & (self.global_data["img_global_name"] == img_global_name)
                    & (self.global_data["img_plaque_name"] == img_plate_name)
                    & (self.global_data["plate_number"] == plate_number)
                ]
                if is_unique.empty:
                    request = urllib.request.Request(url_img_global, headers=headers)
                    with urllib.request.urlopen(request) as response, open(
                        dest_folder_img_global, "wb"
                    ) as out_file:
                        out_file.write(response.read())
                    request = urllib.request.Request(url_img_plate, headers=headers)
                    with urllib.request.urlopen(request) as response, open(
                        dest_folder_img_plate, "wb"
                    ) as out_file:
                        out_file.write(response.read())
                    self.global_data = pd.concat(
                        [self.global_data, pd.DataFrame([row])], ignore_index=True
                    )
            self.num_pages_scraped += 1
        self.global_data.to_csv(self.csv_filepath, encoding="utf-8", index=False)


# Utilisation de la classe AutomotiveLicenceScraper
scraper = AutomotiveLicenceScraper()
scraper.parse()