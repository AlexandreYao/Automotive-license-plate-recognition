import os
import pandas as pd
import urllib.request
from bs4 import BeautifulSoup
import time
from random import randint
import errno

class LicensePlateScraper:
    """A class to scrape automotive license plate data."""

    def __init__(self, country="fr"):
        """Initialize the scraper."""
        self.country = country
        self.base_url = "https://platesmania.com/fr/gallery"
        self.max_pages_to_scrape = 105
        self.num_pages_scraped = 0
        self.plate_img_directory = os.path.join("../data", "images", "plates")
        self.car_img_directory = os.path.join("../data", "images", "cars")
        self.csv_filepath = os.path.join("../data", "data", f"data_{self.country}.csv")
        self.create_directory(os.path.join("../data", "data"))
        self.create_directory(self.plate_img_directory)
        self.create_directory(self.car_img_directory)
        if not os.path.isfile(self.csv_filepath):
            columns = [
                "date",
                "car_model",
                "car_img_name",
                "car_img_url",
                "plate_img_name",
                "plate_img_url",
                "plate_number",
            ]
            self.license_plate_data = pd.DataFrame(columns=columns)
            self.license_plate_data.to_csv(
                self.csv_filepath, encoding="utf-8", index=False
            )
        else:
            self.license_plate_data = pd.read_csv(self.csv_filepath)

    def create_directory(self, directory):
        """Create a directory if it doesn't exist."""
        if not os.path.exists(directory):
            try:
                os.makedirs(directory)
            except OSError as e:
                if e.errno != errno.EEXIST:
                    raise

    def parse(self):
        """Parse the website to extract data."""
        pages_file_name = rf"..\data\data\pages_{self.country}.txt"
        try:
            with open(pages_file_name, "r") as file:
                paginations = [int(num) for num in file.read().split(",")]
                paginations = sorted(list(set(paginations)))
        except FileNotFoundError:
            paginations = []
        url = self.base_url
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"
        }
        while self.num_pages_scraped < self.max_pages_to_scrape:
            pagination = self.num_pages_scraped
            if pagination in paginations:
                self.num_pages_scraped += 1
                continue
            paginations.append(pagination)
            url = self.base_url if pagination == 0 else f"{self.base_url}-{pagination}"
            print(f"{self.num_pages_scraped} - scrapping {url}...")
            req = urllib.request.Request(url, headers=headers)
            try:
                response = urllib.request.urlopen(req)
            except urllib.error.HTTPError as e:
                print("\tHTTP Error")
                print("\tError code:", e.code)
                print("\tError reason:", e.reason)
                print("\tError URL:", e.url)
                print("\tError headers:", e.headers)
                break
            soup = BeautifulSoup(response, "html.parser")
            img_containers = soup.find_all("div", class_="panel panel-grey")
            for img_container in img_containers:
                panel_body = img_container.find("div", class_="panel-body")
                car_model = panel_body.find("h4").find("a").get_text(strip=True)
                date = panel_body.find("small", class_="pull-right").get_text(
                    strip=True
                )
                panel_body_rows = panel_body.find_all("div", class_="row")
                sub_container_car_img = panel_body_rows[0]
                sub_container_plate_img = panel_body_rows[1]
                car_img_url = sub_container_car_img.find("a").find("img")["src"]
                plate_img_url = sub_container_plate_img.find("a").find("img")["src"]
                plate_number = sub_container_plate_img.find("a").find("img")["alt"]
                car_img_name = os.path.basename(car_img_url)
                plate_img_name = os.path.basename(plate_img_url)
                row = {
                    "date": date,
                    "car_model": car_model,
                    "car_img_name": car_img_name,
                    "car_img_url": car_img_url,
                    "plate_img_name": plate_img_name,
                    "plate_img_url": plate_img_url,
                    "plate_number": plate_number,
                }
                is_unique = self.license_plate_data[
                    (self.license_plate_data["car_img_url"] == car_img_url)
                    & (self.license_plate_data["plate_img_url"] == plate_img_url)
                ]
                if is_unique.empty:
                    self.license_plate_data = pd.concat(
                        [self.license_plate_data, pd.DataFrame([row])],
                        ignore_index=True,
                    )
            self.num_pages_scraped += 1
            self.license_plate_data.to_csv(
                self.csv_filepath, encoding="utf-8", index=False
            )
            with open(pages_file_name, "w") as file:
                file.write(",".join(map(str, paginations)))
            print(f"\t...scrapping {url} done {len(paginations)}!")
            time.sleep(randint(1, 10))


# Usage of the LicensePlateScraper class
scraper = LicensePlateScraper(country="pl")
scraper.parse()