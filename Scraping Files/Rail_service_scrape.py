import requests
from bs4 import BeautifulSoup
import pandas as pd
from pathlib import Path
import logging
import re
import pyarrow as pa
import pyarrow.parquet as pq

#----------Scrape------------------------
def scrape_rail_service_data():
    # URL of the rail data page
    url = "https://www.stb.gov/reports-data/rail-service-data/"
    # Send a GET request to the page
    response = requests.get(url)
    # Check if the request was successful
    if response.status_code == 200:
        # Parse the HTML content using BeautifulSoup
        soup = BeautifulSoup(response.content, "html.parser")
        # Find the link to all .xlsx files
        xlsx_links = []
        for link in soup.find_all("a", href=True):
            href = link["href"]
            if href.endswith(".xlsx"):
                xlsx_links.append(href)
        # download each .xlsx file and store in the Data directory under its own folder
        for xlsx_link in xlsx_links:
            file_name = xlsx_link.split("/")[-1]
            # create a better file name
            file_name = file_name.replace(" ", "_").replace("-", "_")
            file_response = requests.get(xlsx_link)
            if file_response.status_code == 200:
                with open(f"./Data/Rail_Service_Data/{file_name}", "wb") as f:
                    f.write(file_response.content)
                print(f"Downloaded: {file_name}")
    else:
        print(f"Failed to retrieve the page. Status code: {response.status_code}")

# -----------Format to paraquet------------------------
            

#----------- Run -----------------
if __name__ == "__main__":
    # get user input to decide whether to scrape or not
    user_input = input("Do you want to scrape the rail service data? (y/n): ")
    if user_input.lower() == "y":
        scrape_rail_service_data()
    else:
        print("Skipping scraping.")
