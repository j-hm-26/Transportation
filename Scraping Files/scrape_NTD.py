import os
import time
from tqdm import tqdm

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager


# ======================
# CONFIG
# ======================
START_URL = (
    "https://www.transit.dot.gov/ntd/ntd-data"
    "?field_product_type_target_id=1026"
    "&year=2024"
    "&combine="
)

DOWNLOAD_DIR = os.path.abspath("ntd_2024_xlsx")
SCROLL_PASSES = 5
PAGE_LOAD_WAIT = 3

os.makedirs(DOWNLOAD_DIR, exist_ok=True)


# ======================
# CHROME DOWNLOAD SETTINGS
# ======================
options = webdriver.ChromeOptions()
options.add_argument("--start-maximized")

prefs = {
    "download.default_directory": DOWNLOAD_DIR,
    "download.prompt_for_download": False,
    "download.directory_upgrade": True,
    "safebrowsing.enabled": True,
}
options.add_experimental_option("prefs", prefs)

# options.add_argument("--headless=new")  # optional


driver = webdriver.Chrome(
    service=Service(ChromeDriverManager().install()),
    options=options
)


# ======================
# STAGE 1: LOAD MAIN PAGE
# ======================
driver.get(START_URL)
time.sleep(5)

for _ in tqdm(range(SCROLL_PASSES), desc="Scrolling main page"):
    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    time.sleep(2)


# ======================
# STAGE 2: COLLECT PRODUCT PAGES
# ======================
product_links = set()
anchors = driver.find_elements(By.TAG_NAME, "a")

for a in tqdm(anchors, desc="Collecting product links"):
    href = a.get_attribute("href")
    if href and "/ntd/data-product/" in href:
        product_links.add(href)

product_links = sorted(product_links)
print(f"\nFound {len(product_links)} product pages")


# ======================
# STAGE 3 + 4: VISIT & DOWNLOAD
# ======================
for product_url in tqdm(product_links, desc="Processing product pages"):
    driver.get(product_url)
    time.sleep(PAGE_LOAD_WAIT)

    xlsx_links = []
    for a in driver.find_elements(By.TAG_NAME, "a"):
        href = a.get_attribute("href")
        if href and href.lower().endswith(".xlsx"):
            xlsx_links.append(href)

    if not xlsx_links:
        continue

    for file_url in tqdm(
        xlsx_links,
        desc="  Triggering downloads",
        leave=False
    ):
        filename = file_url.split("/")[-1]
        filepath = os.path.join(DOWNLOAD_DIR, filename)

        # skip if already downloaded
        if os.path.exists(filepath):
            continue

        # THIS is the key: let Chrome do it
        driver.get(file_url)
        time.sleep(2)


# ======================
# CLEANUP
# ======================
driver.quit()
print("\nAll downloads complete")
