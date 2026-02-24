import os
import csv
import time
import re
from tqdm import tqdm

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager


# ======================
# CONFIG
# ======================
START_URL = "https://www.stb.gov/reports-data/waybill/"
ROOT_DOWNLOAD_DIR = os.path.abspath("stb_waybill_data")

DOWNLOAD_TIMEOUT = 120  # seconds
PAGE_LOAD_WAIT = 4

LOG_FILE = "stb_waybill_failed_downloads.csv"


# ======================
# SETUP DIRECTORIES
# ======================
os.makedirs(ROOT_DOWNLOAD_DIR, exist_ok=True)


# ======================
# CHROME SETUP
# ======================
options = webdriver.ChromeOptions()
options.add_argument("--start-maximized")

prefs = {
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


def human_click(element, pause=0.4):
    ActionChains(driver).move_to_element(element).pause(pause).click().perform()


def wait_for_download(download_dir, timeout=120):
    start = time.time()
    while time.time() - start < timeout:
        if not any(f.endswith(".crdownload") for f in os.listdir(download_dir)):
            return True
        time.sleep(1)
    return False


# ======================
# LOAD PAGE
# ======================
driver.get(START_URL)
time.sleep(PAGE_LOAD_WAIT)


# ======================
# FIND ZIP LINKS
# ======================
zip_links = []

anchors = driver.find_elements(By.TAG_NAME, "a")
for a in anchors:
    href = a.get_attribute("href")
    if href and href.lower().endswith(".zip"):
        zip_links.append((a, href))

print(f"Found {len(zip_links)} ZIP files")


# ======================
# DOWNLOAD FILES
# ======================
log_rows = []

for link, href in tqdm(zip_links, desc="Downloading Waybill ZIP files"):

    # Infer year from filename or URL
    year_match = re.search(r"(19|20)\d{2}", href)
    year = year_match.group(0) if year_match else "unknown"

    year_dir = os.path.join(ROOT_DOWNLOAD_DIR, year)
    os.makedirs(year_dir, exist_ok=True)

    filename = href.split("/")[-1]
    filepath = os.path.join(year_dir, filename)

    if os.path.exists(filepath):
        continue

    try:
        # Set Chrome download directory dynamically
        driver.execute_cdp_cmd(
            "Page.setDownloadBehavior",
            {
                "behavior": "allow",
                "downloadPath": year_dir,
            },
        )

        human_click(link)
        success = wait_for_download(year_dir, DOWNLOAD_TIMEOUT)

        if not success:
            log_rows.append([
                year,
                href,
                "Download timeout"
            ])

    except Exception as e:
        log_rows.append([
            year,
            href,
            str(e)
        ])


# ======================
# WRITE FAILURE LOG
# ======================
if log_rows:
    with open(LOG_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["year", "file_url", "error"])
        writer.writerows(log_rows)

    print(f"\n Some downloads failed. See {LOG_FILE}")
else:
    print("\nAll Waybill ZIP files downloaded successfully 🎉")


# ======================
# CLEANUP
# ======================
driver.quit()
