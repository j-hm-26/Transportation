import os
import csv
import time
from tqdm import tqdm

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager


# ======================
# CONFIG
# ======================
BASE_URL = "https://www.transit.dot.gov/ntd/ntd-data"
ROOT_DOWNLOAD_DIR = os.path.abspath("ntd_all_years_xlsx")

YEARS = range(1997, 2027)
PAGES = [0, 1]

SCROLL_PASSES = 5
PAGE_LOAD_WAIT = 3
DOWNLOAD_TIMEOUT = 120  # seconds

LOG_FILE = "ntd_failed_downloads.csv"


# ======================
# SETUP DIRECTORIES
# ======================
os.makedirs(ROOT_DOWNLOAD_DIR, exist_ok=True)

for year in YEARS:
    os.makedirs(os.path.join(ROOT_DOWNLOAD_DIR, str(year)), exist_ok=True)


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


def scroll_page(passes=5):
    for _ in range(passes):
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(1)


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
# LOGGING SETUP
# ======================
log_rows = []


# ======================
# COLLECT PRODUCT PAGES
# ======================
product_pages = []  # (year, product_url)

for year in tqdm(YEARS, desc="Collecting product pages by year"):
    for page in PAGES:
        url = f"{BASE_URL}?year={year}&combine=&page={page}"
        driver.get(url)
        time.sleep(4)

        scroll_page(SCROLL_PASSES)

        found_any = False
        for a in driver.find_elements(By.TAG_NAME, "a"):
            href = a.get_attribute("href")
            if href and "/ntd/data-product/" in href:
                product_pages.append((year, href))
                found_any = True

        if page == 1 and not found_any:
            break


print(f"\nTotal product pages found: {len(product_pages)}")


# ======================
# VISIT PRODUCT PAGES & DOWNLOAD
# ======================
for year, product_url in tqdm(product_pages, desc="Processing product pages"):
    year_dir = os.path.join(ROOT_DOWNLOAD_DIR, str(year))
    driver.get(product_url)
    time.sleep(PAGE_LOAD_WAIT)

    scroll_page(2)

    links = driver.find_elements(By.TAG_NAME, "a")

    for link in tqdm(links, desc="  Checking files", leave=False):
        href = link.get_attribute("href")
        if not href or not href.lower().endswith(".xlsx"):
            continue

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
                    product_url,
                    href,
                    "Download timeout"
                ])

        except Exception as e:
            log_rows.append([
                year,
                product_url,
                href,
                str(e)
            ])


# ======================
# WRITE FAILURE LOG
# ======================
if log_rows:
    with open(LOG_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["year", "product_page", "file_url", "error"])
        writer.writerows(log_rows)

    print(f"\n⚠️  Some downloads failed. See {LOG_FILE}")
else:
    print("\nAll downloads completed successfully 🎉")


# ======================
# CLEANUP
# ======================
driver.quit()
