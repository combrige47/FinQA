from selenium import webdriver
from selenium.webdriver.common.by import By
from bs4 import BeautifulSoup
import datetime
from selenium.webdriver.firefox.service import Service
from selenium.webdriver.firefox.options import Options
from webdriver_manager.firefox import GeckoDriverManager
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import random, time, sys
from pathlib import Path
root_path = Path(__file__).parent.parent
sys.path.append(str(root_path))
from vector.milvus_stroe import MilvusImporter
from data import kb_service
import akshare as ak

DAYILY_COLLECTION = "kb_1783673983010"
DAYILY_KB_ID = 11
DAYILY_UPLOADER_ID = 5
firefox_options = Options()
firefox_options.add_argument("--headless")
firefox_options.add_argument("--no-sandbox")
firefox_options.add_argument("--disable-dev-shm-usage")
firefox_options.set_preference("permissions.default.image", 2) 
firefox_options.page_load_strategy = "none"

def get_today_news_link():
    df = ak.stock_info_cjzc_em()
    links = df["链接"].tolist()
    return links[0]

def get_content(link_url):
    service = Service(
        executable_path="/home/kang/.wdm/drivers/geckodriver/linux-arm64/v0.37.0/geckodriver"
    )
    driver = webdriver.Firefox(
        service=service,
        options=firefox_options
    )
    driver.get(link_url)
    content = WebDriverWait(driver, 20).until(
        EC.visibility_of_element_located((By.ID, "ContentBody"))
    )
    text = content.text
    driver.quit()
    return text

def save_content_to_file(content, filename):
    with open(filename, "w", encoding="utf-8") as f:
        f.write(content)


def crawl_notice():
    link = get_today_news_link()
    content = get_content(link)
    today = datetime.date.today().strftime("%Y-%m-%d")
    filename = f"notice_{today}.md"
    save_content_to_file(content, filename)
    with open(filename,"rb") as f:
        filebytes = f.read()
    importer = MilvusImporter(
        collection_name=DAYILY_COLLECTION,
    )
    return kb_service.upload_to_kb(kb_id=DAYILY_KB_ID,uploader_id=DAYILY_UPLOADER_ID,
        original_filename=filename,file_bytes=filebytes,importer=importer)

if __name__ == "__main__":
    print(get_today_news_link())