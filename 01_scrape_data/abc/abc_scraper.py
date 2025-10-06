import os
import json
import time
import re
import logging
import hashlib
from datetime import datetime, timedelta
import argparse
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
import undetected_chromedriver as uc
from webdriver_manager.chrome import ChromeDriverManager

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("abc_scraper.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ABCNewsSeleniumScraper:
    def __init__(self, search_term, start_date, end_date, output_dir="abc_news_articles"):
        self.search_term = search_term
        self.base_url = "https://abcnews.go.com/search"
        self.start_date = datetime.strptime(start_date, '%m/%d/%Y')
        self.end_date = datetime.strptime(end_date, '%m/%d/%Y')
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        # Aggregate file
        self.aggregate_file = os.path.join(self.output_dir, "all_articles.json")
        self.all_articles = []
        self.articles_processed = set()
        if os.path.exists(self.aggregate_file):
            try:
                with open(self.aggregate_file, 'r', encoding='utf-8') as f:
                    self.all_articles = json.load(f)
                for art in self.all_articles:
                    self.articles_processed.add(art.get("url"))
                logger.info(f"Loaded {len(self.all_articles)} existing articles from {self.aggregate_file}")
            except Exception as e:
                logger.warning(f"Failed to load aggregate file: {e}")

        # Chrome opts
        opts = Options()
        opts.add_argument("--headless")
        opts.add_argument("--no-sandbox")
        opts.add_argument("--disable-dev-shm-usage")
        opts.add_argument("--disable-gpu")
        opts.add_argument("--window-size=1920,1080")
        opts.add_argument(
            "--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36"
        )
        self.driver = uc.Chrome(options=opts)     # uc will fetch a compatible driver

        logger.info(f"Initialized scraper for '{search_term}' from {start_date} to {end_date}")

    def parse_date(self, date_text):
        for fmt in ['%m/%d/%Y', '%b %d, %Y', '%B %d, %Y', '%m.%d.%Y']:
            try:
                return datetime.strptime(date_text.strip(), fmt)
            except:
                continue
        return None

    def generate_article_id(self, url):
        return hashlib.md5(url.encode()).hexdigest()

    def extract_article_content(self, url):
        data = {'url': url, 'title': '', 'date': '', 'author': '', 'content': ''}
        try:
            self.driver.get(url)
            WebDriverWait(self.driver, 10).until(EC.presence_of_element_located((By.TAG_NAME, 'body')))
            time.sleep(2)

            # VIDEO
            if '/video/' in url or self.driver.find_elements(By.CSS_SELECTOR, 'section.video__section'):
                # Title
                try:
                    data['title'] = self.driver.find_element(By.CSS_SELECTOR, 'h2.video-info-module__text--title').text.strip()
                except: pass
                # Subtitle (vod) or description
                for sel in ['h3.video-info-module__text--subtitle__vod', 'div.Video__Description']:
                    try:
                        data['content'] = self.driver.find_element(By.CSS_SELECTOR, sel).text.strip()
                        break
                    except: pass
                # Date timestamp or Video Head
                try:
                    data['date'] = self.driver.find_element(By.CSS_SELECTOR, 'h3.video-info-module__text--subtitle__timestamp').text.strip()
                except:
                    try:
                        spans = self.driver.find_elements(By.CSS_SELECTOR, 'div.Video__Head span')
                        data['date'] = spans[-1].text.strip() if spans else ''
                    except: pass
            else:
                # ARTICLE PAGE
                # Title
                for sel in ['h1.Article__Headline', 'h1.article-headline', 'h1.article-title', 'h1[class*="headline"]', 'h1']:
                    try:
                        t = self.driver.find_element(By.CSS_SELECTOR, sel).text.strip()
                        if t:
                            data['title'] = t
                            break
                    except: pass
                # Date: timestamp or prism-byline
                dt = ''
                for sel in ['div.TimeStamp__Date', 'span.article-date', 'time']:
                    try:
                        dt = self.driver.find_element(By.CSS_SELECTOR, sel).text.strip(); break
                    except: pass
                if not dt:
                    try:
                        by = self.driver.find_element(By.CSS_SELECTOR, '[data-testid="prism-byline"]')
                        txt = by.text.split('\n')
                        for part in txt:
                            if re.search(r'\d{4}', part): dt = part.strip(); break
                    except: pass
                data['date'] = dt
                # Author
                for sel in ['[data-testid="prism-byline"] a', '.byline a', '.byline', '[class*="byline"]']:
                    try:
                        txt = self.driver.find_element(By.CSS_SELECTOR, sel).text.strip().replace('By ', '')
                        if txt: data['author'] = txt; break
                    except: pass
                # Content
                try:
                    bd = self.driver.find_element(By.CSS_SELECTOR, '[data-testid="prism-article-body"]')
                    paras = bd.find_elements(By.TAG_NAME, 'p')
                    data['content'] = '\n'.join(p.text.strip() for p in paras if p.text.strip())
                except: pass
        except Exception as e:
            logger.error(f"Error extracting {url}: {e}")
        return data

    def process_article(self, url, title=''):
        if url in self.articles_processed: return
        art = self.extract_article_content(url)
        if not art['title'] and title: art['title'] = title
        self.all_articles.append(art)
        self.articles_processed.add(url)
        try:
            with open(self.aggregate_file, 'w', encoding='utf-8') as f:
                json.dump(self.all_articles, f, indent=2, ensure_ascii=False)
            print(f"[Saved to {self.aggregate_file}]")
        except Exception as ex:
            logger.error(f"Error writing aggregate: {ex}")

    def scrape_search_results(self, page=1):
        count = 0
        search_url = f"{self.base_url}?searchtext={self.search_term}&page={page}&sort=date"
        logger.info(f"Scraping {search_url}")
        self.driver.get(search_url)
        WebDriverWait(self.driver, 10).until(EC.presence_of_element_located((By.CSS_SELECTOR, 'section.ContentRoll, div.search-results')))
        time.sleep(2)
        # find containers
        containers = []
        for sel in ['section.ContentRoll__Item', 'div.ContentRoll__Item', 'article.ContentList__Item', 'div.search-result', 'div.search-results__item']:
            els = self.driver.find_elements(By.CSS_SELECTOR, sel)
            if els: containers = els; break
        if not containers: return 0
        links = []
        for item in containers:
            for sel in ['a.AnchorLink', 'h2 a', 'h3 a', 'a[href*="/story/"]', 'a[href*="/video/"]', 'a[href*="/live-updates"]']:
                try:
                    a = item.find_element(By.CSS_SELECTOR, sel)
                    href = a.get_attribute('href')
                    if href and 'abcnews.go.com' in href and not re.search(r'\.(png|jpe?g|gif)(\?.*)?$', href, re.IGNORECASE):
                        title = a.text.strip()
                        if href not in [u for u,_ in links]: links.append((href,title))
                        break
                except: pass
        for href,title in links:
            self.process_article(href, title)
            count +=1
        return count

    def run(self, max_pages=50):
        for p in range(1, max_pages+1):
            c = self.scrape_search_results(p)
            if c==0 and p>1: break
            time.sleep(2)
        logger.info(f"Done: {len(self.all_articles)} articles; saved to {self.aggregate_file}")

    def cleanup(self):
        try: self.driver.quit()
        except: pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Scrape ABC News articles by search term and date range")
    parser.add_argument('--search_term', default='china')
    parser.add_argument('--start_date', default='05/05/2022')
    parser.add_argument('--end_date', default='05/11/2025')
    parser.add_argument('--output_dir', default='abc_news_articles')
    parser.add_argument('--max_pages', type=int, default=3)
    args = parser.parse_args()
    scraper = ABCNewsSeleniumScraper(args.search_term, args.start_date, args.end_date, args.output_dir)
    scraper.run(max_pages=args.max_pages)
