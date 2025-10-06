import sys
import json
import os
import time
import random
import logging
from datetime import datetime
from urllib.parse import urlencode, urlparse
from bs4 import BeautifulSoup

from selenium import webdriver
import undetected_chromedriver as uc
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from selenium.common.exceptions import TimeoutException

class ChinaComListScraper:
    def __init__(self, config=None, visible=True):
        self.config = {
            'base_url': 'http://query.china.com.cn/query/cn.html',
            'keyword': '美国',
            'output_dir': 'scraped_list',
            'start_date': '2022-05-01',
            'end_date': '2025-05-07',
            'max_pages': 10,
            'delay_min': 1,
            'delay_max': 3,
            'timeout': 15,
        }
        if config:
            self.config.update(config)
        self.start_date = datetime.strptime(self.config['start_date'], '%Y-%m-%d')
        self.end_date   = datetime.strptime(self.config['end_date'],   '%Y-%m-%d')

        os.makedirs(self.config['output_dir'], exist_ok=True)
        self.results_file = os.path.join(self.config['output_dir'], 'articles_list.json')
        self.logger = self._setup_logger()

        # load previously scraped articles
        self.articles = []
        self.visited_urls = set()
        if os.path.exists(self.results_file):
            try:
                with open(self.results_file, 'r', encoding='utf-8') as f:
                    self.articles = json.load(f)
                    self.visited_urls = {a['url'] for a in self.articles}
                    self.logger.info(f"Loaded {len(self.articles)} existing articles")
            except Exception as e:
                self.logger.error(f"Error loading existing articles: {e}")

        # Selenium setup
        chrome_opts = uc.ChromeOptions()
        if not visible:
            chrome_opts.add_argument('--headless')
        chrome_opts.add_argument('--ignore-certificate-errors')
        chrome_opts.add_argument('--disable-gpu')
        chrome_opts.add_argument('--no-sandbox')
        chrome_opts.add_argument('--disable-dev-shm-usage')
        chrome_opts.add_argument('--window-size=1920,1080')
        service = Service(ChromeDriverManager().install())
        self.driver = webdriver.Chrome(options=chrome_opts)
        if visible:
            try:
                self.driver.maximize_window()
            except Exception:
                pass

    def _setup_logger(self):
        logger = logging.getLogger('ListScraper')
        logger.setLevel(logging.INFO)
        fmt = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        ch = logging.StreamHandler(); ch.setLevel(logging.INFO); ch.setFormatter(fmt)
        fh = logging.FileHandler(os.path.join(self.config['output_dir'], 'scraper.log'), encoding='utf-8')
        fh.setLevel(logging.INFO); fh.setFormatter(fmt)
        logger.addHandler(ch); logger.addHandler(fh)
        return logger

    def save_progress(self):
        with open(self.results_file, 'w', encoding='utf-8') as f:
            json.dump(self.articles, f, ensure_ascii=False, indent=2)
        self.logger.info(f"Saved progress: {len(self.articles)} articles")

    def scrape(self):
        # build first page URL
        params = {
            'kw': self.config['keyword'],
            'exactExtend': 'order3',
            'dateRelevance': 'order1',
            'date_start': self.config['start_date'],
            'date_end': self.config['end_date']
        }
        first_page = f"{self.config['base_url']}?{urlencode(params)}"
        self.logger.info(f"Loading first page: {first_page}")
        self.driver.get(first_page)

        for current_page in range(1, self.config['max_pages'] + 1):
            self.logger.info(f"Processing page {current_page}")
            try:
                WebDriverWait(self.driver, self.config['timeout']).until(
                    EC.presence_of_all_elements_located((By.CSS_SELECTOR, 'li.row.mb-3.article-list'))
                )
            except TimeoutException:
                self.logger.error(f"No articles found on page {current_page}, aborting.")
                break

            # screenshot for debug
            path = os.path.join(self.config['output_dir'], f"page_{current_page}.png")
            try:
                self.driver.save_screenshot(path)
                self.logger.info(f"Screenshot saved: {path}")
            except Exception as e:
                self.logger.error(f"Failed screenshot: {e}")

            # parse articles
            soup = BeautifulSoup(self.driver.page_source, 'html.parser')
            items = soup.select('li.row.mb-3.article-list')
            self.logger.info(f"Page {current_page}: {len(items)} item(s) found")
            if not items:
                break

            # collect new URLs
            urls_to_process = []
            for it in items:
                link = it.select_one('h5 a') or it.find('a', href=lambda h: h and 'content_' in h)
                if not link:
                    continue
                url = link['href'].strip()
                if not url.startswith('http'):
                    base = f"{urlparse(self.driver.current_url).scheme}://{urlparse(self.driver.current_url).netloc}"
                    url = base + url if url.startswith('/') else base + '/' + url
                if url in self.visited_urls:
                    continue
                date_raw = it.select_one('div.date').get_text(strip=True).split()[0]
                dt = datetime.strptime(date_raw, '%Y-%m-%d')
                if not (self.start_date <= dt <= self.end_date):
                    continue
                urls_to_process.append({
                    'url':    url,
                    'title':  link.get_text(strip=True),
                    'date':   dt.strftime('%Y-%m-%d'),
                    'source': it.select_one('div.source').get_text(strip=True) if it.select_one('div.source') else ''
                })

            # scrape each new article
            articles_processed = 0
            main = self.driver.current_window_handle
            for info in urls_to_process:
                self.visited_urls.add(info['url'])
                articles_processed += 1
                try:
                    self.driver.execute_script("window.open('');")
                    self.driver.switch_to.window(self.driver.window_handles[-1])
                    info['content'] = self.scrape_article(info['url'])
                    self.articles.append(info)
                    self.driver.close()
                    self.driver.switch_to.window(main)
                except Exception as e:
                    self.logger.error(f"Error scraping {info['url']}: {e}")
                    self.driver.close()
                    self.driver.switch_to.window(main)
                if articles_processed % 5 == 0:
                    self.save_progress()

            self.logger.info(f"New articles this page: {articles_processed}")
            self.save_progress()

            # go to next page regardless of articles_processed
            if current_page < self.config['max_pages']:
                if not self.go_to_next_page(current_page):
                    self.logger.error("Could not navigate to next page, stopping.")
                    break
            else:
                self.logger.info("Reached max_pages limit.")
                break

        self.driver.quit()
        self.save_progress()
        self.logger.info(f"Scraping complete. Total articles: {len(self.articles)}")

    def go_to_next_page(self, current_page):
        """Click the numbered link via data-val, then fallback to arrows."""
        try:
            self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(random.uniform(1,2))

            # 1) click page N+1 via data-val
            target = str(current_page + 1)
            sel = f"a.pages[data-val='{target}']"
            try:
                btn = WebDriverWait(self.driver, 5).until(
                    EC.element_to_be_clickable((By.CSS_SELECTOR, sel))
                )
                self.logger.info(f"Clicking page {target}")
                btn.click()
                time.sleep(random.uniform(self.config['delay_min'], self.config['delay_max']))
                return True
            except Exception:
                self.logger.warning(f"No numeric link for page {target}")

            # 2) fallback: #nextPage
            try:
                btn = self.driver.find_element(By.ID, "nextPage")
                self.logger.info("Clicking #nextPage")
                btn.click()
                time.sleep(random.uniform(self.config['delay_min'], self.config['delay_max']))
                return True
            except Exception:
                pass

            # 3) fallback: “下一页”
            try:
                btn = self.driver.find_element(By.LINK_TEXT, "下一页")
                self.logger.info("Clicking ‘下一页’")
                btn.click()
                time.sleep(random.uniform(self.config['delay_min'], self.config['delay_max']))
                return True
            except Exception:
                pass

            # 4) fallback: “»”
            try:
                btn = self.driver.find_element(By.LINK_TEXT, "»")
                self.logger.info("Clicking ‘»’")
                btn.click()
                time.sleep(random.uniform(self.config['delay_min'], self.config['delay_max']))
                return True
            except Exception:
                pass

            return False
        except Exception as e:
            self.logger.error(f"Error in go_to_next_page: {e}")
            return False

    def scrape_article(self, url):
        self.logger.info(f"Scraping article: {url}")
        try:
            self.driver.get(url)
            time.sleep(random.uniform(2,3.5))
            WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.TAG_NAME, "p"))
            )
            art = BeautifulSoup(self.driver.page_source, 'html.parser')

            selectors = [
                'div#text_content', 'div.text_content', 'div.article_content',
                'div.content', 'article', 'div.rich_media_content'
            ]
            for sel in selectors:
                cont = art.select_one(sel)
                if cont:
                    ps = [p.get_text(strip=True) for p in cont.find_all('p') if len(p.get_text(strip=True))>10]
                    if ps:
                        return "\n".join(ps)

            # fallback
            ps = [p.get_text(strip=True) for p in art.find_all('p') if len(p.get_text(strip=True))>10]
            return "\n".join(ps) if ps else "Content extraction failed"

        except Exception as e:
            self.logger.error(f"Error scraping article {url}: {e}")
            return f"Error: {e}"

if __name__ == '__main__':
    config = {
        'max_pages': 1000,
        'keyword': '美国',
        'delay_min': 2,
        'delay_max': 4,
        'timeout': 20,
        'start_date': '2022-05-01',
        'end_date': '2025-05-07'
    }
    ChinaComListScraper(config, visible=True).scrape()
