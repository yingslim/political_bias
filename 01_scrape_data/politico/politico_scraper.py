import json
import time
import traceback
import re
from datetime import datetime
import argparse
import urllib.parse

import undetected_chromedriver as uc
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, WebDriverException, StaleElementReferenceException

class PoliticoScraper:
    def __init__(self, headless=False,
                 start_date="2022-05-05", end_date="2025-05-14"):
        """
        headless: run without UI
        start_date/end_date: YYYY-MM-DD strings defining the desired date window
        """
        self.base_query   = "china"
        self.url_base     = "https://www.politico.com/search"
        self.output_file  = "politico_china_articles.json"
        # date range
        self.range_start = datetime.strptime(start_date, "%Y-%m-%d")
        self.range_end   = datetime.strptime(end_date,   "%Y-%m-%d")

        # setup undetected-chrome with timeout
        opts = Options()
        if headless:
            opts.add_argument("--headless")
        opts.add_argument("--disable-gpu")
        opts.add_argument("--no-sandbox")
        opts.add_argument("--disable-dev-shm-usage")
        opts.add_argument("--window-size=1920,1080")
        opts.add_argument("--disable-notifications")
        opts.add_argument("--disable-popup-blocking")
        opts.add_argument(
            "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/112.0.0.0 Safari/537.36"
        )
        self.driver = uc.Chrome(options=opts)
        # increase page load timeout to reduce timeouts
        self.driver.set_page_load_timeout(60)

        # load or init articles
        try:
            with open(self.output_file, 'r', encoding='utf-8') as f:
                self.articles = json.load(f)
            print(f"Loaded {len(self.articles)} existing articles")
        except (FileNotFoundError, json.JSONDecodeError):
            self.articles = []
            with open(self.output_file,'w',encoding='utf-8') as f:
                json.dump(self.articles,f)
            print("Created new articles file")

        self.processed_urls = {a['url'] for a in self.articles}

    def fetch_url(self, url: str) -> bool:
        """Attempt up to 3 times to load a URL, returning True if successful"""
        for attempt in range(1, 4):
            try:
                self.driver.get(url)
                return True
            except Exception as e:
                print(f"    ! load attempt {attempt} failed: {e}")
                time.sleep(2)
        print(f"    ! failed to load {url} after 3 attempts")
        return False

    def _parse_date(self, raw: str) -> datetime:
        raw = raw.strip()
        raw = re.sub(r"\s+[A-Z]{2,4}$", "", raw)
        try:
            return datetime.fromisoformat(raw)
        except ValueError:
            pass
        fmts = [
            "%m/%d/%y %I:%M %p",
            "%m/%d/%Y %I:%M %p",
            "%m/%d/%Y",
            "%b %d, %Y %I:%M %p",
            "%B %d, %Y %I:%M %p",
            "%b %d, %Y",
            "%B %d, %Y",
            "%Y-%m-%dT%H:%M:%S",
            "%Y-%m-%d %H:%M:%S"
        ]
        for fmt in fmts:
            try:
                return datetime.strptime(raw, fmt)
            except ValueError:
                continue
        raise ValueError(f"Unknown date format: '{raw}'")

    def dismiss_popup(self):
        try:
            selectors = [
                "button.close","button.dismiss","div.modal-close",
                "button.x-close","button[aria-label='Close']",
                ".modal-backdrop","div.modal-content button",
                "button.button-close",".dialog-close",".close-button"
            ]
            for sel in selectors:
                for btn in self.driver.find_elements(By.CSS_SELECTOR, sel):
                    if btn.is_displayed(): btn.click(); time.sleep(0.5)
            self.driver.execute_script(
                """
                document.querySelectorAll('.piano-sign-up, .subscribe-form, .newsletter-signup').forEach(e=>e.remove());
                document.querySelectorAll('.modal, .modal-backdrop').forEach(e=>e.remove());
                document.body.style.overflow = 'auto'; document.body.style.position='static';
                """
            )
        except Exception:
            pass

    def handle_cookie_consent(self):
        try:
            for sel in [
                "button#onetrust-accept-btn-handler",
                "button.accept-cookies","button.cookie-accept",
                "[id*='cookie'] button[id*='accept']",
                "[class*='cookie'] button[class*='accept']",
                "button[aria-label*='Accept']"
            ]:
                for btn in self.driver.find_elements(By.CSS_SELECTOR, sel):
                    if btn.is_displayed(): btn.click(); time.sleep(0.5); return
        except Exception:
            pass

    def extract_article(self, url: str, override_title: str=None) -> dict:
        print(f"  → Fetching {url}")
        if not self.fetch_url(url):
            return None
        time.sleep(2)
        self.handle_cookie_consent(); self.dismiss_popup()

        art = {"url":url, "title":override_title or "", "date":"", "authors":"", "content":"", "scraped_at":datetime.now().isoformat()}
        # title
        if not override_title:
            try:
                h1 = WebDriverWait(self.driver,5).until(EC.presence_of_element_located((By.CSS_SELECTOR,"h1.headline,h1.article-title")))
                art['title'] = h1.text.strip()
            except Exception: pass
        # date
        dt_text=""
        for sel in ["time[itemprop='datePublished']","p.timestamp time","p.story-meta__timestamp time","time.story-meta__timestamp","span.is-hidden[itemprop='dateModified']"]:
            try:
                tm=self.driver.find_element(By.CSS_SELECTOR,sel)
                dt_text = tm.get_attribute('datetime') or tm.text
                if dt_text: break
            except: continue
        art['date']=dt_text.strip()
        # authors
        authors=[]
        try:
            for a in self.driver.find_elements(By.CSS_SELECTOR,"p.story-meta__authors a,span.vcard"):
                t=a.text.strip();
                if t: authors.append(t)
        except: pass
        art['authors'] = ', '.join(authors)
        # content
        paras=[]
        try:
            self.dismiss_popup()
            for p in self.driver.find_elements(By.CSS_SELECTOR,"div.story-text p,div.story-text__paragraph"):
                t=p.text.strip();
                if t: paras.append(t)
        except StaleElementReferenceException:
            time.sleep(0.5); self.dismiss_popup()
            for p in self.driver.find_elements(By.CSS_SELECTOR,"div.story-text p,div.story-text__paragraph"):
                t=p.text.strip();
                if t: paras.append(t)
        except: pass
        if not paras:
            try:
                js=("var ps=[]; document.querySelectorAll('article p, .article-content p').forEach(function(p){"
                    "var t=p.innerText.trim(); if(t&&!p.closest('.modal')&&!p.closest('header')&&!p.closest('footer')) ps.push(t);} );"
                    "return ps.join('\n\n');")
                text=self.driver.execute_script(js)
                paras=text.split('\n\n') if text else []
            except: pass
        art['content']='\n\n'.join(paras)
        return art
    def save_article(self, art:dict) -> bool:
        """
        Append and persist articles if they're new, in date range, and not a duplicate of content.
        """
        if not art:
            return False
        url = art['url']
        # Skip already processed URL
        if url in self.processed_urls:
            print(f"  • skipping duplicate URL: {url}")
            return False
        # Simple content-based deduplication
        existing = {a['content'] for a in self.articles}
        if art.get('content') in existing:
            print("  • skipping duplicate content")
            self.processed_urls.add(url)
            return False
        # Date filtering
        raw_date = art.get('date', '').strip()
        if not raw_date:
            print(f"  • skipping {url} due to missing date")
            self.processed_urls.add(url)
            return False
        try:
            dt = self._parse_date(raw_date)
        except Exception as e:
            print(f"  • skipping due to date parse error: {e}")
            self.processed_urls.add(url)
            return False
        if not (self.range_start <= dt <= self.range_end):
            print(f"  • skipping as {dt.date()} is out of range")
            self.processed_urls.add(url)
            return False
        # Save the article
        self.articles.append(art)
        self.processed_urls.add(url)
        with open(self.output_file, 'w', encoding='utf-8') as f:
            json.dump(self.articles, f, indent=2, ensure_ascii=False)
        print(f"Saved: {art['title']} ({dt.date()})")
        return True


    def get_search_url(self,page:int)->str:
        return f"{self.url_base}/{page}?s=newest&q={urllib.parse.quote(self.base_query)}"

    def scrape_search_results(self,max_pages=None)->int:
        page,new_count=1,0
        while max_pages is None or page<=max_pages:
            url=self.get_search_url(page)
            print(f"\n=== Page {page}: {url}")
            if not self.fetch_url(url):
                page+=1; continue
            time.sleep(2); self.handle_cookie_consent(); self.dismiss_popup()
            try:
                blocks=self.driver.find_elements(By.CSS_SELECTOR,"ul.story-frag-list li article.story-frag,article.story-frag")
            except: break
            to_visit=[]
            for blk in blocks:
                try:
                    link=blk.find_element(By.CSS_SELECTOR,"h3 a")
                    href,title=link.get_attribute('href'),link.text.strip()
                    if href and href not in self.processed_urls: to_visit.append((href,title))
                except: continue
            for href,title in to_visit:
                art=self.extract_article(href,override_title=title)
                if self.save_article(art): new_count+=1
                time.sleep(1)
            page+=1
        print(f"\nDone: {new_count} new articles saved.")
        return new_count

    def close(self):
        self.driver.quit()
        print("Browser closed.")


def run_scraper():
    parser=argparse.ArgumentParser()
    parser.add_argument("--headless",action="store_true")
    parser.add_argument("--start",default="2022-05-05",help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end",default="2025-05-14",help="End date (YYYY-MM-DD)")
    parser.add_argument("--pages",type=int,default=None,help="Max pages to scrape")
    args=parser.parse_args()
    scraper=PoliticoScraper(headless=args.headless,start_date=args.start,end_date=args.end)
    try:
        print(f"Scraping up to {args.pages or 'all'} pages...")
        scraper.scrape_search_results(max_pages=args.pages)
    except Exception as e:
        print("Fatal error:",e)
        traceback.print_exc()
    finally:
        scraper.close()

if __name__=="__main__":
    run_scraper()
