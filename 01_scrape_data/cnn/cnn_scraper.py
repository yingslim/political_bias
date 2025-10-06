import time
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def debug_cnn_search():
    # Setup Chrome with minimal options
    chrome_opts = Options()
    # chrome_opts.add_argument('--headless')  # Comment out for debugging
    chrome_opts.add_argument('--disable-gpu')
    chrome_opts.add_argument('--no-sandbox')
    chrome_opts.add_argument('--disable-dev-shm-usage')
    chrome_opts.add_argument('--window-size=1920,1080')
    chrome_opts.add_argument('--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36')
    
    driver = webdriver.Chrome(options=chrome_opts)
    
    try:
        # Test basic navigation first
        logger.info("Testing basic CNN navigation...")
        driver.get('https://edition.cnn.com')
        logger.info(f"Successfully loaded CNN homepage. Title: {driver.title}")
        time.sleep(3)
        
        # Now try the search page
        search_url = 'https://edition.cnn.com/search?q=China&from=0&size=10&sort=newest'
        logger.info(f"Navigating to search URL: {search_url}")
        driver.get(search_url)
        
        # Wait a bit and check what we got
        time.sleep(5)
        logger.info(f"Search page title: {driver.title}")
        logger.info(f"Current URL: {driver.current_url}")
        
        # Save the page source
        with open('cnn_search_debug.html', 'w', encoding='utf-8') as f:
            f.write(driver.page_source)
        logger.info("Saved page source to cnn_search_debug.html")
        
        # Check for common elements
        soup = BeautifulSoup(driver.page_source, 'html.parser')
        
        # Look for various possible selectors
        selectors_to_try = [
            'div.container__item',
            '.search__result',
            '.cnn-search__result',
            '.search-results__item',
            'div[data-module="SearchResults"]',
            '.search-results',
            'article',
            '.card',
            'h3 a',  # Generic headline links
            'a[href*="/2024/"]',  # Links with dates
            'a[href*="/2025/"]'
        ]
        
        for selector in selectors_to_try:
            elements = soup.select(selector)
            if elements:
                logger.info(f"Found {len(elements)} elements with selector '{selector}'")
                # Print first few elements for inspection
                for i, elem in enumerate(elements[:3]):
                    logger.info(f"  Element {i+1}: {elem.name} - {elem.get('class')} - Text: {elem.get_text(strip=True)[:100]}...")
            else:
                logger.debug(f"No elements found with selector '{selector}'")
        
        # Check if we're being redirected or blocked
        page_text = driver.page_source.lower()
        if 'blocked' in page_text or 'captcha' in page_text or 'forbidden' in page_text:
            logger.error("Appears to be blocked by CNN")
        elif 'search' not in page_text:
            logger.warning("Page doesn't seem to contain search functionality")
        else:
            logger.info("Page appears to be loaded normally")
        
        # Try to find any links that look like articles
        all_links = soup.find_all('a', href=True)
        article_links = []
        for link in all_links:
            href = link['href']
            if any(year in href for year in ['/2024/', '/2025/', '/2023/']):
                article_links.append(href)
        
        logger.info(f"Found {len(article_links)} potential article links")
        for i, link in enumerate(article_links[:5]):
            logger.info(f"  Article link {i+1}: {link}")
        
        # Wait so you can see the browser window
        logger.info("Waiting 10 seconds so you can inspect the browser window...")
        time.sleep(10)
        
    except Exception as e:
        logger.error(f"Error during debugging: {e}")
        import traceback
        logger.error(traceback.format_exc())
    finally:
        driver.quit()

def test_alternative_approach():
    """Test using requests instead of Selenium"""
    import requests
    
    logger.info("Testing direct HTTP request to CNN search...")
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'Accept-Encoding': 'gzip, deflate',
        'Connection': 'keep-alive',
    }
    
    try:
        url = 'https://edition.cnn.com/search?q=China&from=0&size=10&sort=newest'
        response = requests.get(url, headers=headers, timeout=30)
        
        logger.info(f"HTTP Status: {response.status_code}")
        logger.info(f"Response headers: {dict(response.headers)}")
        
        with open('cnn_requests_debug.html', 'w', encoding='utf-8') as f:
            f.write(response.text)
        logger.info("Saved requests response to cnn_requests_debug.html")
        
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Check for search results
            search_results = soup.select('div.container__item, .search__result')
            logger.info(f"Found {len(search_results)} search results with requests")
            
    except Exception as e:
        logger.error(f"Error with requests approach: {e}")

if __name__ == '__main__':
    logger.info("Starting CNN debug session...")
    debug_cnn_search()
    logger.info("Selenium debugging complete.")
    
    logger.info("Testing alternative HTTP approach...")
    test_alternative_approach()
    logger.info("All debugging complete.")