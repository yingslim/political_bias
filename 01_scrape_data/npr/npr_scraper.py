# limtiation: free 20 articles daily on NPR website
import time
import json
import datetime
from selenium import webdriver
import undetected_chromedriver as uc
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
from webdriver_manager.chrome import ChromeDriverManager

class NPRScraper:
    def __init__(self, headless=False):
        self.setup_driver(headless)
        self.articles_data = []
        self.output_file = "npr_china_articles.json"
        
    def setup_driver(self, headless):
        """Set up the Chrome WebDriver"""
        chrome_options = uc.ChromeOptions()
        if headless:
            chrome_options.add_argument("--headless")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--window-size=1920,1080")
        chrome_options.add_argument("--ignore-certificate-errors")
        chrome_options.add_argument("--ignore-ssl-errors")
        chrome_options.add_argument("--disable-web-security")
        chrome_options.add_argument("--allow-running-insecure-content")
        chrome_options.add_argument("--disable-features=IsolateOrigins,site-per-process")
        
        # Initialize the Chrome driver
        self.driver = uc.Chrome(options=chrome_options)
        
    def save_article_to_json(self, article_data):
        """Save a single article to the JSON file"""
        try:
            # Read existing data if file exists
            try:
                with open(self.output_file, 'r') as f:
                    try:
                        existing_data = json.load(f)
                    except json.JSONDecodeError:
                        existing_data = []
            except FileNotFoundError:
                existing_data = []
            
            # Append new article
            existing_data.append(article_data)
            
            # Write back to file
            with open(self.output_file, 'w') as f:
                json.dump(existing_data, f, indent=2)
                
            print(f"Article saved: {article_data['title']}")
        except Exception as e:
            print(f"Error saving article to JSON: {e}")
        
    def parse_date(self, date_text):
        """Parse date from various formats into a standardized format"""
        import datetime
        import re
        
        if not date_text or date_text == "Unknown":
            return "Unknown"
            
        try:
            # Clean up date text
            date_text = date_text.replace('\n', ' ').strip()
            
            # Remove any trailing bullet points or other common separators
            date_text = re.sub(r'[•|·|,]\s*$', '', date_text.strip())
            
            # For formats like "MAY 5, 2025 · 4:07 PM ET"
            match = re.search(r'([A-Za-z]+)\s+(\d{1,2}),\s+(\d{4})', date_text, re.IGNORECASE)
            if match:
                month_str, day_str, year_str = match.groups()
                month_dict = {
                    'january': 1, 'february': 2, 'march': 3, 'april': 4,
                    'may': 5, 'june': 6, 'july': 7, 'august': 8,
                    'september': 9, 'october': 10, 'november': 11, 'december': 12,
                    'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
                    'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12
                }
                month_num = month_dict.get(month_str.lower(), 1)
                return f"{year_str}-{month_num:02d}-{int(day_str):02d}"
            
            # Try parsing common formats
            for fmt in ["%B %d, %Y", "%b %d, %Y", "%Y-%m-%d", "%m/%d/%Y"]:
                try:
                    return datetime.datetime.strptime(date_text, fmt).strftime("%Y-%m-%d")
                except ValueError:
                    continue
                    
            # If we get here, try more complex approaches
            # Extract date part from formats with time like "MAY 5, 2025 · 4:07 PM ET"
            time_split = re.split(r'[·|\|]', date_text)
            if len(time_split) > 1:
                date_part = time_split[0].strip()
                try:
                    # Try to parse just the date part
                    for fmt in ["%B %d, %Y", "%b %d, %Y"]:
                        try:
                            return datetime.datetime.strptime(date_part, fmt).strftime("%Y-%m-%d")
                        except ValueError:
                            continue
                except Exception:
                    pass
            
            # If nothing worked, print a warning and return as is
            print(f"Warning: Could not parse date format: '{date_text}'")
            return date_text
            
        except Exception as e:
            print(f"Error parsing date '{date_text}': {e}")
            return date_text
        
    def extract_article_content(self, url):
        """Extract content from an individual article page"""
        try:
            print(f"Extracting content from: {url}")
            self.driver.get(url)
            
            # Wait for the page to load with longer timeout
            WebDriverWait(self.driver, 30).until(
                EC.presence_of_element_located((By.TAG_NAME, "body"))
            )
            
            # Add pause for complex JS rendering
            time.sleep(3)
            
            # Extract title - check various selectors based on the screenshots
            title = "Unknown Title"
            for title_selector in [
                "h1.storytitle", "h1.title", ".storytitle", "#storytitle", 
                "h1", ".contentheader-title", ".contentheader h1"
            ]:
                try:
                    title_elem = self.driver.find_element(By.CSS_SELECTOR, title_selector)
                    title = title_elem.text.strip()
                    if title:
                        break
                except NoSuchElementException:
                    continue
            
            # Extract date based on the DOM structure in screenshots
            date = "Unknown"
            for date_selector in [
                ".dateblock", ".date", "time", ".story-meta time", 
                ".story-meta .date", "time.dateblock", 
                ".story-meta_one-inner time", ".dateblock span.date"
            ]:
                try:
                    date_elem = self.driver.find_element(By.CSS_SELECTOR, date_selector)
                    date_raw = date_elem.text.strip() or date_elem.get_attribute("datetime")
                    date = self.parse_date(date_raw)
                    if date != "Unknown":
                        break
                except NoSuchElementException:
                    continue
            
            # Extract author with fallback to "NPR Staff"
            author = "NPR Staff"  # Default value when author not found
            for author_selector in [
                "[aria-label='byline']", "a[rel='author']", ".byline", 
                ".byline-name", ".story-meta .byline", ".storytext .byline"
            ]:
                try:
                    author_elem = self.driver.find_element(By.CSS_SELECTOR, author_selector)
                    author_text = author_elem.text.strip()
                    if author_text:
                        author = author_text
                        break
                except NoSuchElementException:
                    continue
            
            # Extract content paragraphs - using multiple selectors based on screenshots
            body_content = ""
            for content_selector in [
                "#storytext p", ".storytext p", ".story-text p", 
                "article p", "#storytext", ".storyText p", 
                "#storyText p", ".storytext-container p",
                ".storytext div p", "#mainContent p"
            ]:
                try:
                    content_elements = self.driver.find_elements(By.CSS_SELECTOR, content_selector)
                    if content_elements:
                        body_content = "\n\n".join([p.text.strip() for p in content_elements if p.text.strip()])
                        if body_content:
                            break
                except Exception:
                    continue
            
            # If we still don't have content, try a more general approach
            if not body_content:
                try:
                    article_elem = self.driver.find_element(By.TAG_NAME, "article")
                    paragraphs = article_elem.find_elements(By.TAG_NAME, "p")
                    body_content = "\n\n".join([p.text.strip() for p in paragraphs if p.text.strip()])
                except Exception as e:
                    print(f"Error extracting body content with general approach: {e}")
            
            # Create article data dictionary
            article_data = {
                "title": title,
                "date": date,
                "author": author,
                "url": url,
                "body_content": body_content
            }
            
            # Save individual article to JSON
            self.save_article_to_json(article_data)
            
            return article_data
            
        except Exception as e:
            print(f"Error extracting article content from {url}: {e}")
            # Try to take a screenshot of the error state
            try:
                self.driver.save_screenshot(f"error_article_{url.split('/')[-1]}.png")
                print(f"Screenshot saved for error on article")
            except:
                pass
            return None
    
    def scrape_search_results(self, start_date, end_date):
        """Scrape articles from search results within the specified date range"""
        base_url = "https://www.npr.org/search/"
        page = 1
        
        # Convert dates to datetime objects for comparison
        start_date_obj = datetime.datetime.strptime(start_date, "%Y-%m-%d")
        end_date_obj = datetime.datetime.strptime(end_date, "%Y-%m-%d")
        
        continue_scraping = True
        
        while continue_scraping:
            search_url = f"{base_url}?query=china&page={page}"
            print(f"Scraping page {page}: {search_url}")
            
            try:
                self.driver.get(search_url)
                
                # Wait for page to load
                WebDriverWait(self.driver, 30).until(
                    EC.presence_of_element_located((By.TAG_NAME, "body"))
                )
                
                # Add a pause to ensure JS rendering completes
                time.sleep(5)
                
                # Save screenshot for debugging
                self.driver.save_screenshot(f"search_page_{page}.png")
                
                # Get all article links from search results
                article_links = []
                for link_selector in [
                    "article a", ".item-info a", ".title a", "h2 a", 
                    "a.title", ".result-item a", ".result-item h3 a"
                ]:
                    try:
                        links = self.driver.find_elements(By.CSS_SELECTOR, link_selector)
                        for link in links:
                            url = link.get_attribute("href")
                            if url and ("/2022/" in url or "/2023/" in url or "/2024/" in url or "/2025/" in url):
                                if url not in article_links:
                                    article_links.append(url)
                    except Exception as e:
                        print(f"Error finding links with selector {link_selector}: {e}")
                
                if not article_links:
                    print("No article links found on this page. Stopping search.")
                    break
                
                print(f"Found {len(article_links)} article links on page {page}")
                
                # Process each article
                for article_url in article_links:
                    article_data = self.extract_article_content(article_url)
                    
                    # If article has a date, check if it's within our range
                    if article_data and article_data["date"] != "Unknown":
                        try:
                            article_date = datetime.datetime.strptime(article_data["date"], "%Y-%m-%d")
                            
                            # Stop if we've gone before our start date
                            if article_date < start_date_obj:
                                print(f"Found article from {article_data['date']} which is before start date {start_date}. Stopping.")
                                continue_scraping = False
                                break
                            
                            # Skip if after our end date
                            if article_date > end_date_obj:
                                print(f"Skipping article from {article_data['date']} which is after end date {end_date}")
                                continue
                        except Exception as e:
                            print(f"Error comparing dates: {e}")
                    
                # Move to next page
                page += 1
                
                # Respect rate limits
                time.sleep(5)
                
            except Exception as e:
                print(f"Error on page {page}: {e}")
                # Take screenshot for debugging
                self.driver.save_screenshot(f"error_page_{page}.png")
                break
                
    def scrape_with_direct_date_filter(self, start_date, end_date):
        """Alternative approach using NPR's direct date filter in URL"""
        # Convert dates to format expected by NPR URL
        start_date_obj = datetime.datetime.strptime(start_date, "%Y-%m-%d")
        end_date_obj = datetime.datetime.strptime(end_date, "%Y-%m-%d")
        
        # Format for NPR's URL: startDate=05-05-2022&endDate=05-12-2022
        start_date_url = start_date_obj.strftime("%m-%d-%Y")
        end_date_url = end_date_obj.strftime("%m-%d-%Y")
        
        base_url = "https://www.npr.org/search"
        search_url = f"{base_url}?query=china&startDate={start_date_url}&endDate={end_date_url}"
        
        print(f"Using direct date filter URL: {search_url}")
        
        page = 1
        continue_scraping = True
        
        while continue_scraping and page <= 10:  # Limit to 10 pages for safety
            try:
                current_url = f"{search_url}&page={page}"
                print(f"Scraping page {page}: {current_url}")
                
                self.driver.get(current_url)
                time.sleep(5)  # Give time for page to load
                
                # Take screenshot for debugging
                self.driver.save_screenshot(f"page_{page}.png")
                print(f"Saved screenshot as page_{page}.png")
                
                # Try multiple selectors for finding articles
                article_urls = []
                
                for selector in [
                    "article a[href*='/20']", 
                    ".item-info a[href*='/20']",
                    "h2 a[href*='/20']", 
                    ".title a[href*='/20']",
                    "a.title[href*='/20']",
                    "a[href*='npr.org/20']",
                    ".result-info h3 a",
                    ".result-list article a"
                ]:
                    try:
                        links = self.driver.find_elements(By.CSS_SELECTOR, selector)
                        for link in links:
                            url = link.get_attribute("href")
                            if url and ("/2022/" in url or "/2023/" in url or "/2024/" in url or "/2025/" in url):
                                if url not in article_urls:
                                    article_urls.append(url)
                    except Exception as e:
                        print(f"Error finding links with selector {selector}: {e}")
                
                print(f"Found {len(article_urls)} article URLs on page {page}")
                
                if not article_urls:
                    print("No article URLs found, ending search.")
                    break
                
                # Process each article URL
                for url in article_urls:
                    try:
                        self.extract_article_content(url)
                    except Exception as e:
                        print(f"Error processing article {url}: {e}")
                
                # Move to next page
                page += 1
                time.sleep(3)
                
            except Exception as e:
                print(f"Error on page {page}: {e}")
                break
    
    def run(self, start_date="2022-05-05", end_date="2022-05-12"):
        """Run the scraper for the specified date range"""
        try:
            print(f"Starting NPR China articles scraper from {start_date} to {end_date}")
            
            # Try first with the standard search approach
            try:
                self.scrape_search_results(start_date, end_date)
            except Exception as e:
                print(f"Standard search approach failed: {e}")
                print("Trying alternative approach...")
                self.scrape_with_direct_date_filter(start_date, end_date)
                
            print(f"Scraping completed. Data saved to {self.output_file}")
        finally:
            self.driver.quit()

if __name__ == "__main__":
    # Set up retry mechanism
    max_retries = 3
    
    for attempt in range(1, max_retries + 1):
        try:
            print(f"Attempt {attempt} of {max_retries}")
            
            # Initialize and run scraper
            # Using non-headless mode for debugging, change to True for production
            scraper = NPRScraper(headless=False)
            
            # Use the date range specified in the instructions
            scraper.run(start_date="2022-05-05", end_date="2025-05-12")
            
            # If successful, break the retry loop
            break
        except Exception as e:
            print(f"Attempt {attempt} failed with error: {e}")
            if attempt < max_retries:
                print(f"Retrying in 10 seconds...")
                time.sleep(10)
            else:
                print("All retry attempts failed. Please check your network connection and try again.")