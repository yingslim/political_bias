import time
import json
import datetime
import re
from selenium import webdriver
import undetected_chromedriver as uc
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException, StaleElementReferenceException
from webdriver_manager.chrome import ChromeDriverManager


class FoxNewsChinaScraper:
    def __init__(self):
        self.articles = []
        self.base_url = "https://www.foxnews.com/search-results/search#q=china"
        self.start_date = datetime.datetime(2022, 5, 7)
        self.end_date = datetime.datetime(2025, 5, 7)
        self.setup_driver()

    def setup_driver(self):
        """Set up the Chrome webdriver with appropriate options."""
        chrome_options = uc.ChromeOptions()
        # Uncomment the line below if you want to run in headless mode
        # chrome_options.add_argument("--headless")
        chrome_options.add_argument("--window-size=1920,1080")
        chrome_options.add_argument("--disable-notifications")
        chrome_options.add_argument("--disable-popup-blocking")
        chrome_options.add_argument("--disable-infobars")
        
        self.driver =  uc.Chrome(options=chrome_options)
        
    def parse_date(self, date_str):
        """Parse date string into datetime object with enhanced handling."""
        if not date_str or date_str.lower() == "date unavailable":
            return datetime.datetime.now()  # Default to current date if unavailable
            
        # Handle relative dates
        if 'ago' in date_str.lower():
            return datetime.datetime.now()  # Consider recent articles in range
            
        # Handle "Month Day, Year" format (e.g., "May 7, 2025 2:27pm EDT")
        month_day_year_match = re.search(r'(\w+)\s+(\d{1,2}),?\s+(\d{4})', date_str)
        if month_day_year_match:
            month, day, year = month_day_year_match.groups()
            try:
                date_obj = datetime.datetime.strptime(f"{month} {day} {year}", "%B %d %Y")
                return date_obj
            except ValueError:
                try:
                    date_obj = datetime.datetime.strptime(f"{month} {day} {year}", "%b %d %Y")
                    return date_obj
                except ValueError:
                    pass
                    
        # Try standard date formats
        date_formats = [
            "%B %d, %Y",       # May 10, 2025
            "%b %d, %Y",       # May 10, 2025
            "%Y-%m-%d",        # 2025-05-10
            "%m/%d/%Y",        # 05/10/2025
            "%B %d, %Y %I:%M%p %Z",  # May 7, 2025 2:27pm EDT
            "%b %d, %Y %I:%M%p %Z"   # May 7, 2025 2:27pm EDT
        ]
        
        for fmt in date_formats:
            try:
                date_str_clean = re.sub(r'\s+', ' ', date_str).strip()
                return datetime.datetime.strptime(date_str_clean, fmt)
            except ValueError:
                continue
                
        # Try extracting just the date part
        date_match = re.search(r'(\w+ \d{1,2}, \d{4})', date_str)
        if date_match:
            try:
                return datetime.datetime.strptime(date_match.group(1), "%B %d, %Y")
            except ValueError:
                try:
                    return datetime.datetime.strptime(date_match.group(1), "%b %d, %Y")
                except ValueError:
                    pass
        
        print(f"Couldn't parse date: {date_str} - using current date")
        return datetime.datetime.now()
            
    def is_date_in_range(self, date_str):
        """Check if the article date is within our target range."""
        try:
            article_date = self.parse_date(date_str)
            return self.start_date <= article_date <= self.end_date
        except Exception as e:
            print(f"Error processing date: {e}")
            return True  # Include by default if date processing fails
            
    def extract_article_content(self, url):
        """Navigate to article page and extract the main content with improved selectors."""
        try:
            # Wait for article content to load
            WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.TAG_NAME, "article"))
            )
            
            # Try multiple selectors for article content
            content_selectors = [
                "article.article-wrap p",
                ".article-body p",
                ".article-content p",
                "article p",
                ".main-content p"
            ]
            
            for selector in content_selectors:
                try:
                    paragraphs = self.driver.find_elements(By.CSS_SELECTOR, selector)
                    if paragraphs:
                        content = " ".join([p.text for p in paragraphs if p.text.strip()])
                        if content:
                            return content
                except NoSuchElementException:
                    continue
                    
            # If we couldn't get paragraphs, try getting the whole article content
            for selector in [".article-body", ".article-content", "article", ".main-content"]:
                try:
                    content_div = self.driver.find_element(By.CSS_SELECTOR, selector)
                    content = content_div.text
                    if content:
                        return content
                except NoSuchElementException:
                    continue
                    
            return "Content not available"
                    
        except Exception as e:
            print(f"Error extracting content: {e}")
            return "Error extracting content"

    def extract_author(self):
        """Extract author information from article page with improved selectors."""
        author_selectors = [
            ".author-byline",
            ".author-name",
            ".author",
            ".article-meta .author-byline",
            "[class*='author']",
            ".article-info .author"
        ]
        
        for selector in author_selectors:
            try:
                author_element = self.driver.find_element(By.CSS_SELECTOR, selector)
                author_text = author_element.text.strip()
                if author_text:
                    # Clean up author text (remove "By " prefix if present)
                    return author_text.replace("By ", "").strip()
            except NoSuchElementException:
                continue
                
        # Try finding a Fox News author
        try:
            author_element = self.driver.find_element(By.CSS_SELECTOR, "a[href*='person/']")
            return author_element.text.strip()
        except NoSuchElementException:
            pass
            
        return "Fox News"  # Default attribution if no author found

    def extract_date_from_page(self):
        """Extract date from article page with improved selectors."""
        date_selectors = [
            ".article-date time",
            "time.date",
            "span.article-date time",
            ".publish-date",
            ".meta-datetime",
            ".article-meta time",
            ".article-date",
            "time[datetime]",
            "[class*='date']"
        ]
        
        for selector in date_selectors:
            try:
                date_element = self.driver.find_element(By.CSS_SELECTOR, selector)
                
                # First try to get datetime attribute
                date_text = date_element.get_attribute("datetime")
                if date_text:
                    try:
                        date_obj = datetime.datetime.fromisoformat(date_text.replace("Z", "+00:00"))
                        return date_obj.strftime("%B %d, %Y")
                    except ValueError:
                        pass
                
                # Then try text content
                date_text = date_element.text.strip()
                if date_text and date_text.lower() != "published":
                    return date_text
            except NoSuchElementException:
                continue
                
        # If we can't find a date, look for a published string with date
        try:
            published_text = self.driver.find_element(By.XPATH, "//*[contains(text(), 'Published')]").text
            date_match = re.search(r'Published\s+(.+)', published_text)
            if date_match:
                return date_match.group(1).strip()
        except NoSuchElementException:
            pass
            
        return "Date unavailable"

    def process_article(self, article_element):
        """Process an article element and extract relevant data with improved handling."""
        try:
            # Extract title with multiple potential selectors
            title = None
            title_selectors = [
                "h2.title a", "h2.headline a", "h3.title a", 
                "h1", "h2", ".title", ".headline",
                "header h1", ".info h1"
            ]
            
            for selector in title_selectors:
                try:
                    title_element = article_element.find_element(By.CSS_SELECTOR, selector)
                    title = title_element.text.strip()
                    if title:
                        break
                except NoSuchElementException:
                    continue
                    
            if not title:
                print("Could not find title element")
                return None
                
            # Get the URL - try different approaches
            url = None
            try:
                # First try to find a link in the title
                for selector in ["h2.title a", "h2.headline a", "h3.title a", ".title a", "a"]:
                    try:
                        url_element = article_element.find_element(By.CSS_SELECTOR, selector)
                        url = url_element.get_attribute("href")
                        if url:
                            break
                    except NoSuchElementException:
                        continue
                        
                # If still no URL, try to extract from article element
                if not url:
                    url = article_element.get_attribute("data-url")
                
                # If still no URL, the current URL might be the article page
                if not url:
                    url = self.driver.current_url
                    
            except Exception as e:
                print(f"Error extracting URL: {e}")
                return None
                
            if not url:
                print("Could not find URL")
                return None
                
            # Ensure URL is absolute
            if not url.startswith("http"):
                url = f"https://www.foxnews.com{url}"
                
            # Extract date from search result
            date_text = "Date unavailable"
            date_selectors = [
                ".date", ".publish-date", "time", "[class*='date']",
                ".meta time", ".eyebrow time", ".meta-data time"
            ]
            
            for selector in date_selectors:
                try:
                    date_element = article_element.find_element(By.CSS_SELECTOR, selector)
                    date_content = date_element.text.strip()
                    if date_content:
                        date_text = date_content
                        break
                        
                    # If no text, try datetime attribute
                    datetime_attr = date_element.get_attribute("datetime")
                    if datetime_attr:
                        try:
                            date_obj = datetime.datetime.fromisoformat(datetime_attr.replace("Z", "+00:00"))
                            date_text = date_obj.strftime("%B %d, %Y")
                            break
                        except ValueError:
                            pass
                except NoSuchElementException:
                    continue
                    
            # Check for video content before navigation
            is_video = False
            try:
                video_selectors = [
                    ".video-ct", ".video-container", ".clip", "[class*='video']", 
                    ".featured-video", ".play-button", ".play-icon"
                ]
                for selector in video_selectors:
                    if article_element.find_elements(By.CSS_SELECTOR, selector):
                        is_video = True
                        break
                        
                # Also check for video-related classes on the article itself
                article_class = article_element.get_attribute("class")
                if article_class and any(term in article_class for term in ["video", "clip", "featured-video"]):
                    is_video = True
            except Exception:
                pass
                
            # Navigate to the article/video page to get more details
            # Open in new tab to keep search results page
            self.driver.execute_script("window.open('');")
            self.driver.switch_to.window(self.driver.window_handles[1])
            self.driver.get(url)
            time.sleep(3)  # Wait for page to load
            
            # Extract detailed date from page
            detailed_date = self.extract_date_from_page()
            if detailed_date and detailed_date != "Date unavailable":
                date_text = detailed_date
            
            # Check if date is in our target range
            if not self.is_date_in_range(date_text):
                print(f"Skipping article from {date_text} - outside date range")
                self.driver.close()
                self.driver.switch_to.window(self.driver.window_handles[0])
                return None
                
            # Extract content
            if is_video:
                # For videos, get title and description
                video_title = title
                video_description = ""
                
                # Try to get video description
                desc_selectors = [".info p", ".dek", ".video-info p", ".video-description", ".clip-info p"]
                for selector in desc_selectors:
                    try:
                        desc_element = self.driver.find_element(By.CSS_SELECTOR, selector)
                        video_description = desc_element.text.strip()
                        if video_description:
                            break
                    except NoSuchElementException:
                        continue
                
                # Use title as content if no description
                content = video_description if video_description else video_title
                content_type = "video"
            else:
                content = self.extract_article_content(url)
                content_type = "article"
                
            # Extract author
            author = self.extract_author()
            
            # Close the tab and switch back to results
            self.driver.close()
            self.driver.switch_to.window(self.driver.window_handles[0])
            
            article_data = {
                "title": title,
                "date": date_text,
                "content": content,
                "url": url,
                "author": author,
                "type": content_type
            }
            
            print(f"Extracted: {title} ({date_text})")
            return article_data
            
        except Exception as e:
            print(f"Error processing article: {e}")
            # If we opened a new tab, close it and switch back
            if len(self.driver.window_handles) > 1:
                self.driver.close()
                self.driver.switch_to.window(self.driver.window_handles[0])
            return None

    def click_load_more(self):
        """Click the 'Load More' button to reveal more articles."""
        try:
            # Scroll down to make the load more button visible
            self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(1)
            
            # Wait for the Load More button using multiple potential selectors
            load_more_selectors = [
                ".button.load-more", 
                "a.load-more", 
                "button.load-more", 
                "[class*='load-more']",
                ".pagination-load-more",
                ".search-load-more"
            ]
            
            for selector in load_more_selectors:
                try:
                    load_more_button = WebDriverWait(self.driver, 5).until(
                        EC.element_to_be_clickable((By.CSS_SELECTOR, selector))
                    )
                    self.driver.execute_script("arguments[0].scrollIntoView(true);", load_more_button)
                    time.sleep(1)
                    load_more_button.click()
                    time.sleep(3)  # Wait for content to load
                    return True
                except (TimeoutException, NoSuchElementException, StaleElementReferenceException):
                    continue
                    
            print("No more 'Load More' button found")
            return False
            
        except Exception as e:
            print(f"Error clicking load more: {e}")
            return False

    def scrape_articles(self, max_articles=100):
        """Scrape articles from the search results page with improved handling."""
        self.driver.get(self.base_url)
        time.sleep(3)  # Wait for page to load
        
        # Accept any cookies/notifications if present
        try:
            cookie_button_selectors = [
                "button[contains(text(), 'Accept')]",
                "button[contains(text(), 'allow')]",
                "button[contains(text(), 'I agree')]",
                "button[contains(text(), 'Continue')]",
                ".cookie-banner button",
                ".gdpr-banner button",
                "[id*='cookie'] button",
                "[class*='consent'] button"
            ]
            
            for selector in cookie_button_selectors:
                try:
                    accept_button = WebDriverWait(self.driver, 3).until(
                        EC.element_to_be_clickable((By.XPATH, selector))
                    )
                    accept_button.click()
                    time.sleep(1)
                    break
                except TimeoutException:
                    continue
        except Exception:
            pass
            
        article_count = 0
        load_more_count = 0
        max_load_more = 20  # Maximum number of times to click "Load More"
        processed_urls = set()  # Track URLs we've already processed
        
        while article_count < max_articles and load_more_count < max_load_more:
            # Find all article elements with multiple potential selectors
            article_selectors = [
                "article.article", 
                ".search-results-content article",
                ".search-results article",
                ".content article",
                ".results article",
                ".article-list article",
                ".article"
            ]
            
            article_elements = []
            for selector in article_selectors:
                try:
                    elements = self.driver.find_elements(By.CSS_SELECTOR, selector)
                    if elements:
                        article_elements = elements
                        break
                except Exception:
                    continue
                    
            if not article_elements:
                print("No article elements found")
                break
                
            # Process each article
            for article_element in article_elements:
                try:
                    # Check if we already processed this element
                    element_id = article_element.id
                    if element_id in processed_urls:
                        continue
                        
                    processed_urls.add(element_id)
                    
                    article_data = self.process_article(article_element)
                    if article_data:
                        self.articles.append(article_data)
                        article_count += 1
                        print(f"Articles collected: {article_count}/{max_articles}")
                        
                        if article_count >= max_articles:
                            break
                except StaleElementReferenceException:
                    continue
                except Exception as e:
                    print(f"Error processing article element: {e}")
                    continue
            
            if article_count >= max_articles:
                break
                
            # Try to load more articles
            if not self.click_load_more():
                print("No more articles to load")
                break
                
            load_more_count += 1
                
        print(f"Scraped {len(self.articles)} articles")

    def direct_scrape_url(self, url):
        """Directly scrape a single article URL."""
        try:
            self.driver.get(url)
            time.sleep(3)  # Wait for page to load
            
            # Create a dummy article element
            article_element = self.driver.find_element(By.TAG_NAME, "article")
            
            # Get title from page
            title = self.driver.title.replace(" | Fox News", "").strip()
            
            # Get date
            date_text = self.extract_date_from_page()
            
            # Check if video
            is_video = False
            try:
                video_elements = self.driver.find_elements(By.CSS_SELECTOR, 
                    ".video-container, .video-player, [class*='video-'], .featured-video"
                )
                if video_elements:
                    is_video = True
            except NoSuchElementException:
                pass
                
            # Get content
            if is_video:
                # For videos, use title
                video_description = ""
                desc_selectors = [".info p", ".dek", ".video-info p", ".video-description", ".clip-info p"]
                for selector in desc_selectors:
                    try:
                        desc_element = self.driver.find_element(By.CSS_SELECTOR, selector)
                        video_description = desc_element.text.strip()
                        if video_description:
                            break
                    except NoSuchElementException:
                        continue
                
                content = video_description if video_description else title
                content_type = "video"
            else:
                content = self.extract_article_content(url)
                content_type = "article"
                
            # Get author
            author = self.extract_author()
            
            article_data = {
                "title": title,
                "date": date_text,
                "content": content,
                "url": url,
                "author": author,
                "type": content_type
            }
            
            print(f"Directly extracted: {title} ({date_text})")
            self.articles.append(article_data)
            
        except Exception as e:
            print(f"Error directly scraping URL {url}: {e}")
            
    def scrape_from_screenshots(self):
        """Scrape articles from the provided screenshots for testing."""
        sample_urls = [
            "https://www.foxnews.com/politics/trump-says-80-tariff-china-seems-right-weekend-talks-beijing",
            "https://www.foxnews.com/politics/us-china-trade-talks-begin-switzerland-trump-pushes-fair-deal",
            "https://www.foxnews.com/politics/pivotal-trade-talks-beijing-loom-trump-swears-new-us-ambassador-china-what-timing"
        ]
        
        for url in sample_urls:
            self.direct_scrape_url(url)

    def save_to_json(self, filename="fox_news_china_articles.json"):
        """Save scraped articles to a JSON file."""
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.articles, f, ensure_ascii=False, indent=4)
        print(f"Saved {len(self.articles)} articles to {filename}")
        
    def close(self):
        """Close the webdriver."""
        self.driver.quit()

    def run(self, max_articles=100, test_mode=False):
        """Run the complete scraping process."""
        try:
            if test_mode:
                # Use the screenshot sample URLs for testing
                self.scrape_from_screenshots()
            else:
                # Normal scraping operation
                self.scrape_articles(max_articles)
                
            self.save_to_json()
        finally:
            self.close()


if __name__ == "__main__":
    scraper = FoxNewsChinaScraper()
    
    # Set test_mode=True to scrape the sample URLs from screenshots
    # Set test_mode=False to perform normal search scraping
    scraper.run(max_articles=20000, test_mode=False)