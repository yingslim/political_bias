import json
import os
import time
import datetime
import random
import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException, StaleElementReferenceException
from selenium.webdriver.common.action_chains import ActionChains

class CloudflareSolver:
    def __init__(self, driver):
        self.driver = driver

    def detect(self):
        page = self.driver.page_source.lower()
        indicators = [
            "cloudflare",
            "checking your browser",
            "security check to access",
            "please wait while we verify",
            "one more step",
            "ray id:"
        ]
        return any(ind in page for ind in indicators)

    def solve(self, url, max_wait=120):
        """Loads the URL and waits for Cloudflare challenge to be solved manually."""
        domain = url.split('/')[2]
        homepage = f"https://{domain}"
        try:
            # initial visit to set cookies
            self.driver.get(homepage)
            time.sleep(random.uniform(3, 5))
            self.driver.get(url)
        except Exception as e:
            print(f"Error loading URL for Cloudflare solve: {e}")

        # If Cloudflare challenge appears, prompt user
        if self.detect():
            print("" + "-"*50)
            print("CLOUDFLARE CHALLENGE DETECTED! Please complete the challenge in the browser window.")
            print(f"The script will wait up to {max_wait} seconds for manual completion.")
            print("-"*50 + "")
        start = time.time()
        while time.time() - start < max_wait:
            if not self.detect():
                print("Cloudflare challenge appears to be solved!")
                time.sleep(2)
                return True
            time.sleep(5)
            print("Waiting for challenge to be solved...")
        print("Timeout waiting for Cloudflare challenge.")
        return False

class AxiosScraper:
    def __init__(self):
        ua = (
            f"Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            f"AppleWebKit/537.36 (KHTML, like Gecko) "
            f"Chrome/{random.randint(110,123)}.0.0.0 Safari/537.36"
        )
        options = uc.ChromeOptions()
        options.add_argument("--start-maximized")
        options.add_argument("--disable-notifications")
        options.add_argument("--disable-popup-blocking")
        options.add_argument("--disable-blink-features=AutomationControlled")
        options.add_argument(f"--user-agent={ua}")
        options.add_argument("--disable-extensions")
        options.add_argument("--disable-gpu")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")

        self.driver = uc.Chrome(options=options)
        self.driver.execute_cdp_cmd("Network.setUserAgentOverride", {"userAgent": ua})
        self.driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get:()=>undefined})")
        self.driver.set_page_load_timeout(60)

        self.solver = CloudflareSolver(self.driver)
        self.base_url = "https://www.axios.com/results?q=china&sort=1"
        self.output_file = "axios_china_articles.json"
        
        # Setting date range - use datetime objects for consistent comparison
        self.start_date = datetime.datetime(2022, 5, 5).date()
        self.end_date = datetime.datetime(2025, 1, 8).date()
        
        # Track seen URLs to avoid duplicates
        self.seen_urls = set()
        
    def parse_date(self, date_text):
        """Parse date strings with better error handling and debugging."""
        # Clean up the date text
        if date_text is None or not date_text.strip():
            print(f"Empty date text received")
            return datetime.datetime.now().date()
            
        date_text = date_text.strip()
        
        # If date has format like "May 9, 2025 - Politics & Policy"
        if " - " in date_text:
            date_text = date_text.split(" - ")[0].strip()
        
        print(f"Parsing date: '{date_text}'")
        
        # Try common date formats
        formats = [
            "%B %d, %Y",  # May 9, 2025
            "%b %d, %Y",  # May 9, 2025
            "%m/%d/%Y",   # 05/09/2025
            "%Y-%m-%d"    # 2025-05-09
        ]
        
        for fmt in formats:
            try:
                return datetime.datetime.strptime(date_text, fmt).date()
            except ValueError:
                continue
        
        # Handle relative dates
        today = datetime.datetime.now().date()
        rel_text = date_text.lower()
        
        if "hour" in rel_text:
            hours = int(''.join(filter(str.isdigit, rel_text)) or 1)
            return today
        elif "min" in rel_text:
            return today
        elif "day" in rel_text:
            days = int(''.join(filter(str.isdigit, rel_text)) or 1)
            return today - datetime.timedelta(days=days)
        
        # Last resort - try to extract year, month, day directly
        try:
            if "," in date_text and len(date_text) > 8:
                # Handle format like "May 9, 2025"
                month_day, year = date_text.split(",")
                month_str, day_str = month_day.strip().split(" ", 1)
                
                # Convert month name to number
                months = {
                    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
                    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12
                }
                month = months.get(month_str.lower()[:3], 1)
                
                day = int(''.join(filter(str.isdigit, day_str)) or 1)
                year = int(''.join(filter(str.isdigit, year)) or datetime.datetime.now().year)
                
                return datetime.date(year, month, day)
        except Exception as e:
            print(f"Failed to parse complex date format: {e}")
        
        print(f"Could not parse date: '{date_text}', using today's date")
        return today

    def human_like_scroll(self):
        """Scrolls down the page in a human-like manner."""
        scroll_amount = random.randint(300, 700)
        self.driver.execute_script(f"window.scrollBy(0, {scroll_amount});")
        time.sleep(random.uniform(1, 2.5))

    def extract_article_data(self, url):
        """Extracts article content from a given URL."""
        if url in self.seen_urls:
            print(f"URL already processed: {url}")
            return None
            
        self.seen_urls.add(url)
        
        # Open the article in a new tab
        original = self.driver.current_window_handle
        self.driver.execute_script("window.open(arguments[0], '_blank');", url)
        new_tab = [h for h in self.driver.window_handles if h != original][0]
        self.driver.switch_to.window(new_tab)
        time.sleep(random.uniform(3, 5))

        # Handle Cloudflare if detected
        if self.solver.detect():
            if not self.solver.solve(url):
                self.driver.close()
                self.driver.switch_to.window(original)
                return None

        article_data = {"url": url}
        
        try:
            # Extract title
            article_data["title"] = WebDriverWait(self.driver, 15).until(
                EC.presence_of_element_located((By.TAG_NAME, "h1"))
            ).text
            
            # Extract author
            try:
                article_data["author"] = self.driver.find_element(By.CSS_SELECTOR, "[data-cy='byline-author']").text
            except NoSuchElementException:
                article_data["author"] = "Unknown"
            
            # Extract date
            try:
                # Try multiple selectors for the date
                date_selectors = [
                    "[data-cy='time-rubric']", 
                    "time", 
                    ".time-rubric",
                    ".flex-wrap.items-center time",
                    "[data-cy='time']"
                ]
                
                date_text = None
                for selector in date_selectors:
                    try:
                        date_text = self.driver.find_element(By.CSS_SELECTOR, selector).text
                        if date_text:
                            break
                    except:
                        continue
                        
                # Try alternate method using page URL for date
                if not date_text and "/2025/" in url:
                    # Extract date from URL like /2025/05/13/
                    parts = url.split("/")
                    for i, part in enumerate(parts):
                        if part == "2025" and i+2 < len(parts):
                            try:
                                month = int(parts[i+1])
                                day = int(parts[i+2].split("-")[0])
                                date_obj = datetime.date(2025, month, day)
                                article_data["date"] = str(date_obj)
                                print(f"Date extracted from URL: {article_data['date']}")
                            except:
                                pass
                else:
                    article_date = self.parse_date(date_text)
                    article_data["date"] = str(article_date)
                    print(f"Date parsed from text: {article_data['date']}")
            except Exception as e:
                print(f"Date parsing error: {e}")
                article_data["date"] = str(datetime.datetime.now().date())
            
            # Extract content
            content = []
            try:
                # First try to get main content container
                containers = self.driver.find_elements(By.TAG_NAME, "main")
                if not containers:
                    containers = self.driver.find_elements(By.CSS_SELECTOR, "article")
                
                if containers:
                    paras = containers[0].find_elements(By.TAG_NAME, "p")
                    for p in paras:
                        text = p.text.strip()
                        if text and text != "»":
                            content.append(text)
            except Exception as e:
                print(f"Content extraction error: {e}")
                
            article_data["content"] = "\n".join(content)
            
        except Exception as e:
            print(f"Article extraction error: {e}")
        
        # Close tab and switch back
        self.driver.close()
        self.driver.switch_to.window(original)
        
        return article_data

    def save_article(self, data):
        """Saves article data to JSON file."""
        if not data:
            return
            
        # Get existing articles
        articles = []
        if os.path.exists(self.output_file):
            try:
                with open(self.output_file, 'r', encoding='utf-8') as f:
                    articles = json.load(f)
            except:
                articles = []
        
        # Check if URL already exists
        if any(a['url'] == data['url'] for a in articles):
            print(f"Skipping duplicate in file: {data['url']}")
            return
            
        # Add article and save
        articles.append(data)
        with open(self.output_file, 'w', encoding='utf-8') as f:
            json.dump(articles, f, ensure_ascii=False, indent=2)
        print(f"Saved article: {data['title']} ({data['url']})")

    def click_show_more(self):
        """Improved method to click 'Show more results' button."""
        try:
            # First let's take a screenshot for debugging
            self.driver.save_screenshot("before_click_debug.png")
            print("Took screenshot for debugging")
            
            # Print the HTML of the button area to debug
            try:
                button_area = self.driver.find_element(By.XPATH, "//*[contains(text(), 'Show') and contains(text(), 'more')]/..")
                print(f"Button area HTML: {button_area.get_attribute('outerHTML')}")
            except:
                print("Could not find button area for HTML inspection")
            
            # Scroll to the bottom of the page
            self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(3)
            
            # Try multiple approaches to find and click the button
            
            # 1. Direct JavaScript targeting button by text content
            button_js = """
            return Array.from(document.querySelectorAll('button')).find(
                button => button.textContent.includes('Show') && 
                button.textContent.includes('more') && 
                button.textContent.includes('results')
            );
            """
            button = self.driver.execute_script(button_js)
            if button:
                print("Found button via JavaScript text content search")
                self.driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", button)
                time.sleep(2)
                # First try normal click
                try:
                    button.click()
                    print("Clicked button normally")
                    time.sleep(3)
                    return True
                except:
                    # Then try JS click
                    self.driver.execute_script("arguments[0].click();", button)
                    print("Clicked button with JavaScript")
                    time.sleep(3)
                    return True
            
            # 2. Use an Action Chain for the click
            button_selectors = [
                "button.inline-flex", 
                "button[type='button']",
                "button:not([disabled])"
            ]
            
            for selector in button_selectors:
                buttons = self.driver.find_elements(By.CSS_SELECTOR, selector)
                for btn in buttons:
                    try:
                        if "show" in btn.text.lower() and "more" in btn.text.lower():
                            print(f"Found button via selector: {selector}")
                            # Use ActionChains for a more robust click
                            actions = ActionChains(self.driver)
                            actions.move_to_element(btn).pause(1).click().perform()
                            print("Clicked button with ActionChains")
                            time.sleep(3)
                            return True
                    except:
                        continue
            
            # 3. Try finding by specific text inside any element
            for text_pattern in ["Show 10 more results", "Show more results", "more results"]:
                try:
                    xpath = f"//*[contains(text(), '{text_pattern}')]"
                    elements = self.driver.find_elements(By.XPATH, xpath)
                    for elem in elements:
                        if elem.is_displayed():
                            print(f"Found clickable element with text: '{text_pattern}'")
                            # Highlight element for debugging
                            self.driver.execute_script(
                                "arguments[0].style.border='3px solid red'", elem
                            )
                            time.sleep(1)
                            
                            # Try clicking the element's parent if it's not directly clickable
                            try:
                                parent = elem.find_element(By.XPATH, "./..")
                                actions = ActionChains(self.driver)
                                actions.move_to_element(parent).pause(1).click().perform()
                                print("Clicked parent element with ActionChains")
                                time.sleep(3)
                                return True
                            except:
                                # Try direct click with different methods
                                try:
                                    elem.click()
                                    print("Clicked element directly")
                                    time.sleep(3)
                                    return True
                                except:
                                    self.driver.execute_script("arguments[0].click();", elem)
                                    print("Clicked element with JavaScript")
                                    time.sleep(3)
                                    return True
                except Exception as e:
                    print(f"Error with text pattern '{text_pattern}': {e}")
            
            # 4. Try clicking using coordinates of the button
            # This is from the screenshot - span shows coordinates: 172.59 × 18
            try:
                # Try to find the coordinates shown in your screenshot
                self.driver.execute_script("""
                    const simulateClick = function(x, y) {
                        const event = new MouseEvent('click', {
                            view: window,
                            bubbles: true,
                            cancelable: true,
                            clientX: x,
                            clientY: y
                        });
                        document.elementFromPoint(x, y).dispatchEvent(event);
                    };
                    // Coordinates from the screenshot
                    simulateClick(172, 18);
                """)
                print("Performed coordinate-based click at (172, 18)")
                time.sleep(3)
                
                # Also try with slightly different coordinates
                self.driver.execute_script("simulateClick(206, 40);")
                print("Performed coordinate-based click at (206, 40)")
                time.sleep(3)
                return True
            except Exception as e:
                print(f"Coordinate click error: {e}")
                
            # 5. Last resort: Try to find the button in a specific region of the page
            button_region_js = """
            // Get elements in the lower middle part of the page
            const viewportHeight = window.innerHeight;
            const lowerThird = viewportHeight * 0.66;
            const elements = document.elementsFromPoint(
                window.innerWidth / 2, 
                lowerThird
            );
            
            // Look for something button-like
            for (const elem of elements) {
                if (
                    (elem.tagName === 'BUTTON' || 
                     elem.tagName === 'A' || 
                     elem.role === 'button' ||
                     elem.className.includes('button')) && 
                    elem.textContent.toLowerCase().includes('more')
                ) {
                    return elem;
                }
            }
            return null;
            """
            possible_button = self.driver.execute_script(button_region_js)
            if possible_button:
                print("Found potential button in the expected region")
                try:
                    # Highlight it
                    self.driver.execute_script(
                        "arguments[0].style.border='5px solid blue'", possible_button
                    )
                    time.sleep(1)
                    self.driver.execute_script("arguments[0].click();", possible_button)
                    print("Clicked regional element with JavaScript")
                    time.sleep(3)
                    return True
                except Exception as e:
                    print(f"Regional click error: {e}")
            
            # Take an "after" screenshot
            self.driver.save_screenshot("after_click_attempts_debug.png")
            print("Took 'after' screenshot for debugging")
            
            print("All button click attempts failed")
            return False
            
        except Exception as e:
            print(f"Error in click_show_more: {e}")
            return False

    def get_article_links(self):
        """Extract article links on the current page with improved filtering."""
        links = []
        
        # Possible selectors for article links
        link_selectors = [
            "a[href*='/2025/']",  # Links containing /2025/ (most likely articles)
            "a[href*='/2024/']",  # Links containing /2024/ (older articles)
            "a.leading-none[href*='/']",  # Class-based selector seen in your screenshots
            "a[href*='/20']"  # More general URL pattern for articles
        ]
        
        for selector in link_selectors:
            try:
                elements = self.driver.find_elements(By.CSS_SELECTOR, selector)
                for element in elements:
                    try:
                        url = element.get_attribute("href")
                        if url and "/results?" not in url and "/tag/" not in url:
                            links.append({"element": element, "url": url})
                    except StaleElementReferenceException:
                        continue
            except Exception as e:
                print(f"Error finding links with selector '{selector}': {e}")
        
        # Filter to likely article links
        article_links = []
        for link in links:
            url = link["url"]
            # Check if it looks like an article URL with date format
            if "/2025/" in url or "/2024/" in url or "/2023/" in url or "/2022/" in url:
                # Additional article pattern matching
                if len(url.split('/')) >= 5 and "-" in url.split('/')[-1]:
                    article_links.append(link)
        
        print(f"Found {len(article_links)} potential article links")
        return article_links

    def scrape_search_results(self):
        """Main scraping function with improved pagination handling."""
        # Load search results page
        self.driver.get(self.base_url)
        time.sleep(5)
        self.solver.solve(self.base_url)
        
        page_num = 1
        article_count = 0
        load_more_failed_count = 0
        max_pages = 20  # Limit to prevent infinite loops
        
        while page_num <= max_pages:
            print(f"\nProcessing page {page_num}")
            
            # Scroll down a bit to load content
            for _ in range(3):
                self.human_like_scroll()
            
            # Get article links on current page
            article_links = self.get_article_links()
            
            # Process each article
            for link in article_links:
                if link["url"] in self.seen_urls:
                    continue
                    
                print(f"Scraping: {link['url']}")
                article_data = self.extract_article_data(link["url"])
                
                if article_data:
                    # Parse date for comparison
                    try:
                        article_date = datetime.datetime.fromisoformat(article_data['date']).date()
                    except:
                        try:
                            article_date = self.parse_date(article_data['date'])
                        except:
                            print(f"Failed to parse date for comparison: {article_data['date']}")
                            continue
                    
                    print(f"Article date: {article_date}, Range: {self.start_date} to {self.end_date}")
                    
                    # Check date range
                    if self.start_date <= article_date <= self.end_date:
                        self.save_article(article_data)
                        article_count += 1
                    elif article_date < self.start_date:
                        print(f"Found article older than {self.start_date}: {link['url']}")
                        # Articles are usually in chronological order, so we could stop here
                        # But we'll continue to ensure we don't miss any that match our criteria
                    else:
                        print(f"Skipped (date out of range): {link['url']}")
            
            print(f"Progress: {article_count} articles scraped")
            
            # Try to click "Show more results" with enhanced method
            if not self.click_show_more():
                load_more_failed_count += 1
                if load_more_failed_count >= 3:
                    print("Failed to load more results multiple times. Stopping.")
                    break
                # Try one more approach - reload the page and go directly to next page
                if load_more_failed_count == 2:
                    try:
                        # Construct a URL with offset parameter
                        offset = page_num * 10
                        next_page_url = f"{self.base_url}&offset={offset}"
                        print(f"Trying direct URL navigation to page {page_num + 1}: {next_page_url}")
                        self.driver.get(next_page_url)
                        time.sleep(5)
                    except Exception as e:
                        print(f"Direct navigation error: {e}")
            else:
                # Reset failure counter on successful click
                load_more_failed_count = 0
            
            # Wait for new content to load
            time.sleep(random.uniform(5, 8))
            page_num += 1
        
        print(f"\nScraping completed. Total articles scraped: {article_count}")

    def run(self):
        """Main execution function."""
        try:
            self.scrape_search_results()
        except Exception as e:
            import traceback
            print(f"Unhandled exception: {e}")
            print(traceback.format_exc())
        finally:
            self.driver.quit()

if __name__ == "__main__":
    scraper = AxiosScraper()
    scraper.run()