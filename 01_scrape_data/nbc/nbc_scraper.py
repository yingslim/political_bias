import undetected_chromedriver as uc
from selenium.webdriver.chrome.options import Options
import logging

# Suppress selenium logging
logging.getLogger('selenium').setLevel(logging.WARNING)
logging.getLogger('urllib3').setLevel(logging.WARNING)
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException, ElementClickInterceptedException
import pandas as pd
import time
import datetime
import re
import os
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.keys import Keys

class NBCNewsScraper:
    def __init__(self, headless=True, filename="nbc_news_china_articles.json"):
        """Initialize the NBC News scraper with browser settings"""
        chrome_options = uc.ChromeOptions()
        if headless:
            chrome_options.add_argument("--headless")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--window-size=1920,1080")
        chrome_options.add_argument("--log-level=3")  # Suppress console messages
        chrome_options.add_argument("--disable-logging")  # Disable logging
        chrome_options.add_argument("--disable-extensions")
        chrome_options.add_argument("--disable-gpu")
        
        # Initialize the WebDriver with undetected_chromedriver
        self.driver = uc.Chrome(options=chrome_options)
        self.wait = WebDriverWait(self.driver, 10)
        
        # Create a dataframe to store the results
        self.results = pd.DataFrame(columns=['title', 'date', 'author', 'url', 'content'])
        
        # Keep track of URLs we've already scraped to avoid duplicates
        self.scraped_urls = set()
        
        # Filename for incremental saving
        self.filename = filename
        # Load existing results if present
        self.load_existing_results(self.filename)
    
    def __del__(self):
        """Close the browser when the object is deleted"""
        if hasattr(self, 'driver'):
            self.driver.quit()
    
    def handle_cookie_consent(self):
        """Handle cookie consent banners and overlays"""
        try:
            # First try to accept cookies by clicking the Accept button
            accept_buttons = self.driver.find_elements(By.XPATH, 
                "//button[contains(text(), 'Accept') or contains(text(), 'agree') or contains(@id, 'accept') or contains(@id, 'consent')]")
            for button in accept_buttons:
                if button.is_displayed():
                    button.click()
                    time.sleep(1)
                    return True
            
            # Look for onetrust banner specifically
            onetrust_elements = self.driver.find_elements(By.ID, "onetrust-banner-sdk")
            if onetrust_elements and onetrust_elements[0].is_displayed():
                accept_buttons = self.driver.find_elements(By.ID, "onetrust-accept-btn-handler")
                if accept_buttons:
                    accept_buttons[0].click()
                    time.sleep(1)
                    return True
            
            # Try closing dialog by id
            close_buttons = self.driver.find_elements(By.XPATH, "//button[contains(@class, 'close') or contains(@id, 'close')]")
            for button in close_buttons:
                if button.is_displayed():
                    button.click()
                    time.sleep(1)
                    return True
            
        except Exception as e:
            print(f"Error handling cookie consent: {e}")
        
        return False
    
    def search_nbc_news(self, query="china", start_date=None, end_date=None, max_pages=10):
        """Search NBC News for articles with the given query between start_date and end_date"""
        # NBC News search URL
        search_url = f"https://www.nbcnews.com/search/?q={query}"
        self.driver.get(search_url)
        
        # Handle any cookie consent pop-ups
        self.handle_cookie_consent()
        
        # Wait for the search results to load
        try:
            self.wait.until(EC.presence_of_element_located((By.CLASS_NAME, "queryly_item_row")))
        except TimeoutException:
            print("Timeout waiting for search results to load")
            return

        # Parse dates if provided
        self.start_date = None
        if start_date:
            if isinstance(start_date, str):
                self.start_date = datetime.datetime.strptime(start_date, "%Y-%m-%d")
            else:
                self.start_date = start_date
                
        self.end_date = None
        if end_date:
            if isinstance(end_date, str):
                self.end_date = datetime.datetime.strptime(end_date, "%Y-%m-%d")
            else:
                self.end_date = end_date
        
        # Start collecting articles
        print("=== Starting article collection ===")
        self._debug_pagination_structure()
        self._process_page()
        
        # Move through pagination until we find articles older than start_date or no more pages
        page_count = 1
        while page_count < max_pages:
            try:
                print(f"\n=== Attempting to navigate to page {page_count + 1} ===")
                
                # Try multiple navigation methods
                navigation_success = False
                
                # Method 1: Find and click next button
                next_button = self._find_next_button()
                if next_button:
                    print("Found next button, attempting click...")
                    if self._click_next_button_safe():
                        navigation_success = True
                        print("✓ Button click navigation successful")
                    else:
                        print("× Button click failed, trying arrow key method...")
                
                # Method 2: Use arrow keys if button click failed
                if not navigation_success:
                    print("Trying arrow key navigation...")
                    if self._navigate_with_arrows():
                        navigation_success = True
                        print("✓ Arrow key navigation successful")
                    else:
                        print("× Arrow key navigation failed")
                
                # Method 3: Try keyboard navigation on the page
                if not navigation_success:
                    print("Trying Page Down + Enter navigation...")
                    if self._navigate_with_keyboard():
                        navigation_success = True
                        print("✓ Keyboard navigation successful")
                
                if not navigation_success:
                    print("❌ All navigation methods failed, ending pagination")
                    break
                
                # Wait for new content to load
                print("Waiting for new page content...")
                time.sleep(3)
                
                # Wait for new search results to load
                try:
                    self.wait.until(EC.presence_of_element_located((By.CLASS_NAME, "queryly_item_row")))
                    print("✓ New content loaded successfully")
                except TimeoutException:
                    print("❌ Timeout waiting for new page to load")
                    break
                
                # Process the new page
                print(f"Processing page {page_count + 1}...")
                if not self._process_page():
                    print("Reached articles outside date range, stopping pagination")
                    break
                    
                page_count += 1
                print(f"✓ Successfully processed page {page_count}")
                
            except (NoSuchElementException, TimeoutException) as e:
                print(f"Error with pagination: {e}")
                break
                
        print(f"Scraped {page_count} pages with {len(self.results)} articles")
        return self.results

    def _find_next_button(self):
        """Find the next button using multiple strategies"""
        print("Looking for next button...")
        
        # Debug: Print current pagination structure
        try:
            pagination_debug = self.driver.execute_script("""
                const paginationElements = Array.from(document.querySelectorAll('a')).filter(a => 
                    a.textContent.toLowerCase().includes('next') || 
                    a.textContent.toLowerCase().includes('page') ||
                    a.className.toLowerCase().includes('next') ||
                    a.className.toLowerCase().includes('page')
                );
                return paginationElements.map(el => ({
                    text: el.textContent.trim(),
                    href: el.href,
                    className: el.className,
                    visible: el.offsetParent !== null
                }));
            """)
            print(f"Pagination debug info: {pagination_debug}")
        except:
            pass
        
        # Strategy 1: Look for exact "Next Page" text (case insensitive)
        try:
            next_buttons = self.driver.find_elements(By.XPATH, "//a[translate(normalize-space(text()), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz')='next page']")
            if next_buttons:
                for button in next_buttons:
                    if button.is_displayed():
                        print(f"Found 'Next Page' button: {button.text}")
                        return button
        except Exception as e:
            print(f"Strategy 1 failed: {e}")
        
        # Strategy 2: Look for "Next Page" anywhere in text
        try:
            next_buttons = self.driver.find_elements(By.XPATH, "//a[contains(translate(text(), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'next page')]")
            if next_buttons:
                for button in next_buttons:
                    if button.is_displayed():
                        print(f"Found Next Page button: {button.text}")
                        return button
        except Exception as e:
            print(f"Strategy 2 failed: {e}")
        
        # Strategy 3: Look for elements with "next" in class name
        try:
            next_buttons = self.driver.find_elements(By.XPATH, "//a[contains(@class, 'next') or contains(@class, 'Next')]")
            for button in next_buttons:
                if button.is_displayed() and "disabled" not in button.get_attribute("class").lower():
                    print(f"Found next button by class: {button.text} (class: {button.get_attribute('class')})")
                    return button
        except Exception as e:
            print(f"Strategy 3 failed: {e}")
        
        # Strategy 4: Look for pagination arrows
        try:
            arrow_selectors = [
                "//a[contains(text(), '›')]",
                "//a[contains(text(), '→')]", 
                "//a[contains(text(), '>')]",
                "//a[contains(text(), 'next')]",
                "//a[contains(text(), 'Next')]"
            ]
            for selector in arrow_selectors:
                buttons = self.driver.find_elements(By.XPATH, selector)
                for button in buttons:
                    if button.is_displayed() and "disabled" not in button.get_attribute("class").lower():
                        print(f"Found arrow/next button: '{button.text}' (href: {button.get_attribute('href')})")
                        return button
        except Exception as e:
            print(f"Strategy 4 failed: {e}")
        
        # Strategy 5: Look within footer or pagination containers
        try:
            pagination_containers = self.driver.find_elements(By.CSS_SELECTOR, 
                "#hfs-footer, .pagination, .pager, .page-nav, .footer, footer, [class*='pagination'], [id*='pagination']")
            for container in pagination_containers:
                if container.is_displayed():
                    links = container.find_elements(By.TAG_NAME, "a")
                    for link in links:
                        link_text = link.text.strip().lower()
                        if ("next" in link_text and "page" in link_text) or link_text == "next page":
                            print(f"Found next button in container: {link.text}")
                            return link
        except Exception as e:
            print(f"Strategy 5 failed: {e}")
        
        # Strategy 6: JavaScript-based comprehensive search
        try:
            next_button = self.driver.execute_script("""
                // Look for all links
                const allLinks = Array.from(document.querySelectorAll('a'));
                console.log('Total links found:', allLinks.length);
                
                // Filter for potential next buttons
                const candidates = allLinks.filter(link => {
                    const text = link.textContent.toLowerCase().trim();
                    const href = link.href || '';
                    const className = link.className.toLowerCase();
                    
                    // Check if visible
                    if (link.offsetParent === null) return false;
                    
                    // Look for next page indicators
                    return (
                        text.includes('next page') || 
                        text === 'next' ||
                        text === '›' ||
                        text === '→' ||
                        className.includes('next') ||
                        (href.includes('page') && text.includes('next'))
                    );
                });
                
                console.log('Candidates found:', candidates.map(c => ({text: c.textContent.trim(), href: c.href, class: c.className})));
                
                // Return the first viable candidate
                return candidates.length > 0 ? candidates[0] : null;
            """)
            
            if next_button:
                print(f"Found next button via JavaScript: {self.driver.execute_script('return arguments[0].textContent.trim()', next_button)}")
                return next_button
        except Exception as e:
            print(f"Strategy 6 failed: {e}")
        
        # Strategy 7: Look for any link that might navigate to next page based on current URL
        try:
            current_url = self.driver.current_url
            print(f"Current URL: {current_url}")
            
            # Look for links that might be next page based on URL patterns
            all_links = self.driver.find_elements(By.TAG_NAME, "a")
            for link in all_links:
                href = link.get_attribute("href")
                if href and link.is_displayed():
                    # Check if this could be a next page link
                    if ("page" in href.lower() or "offset" in href.lower()) and href != current_url:
                        link_text = link.text.strip()
                        if link_text and not link_text.isdigit() and len(link_text) < 20:
                            print(f"Potential next page link: '{link_text}' -> {href}")
                            return link
        except Exception as e:
            print(f"Strategy 7 failed: {e}")
        
        print("No next button found with any strategy")
        return None

    def _click_next_button_safe(self):
        """Safely click the next button by finding it fresh each time"""
        max_attempts = 3
        original_url = self.driver.current_url
        
        for attempt in range(max_attempts):
            try:
                print(f"Safe click attempt {attempt + 1}")
                
                # Find the button fresh each time to avoid stale references
                next_button = self._find_next_button()
                if not next_button:
                    print("Next button not found on fresh search")
                    return False
                
                print(f"Found fresh button: '{next_button.text}' with href: {next_button.get_attribute('href')}")
                
                # Scroll into view and remove overlays
                self.driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", next_button)
                time.sleep(1)
                self._remove_overlays()
                
                # Try different click methods
                if attempt == 0:
                    print("Trying JavaScript click...")
                    self.driver.execute_script("arguments[0].click();", next_button)
                elif attempt == 1:
                    print("Trying regular click...")
                    next_button.click()
                else:
                    print("Trying ActionChains click...")
                    actions = ActionChains(self.driver)
                    actions.move_to_element(next_button).click().perform()
                
                # Wait and check for navigation
                time.sleep(3)
                new_url = self.driver.current_url
                
                if new_url != original_url:
                    print("✓ URL changed - navigation successful!")
                    return True
                else:
                    print("× URL didn't change")
                    # Check if content changed even if URL stayed same
                    try:
                        WebDriverWait(self.driver, 2).until(
                            EC.staleness_of(self.driver.find_elements(By.CLASS_NAME, "queryly_item_row")[0])
                        )
                        print("✓ Content refreshed!")
                        return True
                    except:
                        print("× Content didn't change either")
                        continue
                        
            except Exception as e:
                print(f"Safe click attempt {attempt + 1} failed: {e}")
                continue
        
        return False

    def _navigate_with_arrows(self):
        """Use arrow keys to navigate to next page"""
        try:
            print("Trying Right Arrow key navigation...")
            
            # Focus on the body or a safe element
            body = self.driver.find_element(By.TAG_NAME, "body")
            body.click()
            
            # Try Right Arrow key
            actions = ActionChains(self.driver)
            actions.send_keys(Keys.ARROW_RIGHT).perform()
            time.sleep(2)
            
            # Check if page changed
            try:
                WebDriverWait(self.driver, 3).until(
                    EC.staleness_of(self.driver.find_elements(By.CLASS_NAME, "queryly_item_row")[0])
                )
                print("✓ Right arrow worked!")
                return True
            except:
                pass
            
            # Try Page Down + Right Arrow
            print("Trying Page Down + Right Arrow...")
            actions = ActionChains(self.driver)
            actions.send_keys(Keys.PAGE_DOWN).perform()
            time.sleep(1)
            actions.send_keys(Keys.ARROW_RIGHT).perform()
            time.sleep(2)
            
            try:
                WebDriverWait(self.driver, 3).until(
                    EC.staleness_of(self.driver.find_elements(By.CLASS_NAME, "queryly_item_row")[0])
                )
                print("✓ Page Down + Right arrow worked!")
                return True
            except:
                pass
            
            # Try End key + Right Arrow (to get to pagination area)
            print("Trying End + Right Arrow...")
            actions = ActionChains(self.driver)
            actions.send_keys(Keys.END).perform()
            time.sleep(1)
            actions.send_keys(Keys.ARROW_RIGHT).perform()
            time.sleep(2)
            
            try:
                WebDriverWait(self.driver, 3).until(
                    EC.staleness_of(self.driver.find_elements(By.CLASS_NAME, "queryly_item_row")[0])
                )
                print("✓ End + Right arrow worked!")
                return True
            except:
                pass
                
        except Exception as e:
            print(f"Arrow navigation failed: {e}")
        
        return False

    def _navigate_with_keyboard(self):
        """Try keyboard navigation methods"""
        try:
            print("Trying Tab + Enter navigation...")
            
            # Focus on body
            body = self.driver.find_element(By.TAG_NAME, "body")
            body.click()
            
            # Try tabbing to next button and pressing enter
            original_elements = self.driver.find_elements(By.CLASS_NAME, "queryly_item_row")
            
            # Tab multiple times to try to reach the next button
            actions = ActionChains(self.driver)
            for i in range(10):  # Tab up to 10 times
                actions.send_keys(Keys.TAB).perform()
                time.sleep(0.5)
                
                # Try pressing Enter
                actions.send_keys(Keys.ENTER).perform()
                time.sleep(2)
                
                # Check if navigation occurred
                try:
                    WebDriverWait(self.driver, 2).until(
                        EC.staleness_of(original_elements[0])
                    )
                    print(f"✓ Tab + Enter worked after {i+1} tabs!")
                    return True
                except:
                    continue
            
            # Try Space instead of Enter
            print("Trying Tab + Space navigation...")
            for i in range(5):
                actions.send_keys(Keys.TAB).perform()
                time.sleep(0.5)
                actions.send_keys(Keys.SPACE).perform()
                time.sleep(2)
                
                try:
                    WebDriverWait(self.driver, 2).until(
                        EC.staleness_of(original_elements[0])
                    )
                    print(f"✓ Tab + Space worked after {i+1} tabs!")
                    return True
                except:
                    continue
                    
        except Exception as e:
            print(f"Keyboard navigation failed: {e}")
        
        return False

    def _remove_overlays(self):
        """Remove overlays that might intercept clicks"""
        overlay_selectors = [
            "#onetrust-consent-sdk",
            "#onetrust-banner-sdk",
            ".onetrust-pc-dark-filter",
            ".modal-backdrop",
            "#gdpr-consent-tool",
            "#consent_blackbar",
            ".fc-consent-root",
            ".cookie-banner",
            "#cookie-banner",
            ".modal",
            ".popup",
            ".overlay",
            ".ad-overlay"
        ]
        
        for selector in overlay_selectors:
            try:
                overlays = self.driver.find_elements(By.CSS_SELECTOR, selector)
                for overlay in overlays:
                    if overlay.is_displayed():
                        self.driver.execute_script("arguments[0].style.display = 'none';", overlay)
                        self.driver.execute_script("arguments[0].style.visibility = 'hidden';", overlay)
                        self.driver.execute_script("arguments[0].remove();", overlay)
            except Exception:
                pass
        
        # Also try to remove any elements with high z-index that might be overlays
        try:
            self.driver.execute_script("""
                const elements = document.querySelectorAll('*');
                for (let el of elements) {
                    const style = window.getComputedStyle(el);
                    if (parseInt(style.zIndex) > 1000 && style.position === 'fixed') {
                        el.style.display = 'none';
                    }
                }
            """)
        except:
            pass

    def _process_page(self):
        """Process the current search results page"""
        articles = self.driver.find_elements(By.CLASS_NAME, "queryly_item_row")
        print(f"Found {len(articles)} articles on this page")
        page_has_valid_articles = False
        articles_scraped_this_page = 0
        
        for article in articles:
            try:
                url_element = article.find_element(By.TAG_NAME, "a")
                url = url_element.get_attribute("href")
                
                if url in self.scraped_urls:
                    print(f"Skipping already scraped URL: {url}")
                    continue
                
                date_element = article.find_element(By.CLASS_NAME, "date")
                date_text = date_element.text.strip()
                
                try:
                    article_date = datetime.datetime.strptime(date_text, "%b %d, %Y")
                except ValueError:
                    try:
                        article_date = datetime.datetime.strptime(date_text, "%B %d, %Y")
                    except ValueError:
                        print(f"Could not parse date: {date_text}")
                        article_date = None
                
                if article_date:
                    # Check if article is too new
                    if self.end_date and article_date > self.end_date:
                        print(f"Article too new ({article_date}), skipping")
                        continue
                    
                    # Check if article is too old
                    if self.start_date and article_date < self.start_date:
                        print(f"Article too old ({article_date}), stopping page processing")
                        return False
                
                title_element = article.find_element(By.TAG_NAME, "h3")
                title = title_element.text.strip()
                
                print(f"Processing article: {title[:50]}... (Date: {article_date})")
                
                author, content = self._get_article_details(url)
                
                # Add the article to our results
                self.results.loc[len(self.results)] = {
                    'title': title,
                    'date': article_date,
                    'author': author,
                    'url': url,
                    'content': content
                }
                
                # Add URL to set of scraped URLs
                self.scraped_urls.add(url)
                
                # Immediately save after each article
                self.save_results(self.filename)
                
                page_has_valid_articles = True
                articles_scraped_this_page += 1
                print(f"Successfully scraped article: {title[:50]}... (Date: {article_date})")
                
            except Exception as e:
                print(f"Error processing article: {e}")
                continue
        
        print(f"Scraped {articles_scraped_this_page} articles from this page")
        return page_has_valid_articles

    def _get_article_details(self, url):
        """Visit an article page and extract the author and content"""
        self.driver.execute_script("window.open('');")
        self.driver.switch_to.window(self.driver.window_handles[1])
        
        try:
            self.driver.get(url)
            self.wait.until(EC.presence_of_element_located((By.TAG_NAME, "article")))
            self.handle_cookie_consent()
            
            author = "Unknown"
            try:
                byline_elements = self.driver.find_elements(By.CSS_SELECTOR, "span.byline-name")
                if byline_elements:
                    author = byline_elements[0].text.strip()
                    
                if author == "Unknown":
                    author_links = self.driver.find_elements(By.CSS_SELECTOR, "a[href*='/author/']")
                    if author_links:
                        author = author_links[0].text.strip()
                        
                if author == "Unknown":
                    author_selectors = [
                        ".article-inline-byline .byline-name",
                        ".article-byline-author",
                        ".article-body__date-source",
                        ".byline",
                        "div[class*='byline']",
                        ".author-name"
                    ]
                    for selector in author_selectors:
                        author_elements = self.driver.find_elements(By.CSS_SELECTOR, selector)
                        if author_elements:
                            auth_txt = author_elements[0].text.strip()
                            match = re.search(r'By\s+(.+)', auth_txt)
                            author = match.group(1).strip() if match else auth_txt
                            break
            except Exception:
                pass
            
            content = ""
            try:
                content_selectors = [
                    ".body-graf",
                    ".article-body__content p",
                    ".article p",
                    ".entry-content p"
                ]
                for selector in content_selectors:
                    elems = self.driver.find_elements(By.CSS_SELECTOR, selector)
                    if elems:
                        content = "\n".join([e.text for e in elems if e.text])
                        if content:
                            break
            except Exception:
                pass
                
            return author, content
            
        except Exception as e:
            print(f"Error getting article details: {e}")
            return "Unknown", ""
        finally:
            self.driver.close()
            self.driver.switch_to.window(self.driver.window_handles[0])

    def load_existing_results(self, filename="nbc_news_china_articles.json"):
        """Load existing results from a JSON file to avoid duplicates"""
        try:
            if os.path.exists(filename):
                existing_data = pd.read_json(filename)
                if 'url' in existing_data.columns:
                    self.scraped_urls.update(existing_data['url'].tolist())
                self.results = pd.concat([self.results, existing_data], ignore_index=True)
                print(f"Loaded {len(existing_data)} existing articles from {filename}")
                return True
        except Exception as e:
            print(f"Error loading existing results: {e}")
        return False

    def save_results(self, filename="nbc_news_china_articles.json"):
        """Save the results to a JSON file"""
        if len(self.results) > 0:
            results_copy = self.results.copy()
            results_copy['date'] = results_copy['date'].astype(str)
            results_copy = results_copy.drop_duplicates(subset=['url'])
            results_copy.to_json(filename, orient='records', indent=2)
            print(f"Saved {len(results_copy)} articles to {filename}")
        else:
            print("No articles to save")

    def _debug_pagination_structure(self):
        """Debug the pagination structure on the current page"""
        try:
            print("\n=== PAGINATION DEBUG ===")
            
            # Get all links and their text
            all_links = self.driver.find_elements(By.TAG_NAME, "a")
            pagination_links = []
            
            for link in all_links:
                try:
                    text = link.text.strip()
                    href = link.get_attribute("href")
                    class_name = link.get_attribute("class")
                    visible = link.is_displayed()
                    
                    # Filter for potential pagination links
                    if (text and ("next" in text.lower() or "page" in text.lower() or 
                                 "previous" in text.lower() or text in ["›", "‹", ">", "<"] or
                                 text.isdigit())) or "page" in href.lower():
                        pagination_links.append({
                            'text': text,
                            'href': href,
                            'class': class_name,
                            'visible': visible
                        })
                except:
                    continue
            
            print(f"Found {len(pagination_links)} potential pagination links:")
            for i, link in enumerate(pagination_links):
                print(f"  {i+1}. Text: '{link['text']}' | Class: '{link['class']}' | Visible: {link['visible']} | Href: {link['href']}")
            
            # Also check footer specifically
            try:
                footer = self.driver.find_element(By.ID, "hfs-footer")
                footer_links = footer.find_elements(By.TAG_NAME, "a")
                print(f"\nFooter contains {len(footer_links)} links:")
                for i, link in enumerate(footer_links):
                    try:
                        text = link.text.strip()
                        if text and len(text) < 50:  # Don't print very long texts
                            print(f"  Footer link {i+1}: '{text}'")
                    except:
                        pass
            except:
                print("No footer found or couldn't access footer")
            
            print("=== END PAGINATION DEBUG ===\n")
            
        except Exception as e:
            print(f"Error in pagination debug: {e}")

# Usage example
if __name__ == "__main__":
    scraper = NBCNewsScraper(headless=False)
    start_date = "2022-05-05"
    end_date = "2025-05-11"
    try:
        results = scraper.search_nbc_news(query="china", start_date=start_date, end_date=end_date, max_pages=10)
        print(f"Scraped {len(results)} articles")
    finally:
        del scraper