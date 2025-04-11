import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin

# Configuration
WIKI_PAGE = "https://en.wikipedia.org/wiki/Computer_science"  # Starting page
MAX_IMAGES = 1000  # Target number of images
DOWNLOAD_DIR = "./static/images"  # Directory to save images
os.makedirs(DOWNLOAD_DIR, exist_ok=True)

# Store crawled data (image URL, alt text, caption, surrounding text)
image_data = []

def download_image(url, filename):
    try:
        response = requests.get(url, stream=True, timeout=10)
        if response.status_code == 200:
            with open(os.path.join(DOWNLOAD_DIR, filename), 'wb') as f:
                for chunk in response.iter_content(1024):
                    f.write(chunk)
            return True
    except Exception as e:
        print(f"Failed to download {url}: {e}")
    return False

def extract_textual_surrogates(soup, img_tag):
    # Extract alt text
    alt_text = img_tag.get('alt', '')
    
    # Extract caption (from parent figure or table)
    caption = ""
    parent = img_tag.find_parent(['figure', 'table'])
    if parent:
        figcaption = parent.find('figcaption') or parent.find('caption')
        if figcaption:
            caption = figcaption.get_text(strip=True)
    
    # Extract surrounding text (previous and next siblings)
    surrounding_text = []
    for sibling in img_tag.find_all_previous(string=True, limit=2):
        if sibling.strip():
            surrounding_text.append(sibling.strip())
    for sibling in img_tag.find_all_next(string=True, limit=2):
        if sibling.strip():
            surrounding_text.append(sibling.strip())
    
    return {
        'alt_text': alt_text,
        'caption': caption,
        'surrounding_text': ' '.join(surrounding_text)
    }

def crawl_wikipedia(start_url, max_images):
    visited_pages = set()
    pages_to_visit = [start_url]
    image_count = 0
    
    while pages_to_visit and image_count < max_images:
        current_url = pages_to_visit.pop(0)
        if current_url in visited_pages:
            continue
        
        try:
            print(f"Crawling: {current_url}")
            response = requests.get(current_url, timeout=10)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Find all images on the page
            for img in soup.find_all('img', src=True):
                if image_count >= max_images:
                    break
                
                img_url = urljoin(current_url, img['src'])
                if not img_url.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
                    continue  # Skip non-image files
                
                # Download image
                filename = f"img_{image_count}.{img_url.split('.')[-1]}"
                if download_image(img_url, filename):
                    # Extract textual surrogates
                    surrogates = extract_textual_surrogates(soup, img)
                    image_data.append({
                        'image_url': img_url,
                        'filename': filename,
                        **surrogates
                    })
                    image_count += 1
                    print(f"Downloaded {image_count}/{max_images}: {filename}")
            
            # Add linked Wikipedia pages to crawl queue
            for link in soup.find_all('a', href=True):
                href = link['href']
                if href.startswith('/wiki/') and ':' not in href:  # Avoid non-article pages
                    full_url = urljoin(current_url, href)
                    if full_url not in visited_pages:
                        pages_to_visit.append(full_url)
            
            visited_pages.add(current_url)
        
        except Exception as e:
            print(f"Error crawling {current_url}: {e}")
    
    return image_data

# Run the crawler
if __name__ == "__main__":
    print(f"Starting crawl from {WIKI_PAGE} (target: {MAX_IMAGES} images)")
    results = crawl_wikipedia(WIKI_PAGE, MAX_IMAGES)
    
    # Save metadata (for indexing later)
    import json
    with open("image_metadata.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nDone! Downloaded {len(results)} images. Metadata saved to 'image_metadata.json'.")