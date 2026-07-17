import asyncio
import aiohttp
from bs4 import BeautifulSoup
import os
import zipfile
import io
import sys
import csv
from urllib.parse import urljoin

BASE_URL = "https://www.1001fonts.com/"
# Note: To get all 60,000, we paginate the homepage which goes up to 5000+ pages.
START_URL = "https://www.1001fonts.com/?page={}"

TARGET_DIR = r"e:\New folder\coding_arc\Font_Identifier_AI\1001_fonts"
CSV_FILE = r"e:\New folder\coding_arc\Font_Identifier_AI\downloaded_fonts.csv"
CONCURRENCY = 15 # Download 15 fonts concurrently

csv_lock = asyncio.Lock()

def load_downloaded_urls():
    downloaded = set()
    if os.path.exists(CSV_FILE):
        with open(CSV_FILE, mode='r', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader, None) # Skip header
            for row in reader:
                if row:
                    downloaded.add(row[0])
    return downloaded

async def save_downloaded_url(url):
    async with csv_lock:
        with open(CSV_FILE, mode='a', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([url])

async def fetch_page(session, url, retries=3):
    for attempt in range(retries):
        try:
            async with session.get(url, timeout=20) as response:
                if response.status == 200:
                    return await response.text()
                elif response.status == 429:
                    print(f"\n[!] Rate limited on {url}. Waiting 10 seconds...")
                    await asyncio.sleep(10)
                else:
                    print(f"\n[!] Error {response.status} on {url}")
        except Exception as e:
            print(f"\n[!] Exception {e} on {url}. Attempt {attempt+1}/{retries}")
            await asyncio.sleep(5)
    return None

async def download_worker(name, session, queue, sem):
    while True:
        url = await queue.get()
        success = False
        async with sem:
            try:
                async with session.get(url, timeout=30) as response:
                    if response.status == 200:
                        content = await response.read()
                        try:
                            with zipfile.ZipFile(io.BytesIO(content)) as zip_ref:
                                for file_info in zip_ref.infolist():
                                    if file_info.filename.lower().endswith(('.ttf', '.otf')):
                                        filename = os.path.basename(file_info.filename)
                                        if filename:
                                            target_path = os.path.join(TARGET_DIR, filename)
                                            if not os.path.exists(target_path):
                                                with open(target_path, 'wb') as f:
                                                    f.write(zip_ref.read(file_info.filename))
                                                sys.stdout.write(f"\rDownloaded {filename}" + " " * 20)
                                                sys.stdout.flush()
                            success = True
                        except zipfile.BadZipFile:
                            pass
            except Exception as e:
                pass
        
        # If successfully downloaded and extracted, log it in CSV so we never download it again
        if success:
            await save_downloaded_url(url)
            
        queue.task_done()

async def main():
    os.makedirs(TARGET_DIR, exist_ok=True)
    
    seen_links = load_downloaded_urls()
    print(f"Loaded {len(seen_links)} previously downloaded URLs.")
    
    # Initialize CSV if it doesn't exist
    if not os.path.exists(CSV_FILE):
        with open(CSV_FILE, mode='w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["download_url"])
            
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    conn = aiohttp.TCPConnector(limit=50)
    sem = asyncio.Semaphore(CONCURRENCY)
    queue = asyncio.Queue()
    
    async with aiohttp.ClientSession(headers=headers, connector=conn) as session:
        # Start consumers
        consumers = []
        for i in range(CONCURRENCY):
            task = asyncio.create_task(download_worker(f"Worker-{i}", session, queue, sem))
            consumers.append(task)
            
        print("Scraper started. Scraping pages and downloading simultaneously...")
        page = 1
        
        while True:
            url = START_URL.format(page)
            html = await fetch_page(session, url)
            if not html:
                print(f"\n[!] Failed to fetch page {page} after retries. The server might be blocking us. Stopping discovery phase.")
                break
            
            soup = BeautifulSoup(html, 'html.parser')
            links = soup.find_all('a', class_='btn-download')
            
            new_links = 0
            for link in links:
                href = link.get('href')
                if href and href.startswith('/download/'):
                    full_url = urljoin(BASE_URL, href)
                    if full_url not in seen_links:
                        seen_links.add(full_url)
                        queue.put_nowait(full_url)
                        new_links += 1
            
            sys.stdout.write(f"\rScanned page {page} | Queue size: {queue.qsize()} | Total unique links: {len(seen_links)}" + " " * 10)
            sys.stdout.flush()
            
            if len(links) == 0:
                print(f"\nNo download links found at page {page}. Reached the end! Waiting for queue to finish...")
                break
                
            page += 1
            
        # Wait until the queue is fully processed
        await queue.join()
        
        # Cancel consumers
        for c in consumers:
            c.cancel()
            
        print("\nAll fonts downloaded successfully!")

if __name__ == '__main__':
    asyncio.run(main())
