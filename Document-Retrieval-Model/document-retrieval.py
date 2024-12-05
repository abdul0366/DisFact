import os
import aiohttp
import asyncio
from bs4 import BeautifulSoup
import pandas as pd
import random

base_url = 'https://www.fema.gov'
disasters_url = f'{base_url}/disaster/declarations?field_dv2_declaration_date_value%5Bmin%5D=1900&field_dv2_declaration_date_value%5Bmax%5D=2024&field_dv2_declaration_type_value=All&field_dv2_incident_type_target_id_selective=All'
articles_data = []

user_agents = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_13_4) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/66.0.3359.181 Safari/537.36',
    'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/63.0.3239.132 Safari/537.36',
    'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/47.0.2526.106 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:71.0) Gecko/20100101 Firefox/71.0',
]

def get_random_headers():
    return {
        'User-Agent': random.choice(user_agents),
        'Accept-Language': 'en-US,en;q=0.9',
    }

async def fetch(session, url, headers):
    async with session.get(url, headers=headers) as response:
        return await response.text()

async def scrape_article(session, article_url, disaster_id):
    try:
        await asyncio.sleep(random.uniform(1, 10))
        print(f"Scraping article: {article_url}")
        headers = get_random_headers()
        article_page_content = await fetch(session, article_url, headers)
        article_soup = BeautifulSoup(article_page_content, 'html.parser')

        title_tag = article_soup.find('h1', class_='page-title')
        title = title_tag.text.strip() if title_tag else 'No title found'

        date_elem = article_soup.find('time', class_='datetime')
        date = date_elem['datetime'].strip() if date_elem and 'datetime' in date_elem.attrs else 'No date available'

        divs = article_soup.find_all('div', class_='clearfix text-formatted field field--name-body field--type-text-with-summary field--label-hidden field__item')
        full_text = ' '.join([p.get_text(separator=' ', strip=True) for div in divs for p in div.find_all('p')])

        # Capture text from blockquote within span
        blockquote_text = ' '.join([bq.get_text(separator=' ', strip=True) for bq in article_soup.select('span blockquote')])

        full_text = (full_text + ' ' + blockquote_text).strip() if full_text or blockquote_text else 'No text available'

        print(full_text)

        articles_data.append({
            'Disaster ID': disaster_id,
            'Title': title,
            'Date': date,
            'Text': full_text,
            'URL': article_url
        })
        print(f"Finished scraping article: {article_url}")
    except Exception as e:
        print(f"Error scraping {article_url}: {e}")

async def scrape_articles(session, articles_url, disaster_id):
    try:
        await asyncio.sleep(random.uniform(.25, 5))
        print(f"Scraping articles from: {articles_url}")
        headers = get_random_headers()
        page_content = await fetch(session, articles_url, headers)
        soup = BeautifulSoup(page_content, 'html.parser')

        articles = soup.select('.views-field.views-field-title a')
        article_urls = [base_url + article['href'] for article in articles]

        tasks = [scrape_article(session, url, disaster_id) for url in article_urls]
        await asyncio.gather(*tasks)

        print(f"Finished scraping articles from: {articles_url}")
    except Exception as e:
        print(f"Error accessing articles at {articles_url}: {e}")

async def scrape_disaster_page(session, page_num):
    try:
        await asyncio.sleep(random.uniform(.5, 1.9))
        print(f"\n\n\nAccessing disaster declarations page: {page_num}")
        headers = get_random_headers()
        page_content = await fetch(session, f'{disasters_url}&page={page_num}', headers)
        soup = BeautifulSoup(page_content, 'html.parser')

        central_column_links = soup.select('span.field-content.list-view-title a[href^="/disaster/"]')
        disaster_links = [base_url + a['href'] for a in central_column_links]
        print(f"Found {len(central_column_links)} disaster declarations on page {page_num}")

        declaration_counter = 1
        for disaster_link in disaster_links:
            disaster_id = disaster_link.split('/')[-1]
            print(f"\nProcessing disaster declaration number {declaration_counter} on page {page_num}: {disaster_link}")
            await asyncio.sleep(random.uniform(.5, 5))
            print("sleep done!")
            news_media_url = f'{disaster_link}/news-media'
            reports_notices_url = f'{disaster_link}/notices'

            await asyncio.gather(
                scrape_articles(session, news_media_url, disaster_id),
                scrape_articles(session, reports_notices_url, disaster_id)
            )
            declaration_counter += 1
    except Exception as e:
        print(f"Error accessing disaster declarations page {page_num}: {e}\n")

async def main():

    start_page = int(input("Enter start page: "))

    async with aiohttp.ClientSession() as session:
        for page_num in range(start_page, 491): # max = 490
            await scrape_disaster_page(session, page_num)
            if (page_num + 1) % 3 == 0:
                save_data_to_csv(start_page, page_num)
                start_page = page_num + 1

        if articles_data:
            save_data_to_csv(start_page, 490)

def save_data_to_csv(start_page, end_page):
    csv_file_path = f'fema_articles_pages_{start_page}_to_{end_page}.csv'
    df = pd.DataFrame(articles_data)
    df.to_csv(csv_file_path, index=False)
    print(f'Data saved to {csv_file_path}')
    articles_data.clear()

if __name__ == '__main__':
    asyncio.run(main())
    print('Scraping completed.')
