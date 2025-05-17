from bs4 import BeautifulSoup
import requests
import logging 
import os
from tqdm import tqdm


"""
Настройка
"""
url: str = 'https://ftp.ncbi.nlm.nih.gov/pubchem/Compound/CURRENT-Full/SDF/'
logging.basicConfig(level=logging.INFO, format='%(message)s') 
_type = '.sdf.gz'

name_dir = os.path.basename('__archive_data__')
path_dir = os.path.join(os.getcwd(), name_dir)
os.makedirs(path_dir, exist_ok=True)
    

"""
Получаем HTML-код страницы по указанному URL.
"""
def fetch_page(url: str) -> str:
    response: requests.Response = requests.get(url)
    if response.status_code != 200:
        raise Exception(f"Ошибка при получении страницы: {response.status_code}")
    return response.text


"""
Получаем ссылки до конкретных архивов с помощью текущей функции-корутины
"""
def get_links(soup: BeautifulSoup, base_url: str):
    for link in soup.find_all('a'):
        href: str = link.get('href')
        if href and href.endswith(_type):
            yield base_url, href


"""
Скачиваем файл по указанному URL
"""
def download_file(base_url, href):
    file_url = f'{base_url}{href}'
    response = requests.get(file_url, stream=True)
    if response.status_code != 200:
        logging.info(f"Ошибка при скачивании файла {file_url}: {response.status_code}")
        return
        
    filename = os.path.join(path_dir, href)
    total_size = int(response.headers.get('content-length', 0))
    with open(filename, 'wb') as f:
        with tqdm(total=total_size, unit='B', unit_scale=True, desc=href) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                pbar.update(len(chunk))  


"""
Запуск алгоритма
"""
def main():
    page_content = fetch_page(url)
    soup = BeautifulSoup(page_content, 'html.parser')

    for base_url, href in get_links(soup, url):
        print(href)
        # download_file(base_url, href)

if __name__ == '__main__':
    main()
    os._exit(0)