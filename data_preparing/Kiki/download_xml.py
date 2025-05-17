import os, requests
from tqdm import tqdm
from os.path import basename, join
from bs4 import BeautifulSoup
from _notification import send_msg
from _logger import Logger
from typing import List, Tuple
from concurrent.futures import ThreadPoolExecutor

url: str = 'https://ftp.ncbi.nlm.nih.gov/pubchem/Compound/CURRENT-Full/XML/'
_type = '.xml.gz'
_workers = 5
name_dir = basename('__xml_data__')
path_dir = join(os.getcwd(), name_dir)
os.makedirs(path_dir, exist_ok=True)
logger = Logger(debug=True, namelog='downloading_files')
    

def get_links(url):
    response = requests.get(url)
    response.raise_for_status()  
    soup = BeautifulSoup(response.text, 'html.parser')
    links = []
    for a_tag in soup.find_all('a', href=True):
        href = a_tag['href']
        if href.endswith(_type):
            full_link = (url, href)
            links.append(full_link)
    return links


def download_file(base_url, href):
    file_url = f'{base_url}{href}'
    response = requests.get(file_url, stream=True)
    if response.status_code != 200:
        logger.error(f"Ошибка при скачивании файла {file_url}: {response.status_code}")
        return
    filename = join(path_dir, href)
    total_size = int(response.headers.get('content-length', 0))
    with open(filename, 'wb') as f:
        with tqdm(total=total_size, unit='B', unit_scale=True, desc=href) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                pbar.update(len(chunk))  


def main():
    links: List[Tuple[str, str]] = get_links(url)
    with ThreadPoolExecutor(max_workers=_workers) as executor:
        executor.map(lambda link: download_file(*link), links)


if __name__ == '__main__':
    send_msg('Скачивание началось')
    main()
    send_msg('Файлы скачаны')
    os._exit(0)