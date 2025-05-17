```bash
screen -dmS logging_data python3 "/home/lipatovdn/SMILES_to_IUPAC/data_preparing/process_data.py"
screen -dmS encode python3 "/home/lipatovdn/SMILES_to_IUPAC/data_preparing/_jam_encode.py"
```

УДАЛИТЬ СТОППЕРЫ

то есть мы берем преферед июпак и абсолютный смайлс
если к концу недели ситуация с сервером не изменится то будем поступать так
обрабатывать файл и как только он обработался удалять одноименный архив который обрабатывали

не забыть потом удалить оттуда все дубликаты и взять подсчеты

```bash
scp -F absol_path/config karamanovada@sodium:path_to_file desktop
scp -F C:\Users\tru60\denis\config.txt lipatovdn@sodium:/home/lipatovdn/SMILES_to_IUPAC/data_preparing/jambalaya.rest .
```

https://github.com/google/sentencepiece


scp -r -F C:\Users\tru60\denis\config.txt "C:\Users\tru60\OneDrive\Рабочий стол\models"  hydrogen:/home/lipatovdn/SMILES_to_IUPAC/model/a_model/__web_api/
scp -F C:\Users\tru60\denis\config.txt sodium:~/.gitconfig  hydrogen:~/.gitconfig


~/.bashrc 



conda env create -f environment.yml #Без rmolencoder
conda activate iupac
pip install rmolencoder

pip install rmolencoder --no-dependencies #(очень НЕ рекомендуется) После активации окружения