#!/bin/bash

SCRIPT_DIR="/home/lipatovdn/smiles_to_iupac2.0/model"
SCRIPT_NAME="__train🚀.py"

while true; do
  cd "$SCRIPT_DIR" || {
    echo "Ошибка: не удалось перейти в директорию $SCRIPT_DIR"
    exit 1 
  }

  # Запускаем скрипт и перенаправляем вывод в файл (можно убрать перенаправление)
  python3 "$SCRIPT_NAME" # > output.log 2>&1

  # Проверяем код возврата.  Если скрипт завершился с ошибкой, можно добавить обработку.
  if [ $? -ne 0 ]; then
      echo "Скрипт $SCRIPT_NAME завершился с ошибкой.  Код возврата: $?"
      sleep 1
  fi

  echo "Скрипт $SCRIPT_NAME перезапускается..."

done