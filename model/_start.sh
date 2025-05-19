#!/bin/bash

SCRIPT_DIR="/home/lipatovdn/smiles_to_iupac2.0/model"
SCRIPT_NAME="__train🚀.py"
MONITOR_LOG="$SCRIPT_DIR/monitor.log"
while true; do
  cd "$SCRIPT_DIR" || { echo "Ошибка: не удалось перейти в директорию $SCRIPT_DIR"; exit 1; }
  python3 "$SCRIPT_NAME" > script.log 2>&1 &
  PID=$!
  wait "$PID"
  RETURN_CODE=$?
  echo "$(date) - Скрипт $SCRIPT_NAME завершился с кодом возврата: $RETURN_CODE" >> "$MONITOR_LOG"
  if [ $RETURN_CODE -ne 0 ]; then
    echo "$(date) - Скрипт $SCRIPT_NAME вылетел. Перезапуск через 1 секунду..." >> "$MONITOR_LOG"
    sleep 1
  else
    echo "$(date) - Скрипт $SCRIPT_NAME успешно завершился. Завершение мониторинга." >> "$MONITOR_LOG"
    break 
  fi
done