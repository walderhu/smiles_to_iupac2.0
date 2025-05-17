#!/bin/bash

# Цвета для вывода
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Инициализация
clear
echo -e "${BLUE}=== СИСТЕМНЫЙ МОНИТОР ===${NC}"

# Функция для расчета памяти пользователя
user_memory() {
    ps -u $USER -o %mem --no-headers | awk '{sum += $1} END {print sum}'
}

# Основной цикл
while true; do
    # Сохраняем позицию курсора
    tput sc
    
    # Перемещаем курсор на начало вывода данных
    tput cup 1 0
    
    # 1. CPU (общая загрузка)
    cpu_usage=$(top -bn1 | grep "Cpu(s)" | awk '{print 100 - $8"%"}')
    echo -ne "${YELLOW}CPU общая загрузка:${NC} $cpu_usage"
    tput el
    
    # 2. RAM (использование)
    # Общая память системы
    ram_total=$(free -m | awk '/Mem:/ {print $2}')
    ram_used=$(free -m | awk '/Mem:/ {print $3}')
    ram_percent=$((ram_used*100/ram_total))
    echo -ne "\n${YELLOW}RAM система:${NC} $ram_percent% (${ram_used}M из ${ram_total}M)"
    tput el
    
    # Память пользователя
    user_mem_percent=$(user_memory)
    user_mem_used=$(echo "$ram_total * $user_mem_percent / 100" | bc)
    echo -ne "\n${YELLOW}RAM ваши процессы:${NC} ${user_mem_percent}% (~${user_mem_used}M)"
    tput el
    
    # 3. Ваши процессы (TOP 5 по CPU)
    echo -ne "\n\n${BLUE}=== ВАШИ ПРОЦЕССЫ (TOP 5 по CPU) ===${NC}\n"
    ps -u $USER -o pid,%cpu,%mem,cmd --sort=-%cpu | head -n 6 | awk 'NR<=6 {printf "%-8s %-8s %-8s %s\n", $1, $2"%", $3"%", $4}' | while read -r line; do
        echo -ne "$line"
        tput el
        echo ""
    done
    
    # 4. GPU информация
    echo -ne "\n${BLUE}=== GPU ИНФОРМАЦИЯ ===${NC}\n"
    if command -v nvidia-smi &>/dev/null; then
        gpu_info=$(nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits)
        gpu_usage=$(echo $gpu_info | awk -F',' '{print $1"%"}')
        gpu_mem_used=$(echo $gpu_info | awk -F',' '{print $2/1024}' | xargs printf "%.1f")
        gpu_mem_total=$(echo $gpu_info | awk -F',' '{print $3/1024}' | xargs printf "%.1f")
        gpu_mem_percent=$(echo $gpu_info | awk -F',' '{printf "%.1f", ($2/$3)*100}')
        
        echo -ne "${YELLOW}GPU загрузка:${NC} $gpu_usage"
        tput el
        echo -ne "\n${YELLOW}GPU память:${NC} $gpu_mem_percent% (${gpu_mem_used}G из ${gpu_mem_total}G)"
        tput el
    else
        echo -ne "${RED}Информация о GPU недоступна${NC}"
        tput el
    fi
    
    echo -ne "\n\n${GREEN}Последнее обновление: $(date)${NC}"
    tput el
    
    # Чтение ввода без ожидания (таймаут 1 секунда)
    if read -t 1 -n 1 input; then
        if [[ $input = "q" ]] || [[ $input = "Q" ]]; then
            tput cup 20 0  # Просто переходим вниз
            clear
            echo -e "${GREEN}Мониторинг завершен${NC}"
            tput cnorm
            break
        fi
    fi
    
    # Восстанавливаем позицию для следующего обновления
    tput rc
done