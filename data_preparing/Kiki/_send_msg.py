#!/usr/bin/env python3

import matplotlib.pyplot as plt
import telebot, json, sys, os
from threading import Timer
from typing import Dict, Any
import io  

CONFIG_PATH = os.path.expanduser("~/SMILES_to_IUPAC/data_preparing/__property.json")

def load_config() -> Dict[str, Any]:
    try:
        with open(CONFIG_PATH, 'r', encoding='utf-8') as file:
            return json.load(file)
    except FileNotFoundError:
        raise SystemExit(f"Config file not found at {CONFIG_PATH}")
    except json.JSONDecodeError:
        raise SystemExit(f"Invalid JSON in config file {CONFIG_PATH}")

def send_msg(text: str, delete_after=None) -> None:
    config = load_config()
    try:
        bot = telebot.TeleBot(config['token'])
        send_message = bot.send_message(config['CHAT_ID'], text)
        if delete_after is not None:
            t = Timer(delete_after, lambda: bot.delete_message(config['CHAT_ID'], send_message.message_id))
            t.start()
            t.join()
    except KeyError as e:
        raise SystemExit(f"Missing required key in config: {e}")
    except telebot.apihelper.ApiException as e:
        raise SystemExit(f"Telegram API error: {e}")


def send_photo(data: list, *, caption: str = "Training Loss", \
                           xlabel='Iteration', ylabel='Loss', delete_after=None, ylim: tuple =None):
    config = load_config()
    try:
        bot = telebot.TeleBot(config['token'])
        plt.figure()  
        plt.plot(data)
        if ylim:
            plt.ylim(ylim)
        plt.title(caption)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        img_stream = io.BytesIO()  
        plt.savefig(img_stream, format='png')  
        img_stream.seek(0)  
        send_message = bot.send_photo(config['CHAT_ID'], photo=img_stream, caption=caption)
        plt.close()
        if delete_after is not None:
            t = Timer(delete_after, lambda: bot.delete_message(config['CHAT_ID'], send_message.message_id))
            t.start()
            t.join()
    except KeyError as e:
        raise SystemExit(f"Missing required key in config: {e}")
    except telebot.apihelper.ApiException as e:
        raise SystemExit(f"Telegram API error: {e}")
    
    
    

def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("Usage: script.py <message>")
    message = ' '.join(sys.argv[1:])  
    send_msg(f"{message}")

def deb():
    example_data = [0.9, 0.8, 0.6, 0.4, 0.3, 0.2, 0.1, 0.05]
    example_data = list(map(lambda x: x ** 2, example_data))
    send_photo(example_data, caption="Example Training Loss", xlabel='', ylabel='', delete_after=10) 
    
if __name__ == "__main__":
    deb()
    
    
__all__ = ['send_msg', 'send_photo']