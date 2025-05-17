#!/usr/bin/env python3

import matplotlib.pyplot as plt
import telebot
import json
import sys
from threading import Timer
from typing import Dict, Any
import io
import importlib.resources  


def load_config() -> Dict[str, Any]:
    try:
        config_path = importlib.resources.files('lithium').joinpath('__data__/__property.json')
        with config_path.open('r', encoding='utf-8') as file:  # Open the file object
            return json.load(file)
    except FileNotFoundError:
        raise SystemExit(f"Config file not found in package: __data__/__property.json")
    except json.JSONDecodeError:
        raise SystemExit(f"Invalid JSON in config file: __data__/__property.json")

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


def send_photo(data: list, *, caption: str = "Training Loss",
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


if __name__ == "__main__":
    main()

__all__ = ['send_msg', 'send_photo']