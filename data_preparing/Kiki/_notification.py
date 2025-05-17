import telebot, json

def send_msg(text):
    with open('__property.json', 'r', encoding='utf-8') as file:
        data = json.load(file)
    bot = telebot.TeleBot(data['token'])
    bot.send_message(data['CHAT_ID'], text)
    
__all__ = ['send_msg']