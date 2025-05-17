from setuptools import setup, find_packages

setup(
    name="lithium",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A simple package to send messages and photos to Telegram.",
    packages=find_packages(),
    install_requires=[
        "pyTelegramBotAPI",
        "matplotlib",
    ],
    package_data={
        'lithium': ['__data__/__property.json'],  
    },
        entry_points={
        'console_scripts': [
            'send_telegram = lithium._send_msg:main', 
        ],
    },
    include_package_data=True,
)