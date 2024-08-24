import os

import openai
from dotenv import load_dotenv


def initialize():
    load_dotenv()
    openai.api_key = os.getenv('OPENAI_API_KEY')
