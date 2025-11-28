import google.generativeai as genai  
from dotenv import load_dotenv
import os

load_dotenv()
genai.configure(api_key = os.getenv('GEMINI_API_KEY'))

def get_embedding(text):
    model = 'models/text-embedding-004'
    result = genai.get_embed(model = model, content = text)
    return result['embedding']
