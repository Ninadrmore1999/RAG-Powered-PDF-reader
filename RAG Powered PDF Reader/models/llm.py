import google.generativeai as genai 
import os
from dotenv import load_dotenv 
from constants import SYSTEM_PROMPT

load_dotenv()
genai.configure(api_key = os.getenv('GEMINI_API_KEY'))

def ask_llm(question, context):
    prompt = f"{SYSTEM_PROMPT}\n\n context:\n{context}\n\n question:\n{question}\n\n"
    model = genai.GenerativeModel('modesl/gemini-pro')
    response = model.generate_content(prompt)
    return response.text 
