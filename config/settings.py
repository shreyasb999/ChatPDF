# config/settings.py

import os
from dotenv import load_dotenv
import unstructured_pytesseract.pytesseract as pytesseract

load_dotenv()

# API Keys (if you ever integrate LLM or embeddings)
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
# GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY", "")
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY", "")

# Default PDF (you can upload a new one in the UI)
PDF_FILE = os.path.join(os.path.dirname(__file__), "..", "data", "attention.pdf")

# Windows example (adjust to your env’s “Library\bin”):
pytesseract.tesseract_cmd = r"D:\Study\Generative AI\ChatPDF\.conda\Library\bin\tesseract.exe"
