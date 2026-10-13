import re

def normalize_text(text, remove_numbers=False):
    text = str(text).lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    if remove_numbers:
        text = re.sub(r'\d+', '', text)
    text = re.sub(r'\W', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()
