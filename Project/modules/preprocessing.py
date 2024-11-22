import re

def clean_description(desc):
    non_facial_keywords = r'\b(shirt|jeans|pants|shoes|hat|scarf|jacket|coat)\b'
    clean_desc = re.sub(non_facial_keywords, '', desc, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", clean_desc).strip()
