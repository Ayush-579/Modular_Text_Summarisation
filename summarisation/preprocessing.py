import re

def preprocess_dialogue(text):
    """
    Preprocesses dialogue text by:
    - Normalizing casing
    - Removing non-ASCII characters
    - Removing timestamps
    - Replacing speaker tags with special tokens
    - Removing extra spaces
    """
    text = re.sub(r'\[\d{1,2}:\d{2}(?: [APap][Mm])?\]', '', text)  # remove timestamps
    text = re.sub(r'\buser\d*:?', '[USER] ', text)
    text = re.sub(r'\b(agent|customer):?', '[AGENT] ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def preprocess_summary(summary):
    """
    Preprocesses summary text by stripping extra whitespace.
    """
    summary = summary.strip()
    summary = re.sub(r'\s+', ' ', summary)
    return summary
