from transformers import AutoTokenizer

# Load tokenizer (used if needed for decoding tokens later)
tokenizer = AutoTokenizer.from_pretrained("google/pegasus-xsum")

def postprocess_summary(summary):
    """
    Postprocesses model-generated summary by:
    - Cleaning tokenization artifacts
    - Removing excess whitespace
    - Capitalizing the first letter
    """
    summary = summary.strip()

    # Fix spacing around punctuation
    summary = summary.replace(" .", ".").replace(" ,", ",").replace(" ?", "?")
   
    # Capitalize the first letter
    if summary:
        summary = summary[0].upper() + summary[1:]

    return summary
