from difflib import SequenceMatcher

def is_duplicate(text, existing_texts, threshold=0.70):
    text = str(text or "")
    for old in existing_texts:
        score = SequenceMatcher(None, text.lower(), str(old).lower()).ratio()
        if score >= threshold:
            return True
    return False
