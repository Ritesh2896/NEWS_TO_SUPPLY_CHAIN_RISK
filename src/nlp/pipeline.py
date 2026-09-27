from src.relevance.relevance import calculate_relevance
from src.nlp.entity_extraction import extract_entities
from src.nlp.event_extraction import extract_event

def analyze_text(text):
    relevance = calculate_relevance(text)
    entities = extract_entities(text)
    event = extract_event(text)
    return {"relevance": relevance, "entities": entities, "event": event}
