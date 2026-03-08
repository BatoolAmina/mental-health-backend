import re

MANUAL_OVERRIDES = {
    "identity_keywords": ["batool", "person", "human", "student", "user", "name is", "i am a"],
    "high_risk_triggers": ["kill", "die", "suicide", "end my life", "goodbye world", "no more", "harm myself"],
    "positive_triggers": ["happy", "blessed", "joyful", "excited", "amazing", "great", "wonderful"],
    "anxiety_triggers": ["panicking", "scared", "shaking", "chest pain", "fear", "worried"],
    "depression_triggers": ["worthless", "hopeless", "empty", "lonely", "darkness", "no energy"]
}

def apply_logic_rules(text, original_label, original_confidence):
    text_lower = text.lower().strip()
    words = text_lower.split()

    if (any(name in text_lower for name in MANUAL_OVERRIDES["identity_keywords"]) and len(words) <= 4):
        return "Normal", 100.0, "low"

    if any(trigger in text_lower for trigger in MANUAL_OVERRIDES["high_risk_triggers"]):
        return "Suicidal", 99.0, "high"

    if any(trigger in text_lower for trigger in MANUAL_OVERRIDES["positive_triggers"]):
        if original_label in ["depression", "anxiety", "suicidal"]:
            return "Happy", 85.0, "low"

    if any(trigger in text_lower for trigger in MANUAL_OVERRIDES["anxiety_triggers"]):
        if original_label in ["normal", "happy"]:
            return "Anxiety", 90.0, "medium"

    if any(trigger in text_lower for trigger in MANUAL_OVERRIDES["depression_triggers"]):
        if original_label in ["normal", "happy"]:
            return "Depression", 90.0, "medium"

    if original_label == "suicidal" and original_confidence < 60:
        if not any(trigger in text_lower for trigger in MANUAL_OVERRIDES["high_risk_triggers"]):
            return "Normal", 70.0, "low"

    return original_label, original_confidence, None