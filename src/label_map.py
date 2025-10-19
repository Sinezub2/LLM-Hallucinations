CATEGORY_TO_LABEL = {
    "brainstorming": "ANSWER_PREFER",
    "creative_writing": "ANSWER_PREFER",
    "general_qa": "ANSWER_PREFER",
    "open_qa": "ANSWER_PREFER",
    "summarization": "ANSWER_PREFER",   # only if context/source is present
    "classification": "ABSTAIN_PREFER",
    "closed_qa": "ABSTAIN_PREFER",
    "information_extraction": "ABSTAIN_PREFER",
}

ABSTAIN_TRIGGERS = [
    "when did", "what year", "population of",
    "api", "import ", "error:", "exception", "stack trace", "traceback",
    "regulation", "medical", "diagnosis", "legal", "contract", "tax", "finance",
]
