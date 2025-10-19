CATEGORY_TO_LABEL = {
    "brainstorming": "ANSWER_PREFER",
    "creative_writing": "ANSWER_PREFER",
    "general_qa": "ANSWER_PREFER",
    "open_qa": "ANSWER_PREFER",
    "summarization": "ANSWER_PREFER",   # if context present; else ABSTAIN in make_labels
    "classification": "ABSTAIN_PREFER",
    "closed_qa": "ABSTAIN_PREFER",
    "information_extraction": "ABSTAIN_PREFER",
}
