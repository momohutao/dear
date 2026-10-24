"""Shared evaluation text formatting; preserve historical inference whitespace."""


def format_eval_document(title, text, separator=" "):
    # An empty title still contributes its separator. Do not strip: whitespace
    # affects token IDs, positions and the passage truncation boundary.
    return "document: " + title + separator + text
